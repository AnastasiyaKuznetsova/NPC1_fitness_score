import argparse
import os
import re

import torch
import numpy as np
from evo2 import Evo2

DEFAULT_MODEL_ID = "evo2_7b"
DEFAULT_LAYER_NAME = "blocks.28.mlp.l3"
PAD_ID = 0
BATCH_SIZE = 1
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# num_layers and human-readable parameter count per model variant
MODEL_CONFIG = {
    "evo2_1b_base": {"num_layers": 25, "params": "1B"},
    "evo2_7b":      {"num_layers": 32, "params": "7B"},
    "evo2_40b":     {"num_layers": 50, "params": "40B"},
}


def last_layer_name(layer: str, num_layers: int) -> str:
    """Replace the block index in a layer name with the last block index."""
    last_idx = num_layers - 1
    return re.sub(r"(blocks\.)(\d+)(\.)", rf"\g<1>{last_idx}\3", layer, count=1)


def layer_index(layer: str) -> str:
    """Extract block index from a layer name like 'blocks.28.mlp.l3' -> '28'."""
    m = re.search(r"blocks\.(\d+)\.", layer)
    return m.group(1) if m else layer


def build_filename(seq_type: str, params: str, layer: str, mode: str, strand: str) -> str:
    """Build output filename: {seq_type}_Evo2_{params}_L{n_layer}_{mode}_{strand}.npy"""
    return f"{seq_type}_Evo2_{params}_L{layer_index(layer)}_{mode}_{strand}.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Evo2 embeddings from DNA/RNA sequences.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_ID,
        help=f"Evo2 model variant to load (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--layer", default=DEFAULT_LAYER_NAME,
        help=f"Layer name to extract hidden states from (default: {DEFAULT_LAYER_NAME}); ignored when --emb-type last",
    )
    parser.add_argument(
        "--seq-type", default="DNA", choices=["DNA", "RNA"],
        help="Sequence type label used in output filename (default: DNA)",
    )
    parser.add_argument(
        "--strand", default="forward", choices=["forward", "reverse"],
        help="Strand direction — selects input file and labels output (default: forward)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Sequences per forward pass (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--emb-type", default="average", choices=["average", "last"],
        help="Pooling strategy: 'average' = mean over non-padding tokens; 'last' = last token from last layer (default: average)",
    )
    return parser.parse_args()


def extract_embeddings(
    sequences: list[str],
    model,
    df: str,
    emb_type: str,
    layer: str,
    seq_type: str,
    strand: str,
    params: str,
    batch_size: int = BATCH_SIZE,
    pad_id: int = PAD_ID,
) -> None:
    """
    Pooling modes (emb_type):
        'average' -> mean over non-padding tokens, shape (N, D)
        'last'    -> last non-padding token from last layer, shape (N, D)
    """
    os.makedirs("embeddings", exist_ok=True)
    all_pooled = []

    for i, start in enumerate(range(0, len(sequences), batch_size)):
        seqs = sequences[start:start + batch_size]
        token_ids = [model.tokenizer.tokenize(seq) for seq in seqs]
        lengths = [len(t) for t in token_ids]
        max_length = max(lengths)
        padded = [t + [pad_id] * (max_length - len(t)) for t in token_ids]
        input_ids = torch.tensor(padded, dtype=torch.int).to(DEVICE)

        with torch.no_grad():
            _, embeddings = model.forward(input_ids, return_embeddings=True, layer_names=[layer])
            hidden = embeddings[layer]  # B, L, D

            if emb_type == "average":
                mask = torch.zeros((hidden.shape[0], hidden.shape[1]), dtype=torch.bool, device=hidden.device)
                for b, l in enumerate(lengths):
                    mask[b, :l] = True
                mask_expanded = mask.unsqueeze(-1).float()  # B, L, 1
                pooled = (hidden * mask_expanded).sum(1) / mask_expanded.sum(1)
            else:  # last
                last_indices = torch.tensor([l - 1 for l in lengths], device=hidden.device)
                pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_indices]  # B, D

        all_pooled.append(pooled.float().cpu().numpy())
        print(f"Processed {min(start + batch_size, len(sequences))}/{len(sequences)} sequences")

    combined = np.concatenate(all_pooled, axis=0)
    fname = build_filename(seq_type, params, layer, emb_type, strand)
    np.save(f"embeddings/{df}_{fname}", combined)
    print(f"Saved {emb_type} embeddings: {combined.shape} -> {df}_{fname}")


if __name__ == "__main__":
    args = parse_args()

    cfg = MODEL_CONFIG.get(args.model)
    if cfg is None:
        raise ValueError(f"Unknown model {args.model!r}. Add it to MODEL_CONFIG or check the name.")

    # last-token mode always uses the final layer
    layer = last_layer_name(args.layer, cfg["num_layers"]) if args.emb_type == "last" else args.layer
    if args.emb_type == "last":
        print(f"[last-token mode] using last layer: {layer}")

    print(f"Loading {args.model} ...")
    model = Evo2(args.model)
    print("Model loaded.\n")

    for df in ["ref_seq", "mut_seq"]:
        seqs = np.load(f"output/{df}_DNA_{args.strand}.npy")
        extract_embeddings(
            sequences=seqs,
            model=model,
            df=df,
            emb_type=args.emb_type,
            layer=layer,
            seq_type=args.seq_type,
            params=cfg["params"],
            strand=args.strand,
            batch_size=args.batch_size,
        )
