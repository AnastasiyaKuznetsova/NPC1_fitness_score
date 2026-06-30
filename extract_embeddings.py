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
    "evo2_1b_base":  {"num_layers": 25, "params": "1B"},
    "evo2_7b":       {"num_layers": 32, "params": "7B"},
    "evo2_40b":      {"num_layers": 50, "params": "40B"}
}


def last_layer_name(layer: str, num_layers: int) -> str:
    """Replace the block index in a layer name with the last block index."""
    last_idx = num_layers - 1
    return re.sub(r"(blocks\.)(\d+)(\.)", rf"\g<1>{last_idx}\3", layer, count=1)


def layer_index(layer: str) -> str:
    """Extract block index from a layer name like 'blocks.28.mlp.l3' -> '28'."""
    m = re.search(r"blocks\.(\d+)\.", layer)
    return m.group(1) if m else layer

# mut/ref, model, params billions/millions, mode mean/last
def build_filename(seq_type: str, model_name: str, params: str, layer: str, mode: str, strand: str) -> str:
    """Build output filename: {seq_type}_Evo2_{params}_L{n_layer}_{mode}.npy"""
    return f"{seq_type}_Evo2_{params}_L{layer_index(layer)}_{mode}_{strand}.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Evo2 embeddings from DNA/RNA sequences.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_ID,
        help=f"Evo2 model variant to load (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--layer", default=DEFAULT_LAYER_NAME,
        help=f"Layer name to extract hidden states from (default: {DEFAULT_LAYER_NAME}); ignored when --last is set",
    )
    parser.add_argument(
        "--seq-type", default="DNA", choices=["DNA", "RNA"],
        help="Sequence type label used in output filename (default: DNA)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Sequences per forward pass (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--no-average", dest="average", action="store_false",
        help="Return per-token embeddings instead of sequence-averaged embeddings",
    )
    parser.add_argument(
        "--last", action="store_true",
        help="Extract the last (non-padding) token hidden state from the last layer",
    )
    return parser.parse_args()


def extract_embeddings(
    sequences: list[str],
    model,
    df: str,
    average: bool,
    last: bool,
    layer: str,
    seq_type: str,
    strand: str, 
    params: str,
    batch_size: int = BATCH_SIZE,
    pad_id: int = PAD_ID,
) -> np.ndarray:  # type: ignore
    """
    Pooling modes (mutually exclusive, checked by caller):
        average=True  -> mean over non-padding tokens, shape (N, D)
        last=True     -> last non-padding token from last layer, shape (N, D)
        both False    -> full token sequences, saved per batch
    """
    os.makedirs("embeddings", exist_ok=True)
    all_pooled = []
    mode = "mean" if average else "last" if last else "seq"

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

            if average:
                mask = torch.zeros((hidden.shape[0], hidden.shape[1]), dtype=torch.bool, device=DEVICE)
                for b, l in enumerate(lengths):
                    mask[b, :l] = True
                mask_expanded = mask.unsqueeze(-1).float()  # B, L, 1
                pooled = (hidden * mask_expanded).sum(1) / mask_expanded.sum(1)
            elif last:
                last_indices = torch.tensor([l - 1 for l in lengths], device=DEVICE)
                pooled = hidden[torch.arange(hidden.shape[0], device=DEVICE), last_indices]  # B, D
            else:
                pooled = hidden

        print(f"Processed {min(start + batch_size, len(sequences))}/{len(sequences)} sequences")

        if mode == "seq":
            fname = build_filename(seq_type, "Evo2", params, layer, mode, strand) 
            np.save(f"embeddings/{df}_{fname}_{i}", pooled.float().cpu().numpy())
        else:
            all_pooled.append(pooled.float().cpu().numpy())

    if all_pooled:
        combined = np.concatenate(all_pooled, axis=0)
        fname = build_filename(seq_type, "Evo2", params, layer, mode, strand)
        np.save(f"embeddings/{df}_{fname}", combined)
        print(f"Saved {mode} embeddings: {combined.shape} -> {fname}")


if __name__ == "__main__":
    args = parse_args()

    if not args.average and args.last:
        raise ValueError("--last and --no-average are mutually exclusive")

    cfg = MODEL_CONFIG.get(args.model)
    if cfg is None:
        raise ValueError(f"Unknown model {args.model!r}. Add it to MODEL_CONFIG or check the name.")

    # --last always uses the final layer regardless of --layer
    layer = last_layer_name(args.layer, cfg["num_layers"]) if args.last else args.layer
    if args.last:
        print(f"[last-token mode] using last layer: {layer}")

    print(f"Loading {args.model} ...")
    model = Evo2(args.model)
    print("Model loaded.\n")

    for df in ["ref_seq", "mut_seq"]:
        seqs = np.load(f"output/{df}_DNA.npy")
        extract_embeddings(
            sequences=seqs,
            model=model,
            df=df,
            layer=layer,
            seq_type=args.seq_type,
            params=cfg["params"],
            batch_size=args.batch_size,
            average=args.average,
            last=args.last,
            strand=args.strand
        )
