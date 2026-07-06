import argparse
import os
import re

import torch
import numpy as np
from evo2 import Evo2

from transformer_engine.common.recipe import _OverrideLinearPrecision, DelayedScaling, Format # type: ignore[attr-defined]
torch.serialization.add_safe_globals([_OverrideLinearPrecision, DelayedScaling, Format])

DEFAULT_MODEL_ID = "evo2_7b"
DEFAULT_LAYER_NAME = "blocks.28.mlp.l3"
PAD_ID = 0
BATCH_SIZE = 1
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

MODEL_CONFIG = {
    "evo2_1b_base": {"num_layers": 25, "params": "1B"},
    "evo2_7b":      {"num_layers": 32, "params": "7B"},
    "evo2_40b":     {"num_layers": 50, "params": "40B"},
}



def layer_index(layer: str) -> str:
    m = re.search(r"blocks\.(\d+)\.", layer)
    return m.group(1) if m else layer


def build_filename(seq_type: str, params: str, layer: str, mode: str, strand: str) -> str:
    return f"{seq_type}_Evo2_{params}_L{layer_index(layer)}_{mode}_{strand}.npy"


def pool(hidden: torch.Tensor, lengths: list[int], emb_type: str) -> torch.Tensor:
    """Apply pooling to a (B, L, D) hidden state tensor."""
    if emb_type == "average":
        mask = torch.zeros((hidden.shape[0], hidden.shape[1]), dtype=torch.bool, device=hidden.device)
        for b, l in enumerate(lengths):
            mask[b, :l] = True
        mask_expanded = mask.unsqueeze(-1).float()
        return (hidden * mask_expanded).sum(1) / mask_expanded.sum(1)
    else:  # last
        last_indices = torch.tensor([l - 1 for l in lengths], device=hidden.device)
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last_indices]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Evo2 embeddings from DNA/RNA sequences.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID,
                        help=f"Evo2 model variant (default: {DEFAULT_MODEL_ID})")
    parser.add_argument("--layer", nargs="+", default=[DEFAULT_LAYER_NAME],
                        help="One or more layer names to extract. All layers are extracted in a "
                             f"single forward pass. (default: {DEFAULT_LAYER_NAME})")
    parser.add_argument("--seq-type", default="DNA", choices=["DNA", "RNA"],
                        help="Sequence type label in output filename (default: DNA)")
    parser.add_argument("--strand", default="forward", choices=["forward", "reverse"],
                        help="Strand direction — selects input file and labels output (default: forward)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Sequences per forward pass (default: {BATCH_SIZE})")
    parser.add_argument("--emb-type", default="average", choices=["average", "last"],
                        help="Pooling: 'average' = mean over non-padding tokens; "
                             "'last' = last token from last layer (default: average)")
    return parser.parse_args()


def extract_embeddings(
    sequences: list[str],
    model,
    df: str,
    emb_type: str,
    layers: list[str],
    seq_type: str,
    strand: str,
    params: str,
    batch_size: int = BATCH_SIZE,
    pad_id: int = PAD_ID,
) -> None:
    """
    Extract embeddings from one or more layers in a single forward pass per batch.
    Saves one .npy file per layer.
    """
    os.makedirs("embeddings", exist_ok=True)
    all_pooled = {layer: [] for layer in layers}

    for start in range(0, len(sequences), batch_size):
        seqs = sequences[start:start + batch_size]
        token_ids = [model.tokenizer.tokenize(seq) for seq in seqs]
        lengths = [len(t) for t in token_ids]
        max_length = max(lengths)
        padded = [t + [pad_id] * (max_length - len(t)) for t in token_ids]
        input_ids = torch.tensor(padded, dtype=torch.int).to(DEVICE)

        with torch.no_grad():
            _, embeddings = model.forward(input_ids, return_embeddings=True, layer_names=layers)
            for layer in layers:
                hidden = embeddings[layer]  # B, L, D
                pooled = pool(hidden, lengths, emb_type)
                all_pooled[layer].append(pooled.float().cpu().numpy())

        print(f"Processed {min(start + batch_size, len(sequences))}/{len(sequences)} sequences")

    for layer in layers:
        combined = np.concatenate(all_pooled[layer], axis=0)
        fname = build_filename(seq_type, params, layer, emb_type, strand)
        np.save(f"embeddings/{df}_{fname}", combined)
        print(f"Saved {emb_type} embeddings layer {layer_index(layer)}: {combined.shape} -> {df}_{fname}")


if __name__ == "__main__":
    args = parse_args()

    cfg = MODEL_CONFIG.get(args.model)
    if cfg is None:
        raise ValueError(f"Unknown model {args.model!r}. Add it to MODEL_CONFIG or check the name.")

    layers = args.layer

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
            layers=layers,
            seq_type=args.seq_type,
            params=cfg["params"],
            strand=args.strand,
            batch_size=args.batch_size,
        )
