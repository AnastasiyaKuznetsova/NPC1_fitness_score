import argparse
import os

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
AVERAGE = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Evo2 embeddings from DNA sequences.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_ID,
        help=f"Evo2 model variant to load (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--layer", default=DEFAULT_LAYER_NAME,
        help=f"Layer name to extract hidden states from (default: {DEFAULT_LAYER_NAME})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Sequences per forward pass (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--no-average", dest="average", action="store_false",
        help="Return per-token embeddings instead of sequence-averaged embeddings",
    )
    return parser.parse_args()


def extract_embeddings(
    sequences: list[str],
    model,
    df: str,
    layer: str = DEFAULT_LAYER_NAME,
    batch_size: int = BATCH_SIZE,
    pad_id: int = PAD_ID,
    average: bool = AVERAGE,
) -> np.ndarray:  # type: ignore
    """
    Returns hidden-state embeddings, shape (N, hidden_dim) if average=True else (N, seq_len, hidden_dim).

    Args:
        sequences:  list of DNA strings (A/T/C/G, upper-case recommended)
        layer:      hidden layer name to pool
        batch_size: sequences per forward pass
        average:    if True, average embeddings across sequence length
    """
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
            else:
                pooled = hidden

        print(f"Processed {min(start + batch_size, len(sequences))}/{len(sequences)} sequences")
        os.makedirs("embeddings", exist_ok=True)
        np.save(f"embeddings/{df}_emb_DNA_avg_{average}_{i}.npy", pooled.float().cpu().numpy())

if __name__ == "__main__":
    args = parse_args()

    print(f"Loading {args.model} ...")
    model = Evo2(args.model)
    print("Model loaded.\n")

    for df in ["ref_seq", "mut_seq"]:
        seqs = np.load(f"output/{df}_DNA.npy")
        extract_embeddings(seqs, model, df, args.layer, args.batch_size, args.average)
        

                

                


                  
            