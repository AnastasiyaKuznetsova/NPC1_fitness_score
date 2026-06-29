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
    parser.add_argument(
        "--last", action="store_true",
        help="Extract the last (non-padding) token hidden state instead of averaging",
    )
    parser.add_argument(
        "--cls", action="store_true",
        help="Extract the CLS (first) token hidden state instead of averaging",
    )
    return parser.parse_args()


def extract_embeddings(
    sequences: list[str],
    model,
    df: str,
    average: bool,
    last: bool,
    cls: bool,
    layer: str = DEFAULT_LAYER_NAME,
    batch_size: int = BATCH_SIZE,
    pad_id: int = PAD_ID,
) -> np.ndarray:  # type: ignore
    """
    Returns hidden-state embeddings per sequence.

    Pooling modes (mutually exclusive, checked by caller):
        average=True  -> mean over non-padding tokens, shape (N, D)
        last=True     -> last non-padding token,       shape (N, D)
        cls=True      -> first (CLS) token,            shape (N, D)
        all False     -> full token sequence,           shape per-batch (B, L, D), saved per batch
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

            if average:
                mask = torch.zeros((hidden.shape[0], hidden.shape[1]), dtype=torch.bool, device=DEVICE)
                for b, l in enumerate(lengths):
                    mask[b, :l] = True
                mask_expanded = mask.unsqueeze(-1).float()  # B, L, 1
                pooled = (hidden * mask_expanded).sum(1) / mask_expanded.sum(1)
            elif last:
                last_indices = torch.tensor([l - 1 for l in lengths], device=DEVICE)
                pooled = hidden[torch.arange(hidden.shape[0], device=DEVICE), last_indices]  # B, D
            elif cls:
                pooled = hidden[:, 0, :]  # B, D
            else:
                pooled = hidden

        print(f"Processed {min(start + batch_size, len(sequences))}/{len(sequences)} sequences")

        if not average and not last and not cls:
            np.save(f"embeddings/{df}_emb_DNA_seq_{i}.npy", pooled.float().cpu().numpy())
        else:
            all_pooled.append(pooled.float().cpu().numpy())

    if all_pooled:
        mode = "avg" if average else "last" if last else "cls"
        combined = np.concatenate(all_pooled, axis=0)
        np.save(f"embeddings/{df}_emb_DNA_{mode}.npy", combined)
        print(f"Saved {mode} embeddings: {combined.shape}")

if __name__ == "__main__":
    args = parse_args()

    print(f"Loading {args.model} ...")
    model = Evo2(args.model)
    print("Model loaded.\n")

    modes = [not args.average, args.last, args.cls]
    if sum(modes) > 1:
        raise ValueError("--no-average, --last, and --cls are mutually exclusive")

    for df in ["ref_seq", "mut_seq"]:
        seqs = np.load(f"output/{df}_DNA.npy")
        extract_embeddings(sequences=seqs, model=model, df=df, layer=args.layer, batch_size=args.batch_size, average=args.average, last=args.last, cls=args.cls)
        

                

                


                  
            