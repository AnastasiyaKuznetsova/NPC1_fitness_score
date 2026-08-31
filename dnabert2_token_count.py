"""
Tokenize DNA sequences with the DNABERT-2 tokenizer and report how many
tokens each sequence produces (BPE tokenization means token count is not
a fixed fraction of sequence length).

Example
-------
python dnabert2_token_count.py --input output/ref_seq_DNA_forward_20260714.npy
"""

import argparse
import statistics

import numpy as np
from transformers import AutoTokenizer

MODEL_NAME = "zhihan1996/DNABERT-2-117M"


def main():
    parser = argparse.ArgumentParser(
        description="Count DNABERT-2 tokens per sequence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", metavar="FILE", required=True,
                        help="Path to a .npy file of sequence strings (e.g. from prepare_dataset.py).")
    parser.add_argument("--model", default=MODEL_NAME,
                        help=f"HF model name for the tokenizer. Default: {MODEL_NAME}")

    args = parser.parse_args()

    print(f"Loading sequences from {args.input} ...")
    sequences = np.load(args.input, allow_pickle=True)

    print(f"Loading tokenizer: {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    token_counts = []
    for seq in sequences:
        n_tokens = len(tokenizer(seq)["input_ids"])
        token_counts.append(n_tokens)

    bp_per_token = [len(seq) / n for seq, n in zip(sequences, token_counts) if n > 0]

    print(f"\nSequences: {len(sequences)}")
    print(f"Tokens per sequence — mean: {statistics.mean(token_counts):.1f}, "
          f"median: {statistics.median(token_counts):.1f}, "
          f"min: {min(token_counts)}, max: {max(token_counts)}, "
          f"stdev: {statistics.pstdev(token_counts):.1f}")
    print(f"bp per token — mean: {statistics.mean(bp_per_token):.2f}")


if __name__ == "__main__":
    main()
