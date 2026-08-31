"""
Tokenize 2.5 kb DNA sequences with the DNABERT-2 tokenizer and report how
many tokens each sequence produces (BPE tokenization means token count is
not a fixed fraction of sequence length).

Examples
--------
# Quick synthetic test, no data needed
python dnabert2_token_count.py


python dnabert2_token_count.py --input output/ref_seq_DNA_forward_20260714.npy
"""

import argparse
import random
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
    parser.add_argument("--input", metavar="FILE",
                        help="Optional .npy file of sequence strings (e.g. from prepare_dataset.py). "
                             "Each sequence is center-cropped to --length bp before tokenizing. "
                             "If omitted, --n random ACGT sequences are generated instead.")
    parser.add_argument("--length", type=int, default=2500,
                        help="Sequence length in bp. Default: 2500")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of random sequences to generate when --input is not given. Default: 100")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for synthetic sequences. Default: 0")
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

    print(f"\nSequences: {len(sequences)}, each {args.length} bp")
    print(f"Tokens per sequence — mean: {statistics.mean(token_counts):.1f}, "
          f"median: {statistics.median(token_counts):.1f}, "
          f"min: {min(token_counts)}, max: {max(token_counts)}, "
          f"stdev: {statistics.pstdev(token_counts):.1f}")
    print(f"bp per token — mean: {statistics.mean(bp_per_token):.2f}")


if __name__ == "__main__":
    main()
