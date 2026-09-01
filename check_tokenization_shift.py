"""
Diagnose BPE re-tokenization between ref/mut sequence pairs for DNABERT-2.

DNABERT-2 uses a BPE tokenizer, not fixed-width k-mers, so a single-base
substitution can shift token boundaries for everything downstream of the
variant. When that happens, PLL(ref) and PLL(mut) are sums over different
numbers of tokens covering different substrings, and the delta computed in
dnabert2_zero_shot.py measures re-segmentation noise instead of variant
effect -- which shows up as a near-zero Spearman against CADD.

This script tokenizes each ref/mut pair and reports:
  - how many pairs differ in token count
  - how many tokens differ in total (even when lengths match)
  - how far downstream the divergence propagates past the first change

Interpretation: if most pairs differ by more than ~1-2 tokens, whole-window
PLL delta is not a usable variant score -- switch to scoring only tokens
local to the variant.

--ref-file/--mut-file must be row-aligned, same files passed to
dnabert2_zero_shot.py.

Example
-------
python check_tokenization_shift.py \
    --ref-file output/20260831_133902/ref_seq_DNA_forward_2500bp.npy \
    --mut-file output/20260831_133902/mut_seq_DNA_forward_2500bp.npy
"""

import argparse
import sys

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

MODEL_NAME = "zhihan1996/DNABERT-2-117M"


def compare_pair(ref_ids: list, mut_ids: list) -> dict:
    """Summarize how two token-ID sequences diverge."""
    len_ref, len_mut = len(ref_ids), len(mut_ids)

    # First position where the two tokenizations part ways.
    first_diff = None
    for i in range(min(len_ref, len_mut)):
        if ref_ids[i] != mut_ids[i]:
            first_diff = i
            break
    if first_diff is None and len_ref != len_mut:
        first_diff = min(len_ref, len_mut)

    # Length of the common suffix, so we can tell a local edit (divergence
    # closes again quickly) from a cascade (differs to the end).
    suffix = 0
    while (suffix < min(len_ref, len_mut)
           and ref_ids[len_ref - 1 - suffix] == mut_ids[len_mut - 1 - suffix]):
        suffix += 1

    if first_diff is None:
        n_diff_ref = n_diff_mut = 0
    else:
        n_diff_ref = max(0, len_ref - suffix - first_diff)
        n_diff_mut = max(0, len_mut - suffix - first_diff)

    return {
        "n_tokens_ref": len_ref,
        "n_tokens_mut": len_mut,
        "len_delta": len_mut - len_ref,
        "first_diff_idx": first_diff,
        "n_diff_tokens_ref": n_diff_ref,
        "n_diff_tokens_mut": n_diff_mut,
        "frac_tokens_changed": n_diff_ref / len_ref if len_ref else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check whether ref/mut pairs tokenize to the same length under DNABERT-2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ref-file", default="output/20260831_133902/ref_seq_DNA_forward_2500bp.npy",
                        help="Reference sequence windows (.npy).")
    parser.add_argument("--mut-file", default="output/20260831_133902/mut_seq_DNA_forward_2500bp.npy",
                        help="Mutant sequence windows (.npy).")
    parser.add_argument("--model", default=MODEL_NAME,
                        help=f"HF model name. Default: {MODEL_NAME}")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only check the first N pairs. Default: all")
    parser.add_argument("--out", default=None,
                        help="Optional CSV path for the per-pair table.")

    args = parser.parse_args()

    ref_seqs = np.load(args.ref_file, allow_pickle=True)
    mut_seqs = np.load(args.mut_file, allow_pickle=True)
    if len(ref_seqs) != len(mut_seqs):
        sys.exit(f"ERROR: {len(ref_seqs)} reference vs {len(mut_seqs)} mutant sequences — must match")

    if args.limit is not None:
        ref_seqs = ref_seqs[:args.limit]
        mut_seqs = mut_seqs[:args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    rows = []
    for i, (ref, mut) in enumerate(zip(ref_seqs, mut_seqs)):
        ref_ids = tokenizer(str(ref))["input_ids"]
        mut_ids = tokenizer(str(mut))["input_ids"]
        row = compare_pair(ref_ids, mut_ids)
        row["pair_idx"] = i
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(ref_seqs)}")

    df = pd.DataFrame(rows)
    n = len(df)

    n_len_differ = int((df["len_delta"] != 0).sum())
    n_identical = int((df["n_diff_tokens_ref"] == 0).sum())
    n_single_token = int(((df["n_diff_tokens_ref"] <= 1) & (df["n_diff_tokens_mut"] <= 1)).sum())

    print(f"\n{'=' * 62}")
    print(f"Tokenization shift report — {n} ref/mut pairs, model {args.model}")
    print(f"{'=' * 62}")
    print(f"Token counts differ (len_ref != len_mut) : {n_len_differ:5d}  ({n_len_differ / n:6.1%})")
    print(f"Tokenizations identical (no diff at all) : {n_identical:5d}  ({n_identical / n:6.1%})")
    print(f"Clean local edit (<=1 token changed)     : {n_single_token:5d}  ({n_single_token / n:6.1%})")
    print()
    print("Tokens changed per pair (ref side):")
    print(df["n_diff_tokens_ref"].describe(percentiles=[0.5, 0.9, 0.99]).to_string())
    print()
    print("Fraction of the ref window's tokens changed:")
    print(df["frac_tokens_changed"].describe(percentiles=[0.5, 0.9, 0.99]).to_string())
    print()
    print("Length delta (mut - ref) value counts:")
    print(df["len_delta"].value_counts().sort_index().to_string())

    frac_clean = n_single_token / n
    print(f"\n{'=' * 62}")
    if frac_clean > 0.95:
        print("VERDICT: tokenization is stable. Re-segmentation is NOT the cause")
        print("         of the near-zero Spearman — look elsewhere (model capacity,")
        print("         whole-window summation, or CADD/PLL not measuring the same thing).")
    elif frac_clean > 0.5:
        print("VERDICT: mixed. A substantial minority of pairs re-segment, which adds")
        print("         noise to the PLL delta. Worth switching to variant-local scoring.")
    else:
        print("VERDICT: tokenization routinely shifts. Whole-window PLL delta is")
        print("         dominated by re-segmentation noise, not variant effect — this")
        print("         is very likely the main cause of the near-zero Spearman.")
        print("         Fix: score only tokens overlapping the variant position, or")
        print("         length-normalize, rather than summing the full window.")
    print(f"{'=' * 62}")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nSaved per-pair table -> {args.out}")


if __name__ == "__main__":
    main()
