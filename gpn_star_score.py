"""
Look up precomputed GPN-Star variant-effect scores for NPC1 variants from the
public songlab/gpn-star-scores Hugging Face dataset
(https://huggingface.co/datasets/songlab/gpn-star-scores).

Unlike cadd_score.py (which queries a REST API one variant at a time), GPN-Star
scores are pre-computed for every possible human SNV and released as
chromosome-sharded Parquet files. This script range-queries those remote
Parquet files (via pyarrow + fsspec) so only the row groups covering the
variants' positions are downloaded — no whole-chromosome (~1-2 GB) download
needed.

Score columns (see the dataset's precomputed-scores tutorial):
  llr_calibrated      mutation-rate-calibrated alt-vs-ref log-likelihood ratio;
                       more negative = more constrained / larger predicted effect.
  abs_llr_calibrated   independently calibrated magnitude (NOT abs(llr_calibrated)).
  effect_score         -llr_calibrated, so higher = more deleterious (matches the
                        sign convention of CADD_PHRED for easy comparison).

Examples
--------
python gpn_star_score.py                          # default: output/df_preprocessed.csv
python gpn_star_score.py --df output/df_preprocessed_region.csv
"""

import argparse

import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import fsspec

SCORE_ROOT_TEMPLATE = (
    "https://huggingface.co/datasets/songlab/gpn-star-scores/resolve/main/"
    "data/{score_set}/llr/llr_chr{chrom}.parquet"
)


def strip_chr(chrom) -> str:
    return str(chrom).removeprefix("chr").removeprefix("Chr").removeprefix("CHR")


def query_gpn_star(chrom: str, positions: list[int], score_set: str, fs) -> pd.DataFrame:
    url = SCORE_ROOT_TEMPLATE.format(score_set=score_set, chrom=strip_chr(chrom))
    dataset = ds.dataset(url, filesystem=fs, format="parquet")
    filt = (
        (pc.field("pos") >= min(positions))
        & (pc.field("pos") <= max(positions))
        & pc.field("pos").isin(positions)
    )
    return dataset.to_table(filter=filt).to_pandas()


def main():
    parser = argparse.ArgumentParser(
        description="Join precomputed GPN-Star scores onto NPC1 variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--df", default="output/df_preprocessed.csv",
                        help="Input CSV with 'end', 'reference_base', 'alternate_base' columns. "
                             "Default: output/df_preprocessed.csv")
    parser.add_argument("--chrom", default="18",
                        help="Chromosome (the input has no chrom column). Default: 18")
    parser.add_argument("--score-set", default="gpn-star-hg38-m447-200m",
                        help="GPN-Star score set name under data/<score_set>/llr/. "
                             "Default: gpn-star-hg38-m447-200m")
    parser.add_argument("--out", default="output/gpn_star_scores.csv",
                        help="Output CSV path. Default: output/gpn_star_scores.csv")
    args = parser.parse_args()

    print(f"Loading variants from {args.df}")
    df = pd.read_csv(args.df)
    variants = pd.DataFrame({
        "chrom": args.chrom,
        "pos": df["end"],
        "ref": df["reference_base"],
        "alt": df["alternate_base"],
    })
    print(f"Total variants: {len(variants)}")

    fs = fsspec.filesystem("http")
    records = []
    for chrom, group in variants.groupby("chrom"):
        positions = group["pos"].unique().tolist()
        print(f"Querying chrom {chrom}: {len(positions)} unique positions ...")
        scores = query_gpn_star(chrom, positions, args.score_set, fs)
        scores["chrom"] = chrom
        merged = group.merge(scores, on=["chrom", "pos", "ref", "alt"], how="left")
        records.append(merged)

    out_df = pd.concat(records, ignore_index=True)
    out_df["effect_score"] = -out_df["llr_calibrated"]
    out_df = out_df.rename(columns={
        "llr_calibrated": "GPN_llr_calibrated",
        "abs_llr_calibrated": "GPN_abs_llr_calibrated",
        "effect_score": "GPN_effect_score",
    })

    out_df.to_csv(args.out, index=False)
    n_scored = out_df["GPN_llr_calibrated"].notna().sum()
    print(f"\nDone. Scored {n_scored}/{len(out_df)} variants -> {args.out}")


if __name__ == "__main__":
    main()
