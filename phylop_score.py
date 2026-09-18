"""
Look up precomputed PhyloP conservation scores for NPC1 variant positions from
a UCSC bigWig track, queried remotely by byte-range HTTP requests (via
pyBigWig) so no whole-track (multi-GB) download is needed.

PhyloP is a per-base conservation score (phylogenetic p-value from PHAST):
positive = conserved (slower evolution than neutral expectation), negative =
accelerated/fast-evolving. It has no ref/alt dependence — it's a property of
the position, not the substitution — so unlike CADD/GPN there's one score per
site, joined onto every ref/alt pair at that site.

Default track: hg38.phyloP100way (100 vertebrate genomes, UCSC). Pass
--bigwig-url to use a different build/track, e.g. Zoonomia phyloP447way.

Examples
--------
python phylop_score.py                                  # default: output/df_preprocessed.csv
python phylop_score.py --df output/df_preprocessed_region.csv
python phylop_score.py --bigwig-url https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP447way/phyloP447way.bw
"""

import argparse

import pandas as pd
import pyBigWig

DEFAULT_BIGWIG_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw"


def strip_chr(chrom) -> str:
    return str(chrom).removeprefix("chr").removeprefix("Chr").removeprefix("CHR")


def query_phylop(bw, chrom: str, positions: list[int]) -> dict[int, float]:
    scores = {}
    for pos in positions:
        # bigWig is 0-based half-open; VCF-style `pos` here is 1-based.
        vals = bw.values(chrom, pos - 1, pos)
        scores[pos] = vals[0] if vals else None
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Join precomputed PhyloP conservation scores onto NPC1 variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--df", default="output/df_preprocessed.csv",
                        help="Input CSV with 'end', 'reference_base', 'alternate_base' columns. "
                             "Default: output/df_preprocessed.csv")
    parser.add_argument("--chrom", default="18",
                        help="Chromosome (the input has no chrom column). Default: 18")
    parser.add_argument("--bigwig-url", default=DEFAULT_BIGWIG_URL,
                        help="Remote bigWig URL to query. Default: UCSC hg38.phyloP100way")
    parser.add_argument("--out", default="output/phylop_scores.csv",
                        help="Output CSV path. Default: output/phylop_scores.csv")
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

    print(f"Opening remote bigWig: {args.bigwig_url}")
    bw = pyBigWig.open(args.bigwig_url)

    records = []
    for chrom, group in variants.groupby("chrom"):
        bw_chrom = f"chr{strip_chr(chrom)}"
        positions = group["pos"].unique().tolist()
        print(f"Querying chrom {chrom}: {len(positions)} unique positions ...")
        pos_scores = query_phylop(bw, bw_chrom, positions)
        merged = group.copy()
        merged["PhyloP_score"] = merged["pos"].map(pos_scores)
        records.append(merged)

    bw.close()

    out_df = pd.concat(records, ignore_index=True)
    out_df.to_csv(args.out, index=False)
    n_scored = out_df["PhyloP_score"].notna().sum()
    print(f"\nDone. Scored {n_scored}/{len(out_df)} variants -> {args.out}")


if __name__ == "__main__":
    main()
