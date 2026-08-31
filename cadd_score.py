"""
Fetch CADD deleteriousness scores (RawScore, PHRED) for NPC1 variants via the
CADD REST API (https://cadd.bihealth.org).

Reuses the same variant-loading logic as prepare_dataset.py, so any input
that works there works here.

Examples
--------
# Original Excel fitness-score dataset
python cadd_score.py --xlsx data/NPC1_mut_fitness_scores.xlsx

# ClinVar 2-star TSV
python cadd_score.py --tsv data/npc1_2star_GRCh38.tsv

# Single variant from CLI
python cadd_score.py --coords chr18:25290796:A:T

Notes
-----
- Chromosome is stripped of any "chr" prefix before calling the API (CADD's
  API expects e.g. "18", not "chr18").
- CADD scores are position-indexed per-genome-build; use --build to match
  the build your positions were called against (this project uses GRCh38).
- The API only serves SNVs. Indels are skipped with a warning (offline
  scoring via the CADD Snakemake pipeline is needed for those).
- Requests are rate-limited (--delay, default 1s) to be polite to the
  public API. For large datasets, downloading the pre-scored whole-genome
  file and looking up variants locally is much faster.
"""

import argparse
import sys
import time

import pandas as pd
import requests

from prepare_dataset import parse_xlsx, parse_tsv, parse_coords

API_TEMPLATE = "https://cadd.gs.washington.edu/api/v1.0/{version}/{chrom}:{pos}_{ref}_{alt}"
BUILD_VERSION = {"GRCh37": "v1.7", "GRCh38": "GRCh38-v1.7"}


def strip_chr(chrom) -> str:
    return str(chrom).removeprefix("chr").removeprefix("Chr").removeprefix("CHR")


def query_cadd(chrom: str, pos: int, ref: str, alt: str, version: str,
                timeout: int = 30, retries: int = 3) -> dict | None:
    url = API_TEMPLATE.format(version=version, chrom=strip_chr(chrom), pos=pos, ref=ref, alt=alt)
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            hits = resp.json()
            if not hits:
                return None
            return hits[0]
        except requests.exceptions.RequestException as e:
            if attempt == retries:
                print(f"  WARNING: CADD lookup failed for {chrom}:{pos}:{ref}:{alt} — {e}")
                return None
            time.sleep(2 * attempt)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch CADD scores for NPC1 variants via the CADD API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--xlsx", metavar="FILE",
                     help="Excel file with 'end', 'reference_base', 'alternate_base' columns. "
                          "Default: data/NPC1_mut_fitness_scores.xlsx")
    src.add_argument("--tsv", metavar="FILE",
                     help="ClinVar TSV file with PositionVCF/ReferenceAlleleVCF/AlternateAlleleVCF columns.")
    src.add_argument("--coords", nargs="+", metavar="CHR:POS:REF:ALT",
                     help="One or more variants as CHR:POS:REF:ALT.")

    parser.add_argument("--chrom", default="18",
                        help="Chromosome for --xlsx/--tsv modes (they have no chrom column). Default: 18")
    parser.add_argument("--build", choices=["GRCh37", "GRCh38"], default="GRCh38",
                        help="Genome build the positions are called against. Default: GRCh38")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds to sleep between API requests. Default: 1.0")
    parser.add_argument("--out", default="output/cadd_scores.csv",
                        help="Output CSV path. Default: output/cadd_scores.csv")

    args = parser.parse_args()

    if args.xlsx is None and args.tsv is None and args.coords is None:
        args.xlsx = "data/NPC1_mut_fitness_scores.xlsx"

    if args.xlsx:
        print(f"Loading variants from Excel: {args.xlsx}")
        variants = parse_xlsx(args.xlsx)
        variants["chrom"] = args.chrom
    elif args.tsv:
        print(f"Loading variants from ClinVar TSV: {args.tsv}")
        variants = parse_tsv(args.tsv)
        variants["chrom"] = args.chrom
    else:
        print(f"Variants from CLI: {args.coords}")
        variants = parse_coords(args.coords)

    print(f"Total variants: {len(variants)}")
    version = BUILD_VERSION[args.build]

    records = []
    for i, row in variants.iterrows():
        ref, alt = row["ref"], row["alt"]
        if len(ref) != 1 or len(alt) != 1:
            print(f"  SKIP variant {i} ({row['chrom']}:{row['pos']}:{ref}:{alt}) — "
                  f"CADD API only serves SNVs")
            records.append({**row.to_dict(), "CADD_RawScore": None, "CADD_PHRED": None})
            continue

        hit = query_cadd(row["chrom"], row["pos"], ref, alt, version)
        raw = hit.get("RawScore") if hit else None
        phred = hit.get("PHRED") if hit else None
        records.append({**row.to_dict(), "CADD_RawScore": raw, "CADD_PHRED": phred})
        print(f"  [{i+1}/{len(variants)}] {row['chrom']}:{row['pos']}:{ref}:{alt} -> "
              f"RawScore={raw} PHRED={phred}")
        time.sleep(args.delay)

    out_df = pd.DataFrame(records)
    out_df.to_csv(args.out, index=False)
    n_scored = out_df["CADD_PHRED"].notna().sum()
    print(f"\nDone. Scored {n_scored}/{len(out_df)} variants -> {args.out}")


if __name__ == "__main__":
    main()
