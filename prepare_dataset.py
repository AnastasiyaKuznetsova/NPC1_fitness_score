"""
Prepare DNA sequence windows for NPC1 fitness score modeling.

Three input modes:
  1. --xlsx    (default) : original Excel file with 'end', 'reference_base', 'alternate_base' columns
  2. --tsv               : ClinVar TSV (e.g. npc1_2star_GRCh38.tsv) — uses PositionVCF,
                           ReferenceAlleleVCF, AlternateAlleleVCF columns directly.
                           Rows with missing alleles are skipped automatically.
  3. --coords            : one or more genomic coordinates given directly on the CLI
                           format: CHR:POS:REF:ALT  (e.g. chr18:25290796:A:T)

For modes 2 & 3 the chromosome accession is resolved automatically via --accession
(default: NC_000018.10 for chr18). Pass --accession for a different chromosome.

Saves four .npy arrays to --out_dir:
    ref_seq_DNA_forward.npy
    ref_seq_DNA_reverse.npy
    mut_seq_DNA_forward.npy
    mut_seq_DNA_reverse.npy

Each array has shape (N,) of strings, one per variant.

Examples
--------
# Original Excel workflow (unchanged):
python prepare_dataset.py --xlsx data/NPC1_mut_fitness_scores.xlsx

# ClinVar 2-star TSV:
python prepare_dataset.py --tsv data/npc1_2star_GRCh38.tsv

# Single variant from CLI:
python prepare_dataset.py --coords chr18:25290796:A:T

# Multiple variants from CLI:
python prepare_dataset.py --coords chr18:25290796:A:T chr18:25291000:G:C

# Provide a local FASTA instead of fetching:
python prepare_dataset.py --tsv data/npc1_2star_GRCh38.tsv --fasta data/chr18.fa
"""

import argparse
import os
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from Bio import SeqIO
from Bio.Seq import Seq

WINDOW_SIZE = 8192
OUTPUT_DIR = "output"

# Default accession → used when fetching from NCBI
ACCESSION_DEFAULT = "NC_000018.10"
GENOME_URL_TEMPLATE = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=nuccore&id={accession}&rettype=fasta&retmode=text"
)


# ── Genome loading ─────────────────────────────────────────────────────────────

def fetch_genome(accession: str) -> Seq:
    url = GENOME_URL_TEMPLATE.format(accession=accession)
    print(f"Fetching genome {accession} from NCBI ...")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    record = SeqIO.read(StringIO(response.text), "fasta")
    print(f"  Genome length: {len(record.seq):,} bp")
    return record.seq


def load_fasta(path: str) -> Seq:
    print(f"Loading genome from {path} ...")
    records = list(SeqIO.parse(path, "fasta"))
    if len(records) == 0:
        sys.exit(f"ERROR: no sequences found in {path}")
    if len(records) > 1:
        print(f"  WARNING: {len(records)} sequences in FASTA — using first record ({records[0].id})")
    seq = records[0].seq
    print(f"  Sequence length: {len(seq):,} bp")
    return seq


# ── Variant parsing ────────────────────────────────────────────────────────────


def parse_coords(coords: list[str]) -> pd.DataFrame:
    """Parse CHR:POS:REF:ALT strings into a DataFrame."""
    rows = []
    for c in coords:
        parts = c.split(":")
        if len(parts) != 4:
            sys.exit(f"ERROR: --coords must be in CHR:POS:REF:ALT format, got: {c!r}")
        chrom, pos, ref, alt = parts[0], int(parts[1]), parts[2].upper(), parts[3].upper()
        rows.append({"chrom": chrom, "pos": pos, "ref": ref, "alt": alt})
    return pd.DataFrame(rows)


def parse_tsv(path: str) -> pd.DataFrame:
    """Parse a ClinVar TSV (e.g. npc1_2star_GRCh38.tsv) into pos/ref/alt DataFrame.

    Uses the pre-computed VCF columns (PositionVCF, ReferenceAlleleVCF, AlternateAlleleVCF).
    Rows with missing or non-numeric positions are skipped.
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df.columns = [c.lstrip("#") for c in df.columns]
    required = {"PositionVCF", "ReferenceAlleleVCF", "AlternateAlleleVCF"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: TSV missing columns: {missing}")

    n_before = len(df)
    df["PositionVCF"] = pd.to_numeric(df["PositionVCF"], errors="coerce")
    df = df.dropna(subset=["PositionVCF", "ReferenceAlleleVCF", "AlternateAlleleVCF"])
    df = df[
        (df["ReferenceAlleleVCF"].astype(str).str.upper() != "NA") &
        (df["AlternateAlleleVCF"].astype(str).str.upper() != "NA")
    ]
    skipped = n_before - len(df)
    if skipped:
        print(f"  Skipped {skipped} rows with missing/non-numeric position or alleles")

    result = pd.DataFrame({
        "pos": df["PositionVCF"].astype(int).values,
        "ref": df["ReferenceAlleleVCF"].astype(str).str.upper().values,
        "alt": df["AlternateAlleleVCF"].astype(str).str.upper().values,
    })
    print(f"  Loaded {len(result)} variants from TSV ({path})")
    return result


def parse_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, header=1)
    df = df[df["Consequence"] != "splice region"].reset_index(drop=True)
    print(f"  Variants after filtering splice regions: {len(df)}")
    # Normalise to common column names
    return pd.DataFrame({
        "pos": df["end"],
        "ref": df["reference_base"].str.upper(),
        "alt": df["alternate_base"].str.upper(),
    })


# ── Sequence extraction ────────────────────────────────────────────────────────

def extract_window(genome: Seq, pos: int, ref: str, alt: str, window: int) -> tuple[str, str, int]:
    """Return (ref_window, mut_window, edit_start) centered on pos (1-based).

    Works for SNVs, insertions, and deletions.
    The window is centered on the first base of the variant (VCF convention).
    ref_window always spans `window` bases (or fewer at chromosome ends).
    mut_window replaces the ref allele with the alt allele, so its length
    may differ from ref_window for indels.
    edit_start is the 0-based index of the first allele base within the
    window — identical for ref_window and mut_window since both are built
    from the same left context.
    """
    ind = pos - 1  # 0-based index of first ref base
    actual = str(genome[ind: ind + len(ref)]).upper()
    if actual != ref:
        raise ValueError(
            f"Reference mismatch at pos={pos}: expected {ref!r}, genome has {actual!r}"
        )
    # Center the window on the midpoint of the ref allele
    center = ind + len(ref) // 2
    start  = max(0, center - window // 2)
    end    = min(center + window // 2, len(genome))
    ref_window = str(genome[start:end])
    target_len = len(ref_window)
    edit_start = ind - start

    # Build mut_window and adjust trailing context so it matches ref_window length.
    # Insertion (len(alt) > len(ref)): naive mut_window is too long — trim the right end.
    # Deletion  (len(alt) < len(ref)): naive mut_window is too short — extend the right end.
    size_delta = len(alt) - len(ref)
    mut_end    = min(end - size_delta, len(genome))
    mut_window = str(genome[start:ind]) + alt + str(genome[ind + len(ref):mut_end])
    mut_window = mut_window[:target_len]  # trim overshoot
    # Deletions near a chromosome end leave mut_window short — pad with Ns rather than crash.
    if len(mut_window) < target_len:
        mut_window = mut_window + "N" * (target_len - len(mut_window))
    if len(mut_window) != target_len:
        raise ValueError(
            f"Window length mismatch at pos={pos}: ref={target_len}, mut={len(mut_window)}"
        )
    return ref_window, mut_window, edit_start


def reverse_complement(sequences: np.ndarray) -> np.ndarray:
    return np.array([str(Seq(s).reverse_complement()) for s in sequences])


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate reference and mutant DNA windows for Evo2 embedding.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input mode (mutually exclusive)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--xlsx", metavar="FILE",
                     help="Excel file with 'end', 'reference_base', 'alternate_base' columns "
                          "(row 2 header). Default: data/NPC1_mut_fitness_scores.xlsx")
    src.add_argument("--tsv", metavar="FILE",
                     help="ClinVar TSV file with PositionVCF/ReferenceAlleleVCF/AlternateAlleleVCF "
                          "columns (e.g. data/npc1_2star_GRCh38.tsv)")
    src.add_argument("--coords", nargs="+", metavar="CHR:POS:REF:ALT",
                     help="One or more variants as CHR:POS:REF:ALT (e.g. chr18:25290796:A:T).")

    # Genome source
    parser.add_argument("--fasta", metavar="FILE", default=None,
                        help="Local FASTA file for the reference genome. "
                             "If omitted, the genome is fetched from NCBI.")
    parser.add_argument("--accession", default=ACCESSION_DEFAULT,
                        help=f"NCBI accession to fetch when --fasta is not provided. "
                             f"Default: {ACCESSION_DEFAULT}")

    # Output
    parser.add_argument("--out_dir", default=OUTPUT_DIR,
                        help=f"Directory for output .npy files. Default: {OUTPUT_DIR}")
    parser.add_argument("--window", type=int, default=WINDOW_SIZE,
                        help=f"Sequence window size in bp (centered on the variant). "
                             f"Default: {WINDOW_SIZE}")
    parser.add_argument("--prefix", default="",
                        help="Optional prefix for output filenames (e.g. 'chr18_' → "
                             "'chr18_ref_seq_DNA_forward.npy').")

    args = parser.parse_args()

    # Default to xlsx if no source given
    if args.xlsx is None and args.tsv is None and args.coords is None:
        args.xlsx = "data/NPC1_mut_fitness_scores.xlsx"

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Parse variants ─────────────────────────────────────────────────────────
    if args.xlsx:
        print(f"Loading variants from Excel: {args.xlsx}")
        variants = parse_xlsx(args.xlsx)
    elif args.tsv:
        print(f"Loading variants from ClinVar TSV: {args.tsv}")
        variants = parse_tsv(args.tsv)
    else:
        print(f"Variants from CLI: {args.coords}")
        variants = parse_coords(args.coords)

    print(f"Total variants to process: {len(variants)}")

    # Warn if --coords variants span chromosomes other than the active accession.
    if "chrom" in variants.columns:
        unique_chroms = variants["chrom"].dropna().unique()
        if len(unique_chroms) > 1:
            sys.exit(
                f"ERROR: --coords variants span multiple chromosomes {list(unique_chroms)}. "
                "Run separately per chromosome with --accession."
            )

    # ── Load genome ────────────────────────────────────────────────────────────
    if args.fasta:
        genome = load_fasta(args.fasta)
    else:
        genome = fetch_genome(args.accession)

    # ── Build windows ──────────────────────────────────────────────────────────
    ref_seqs, mut_seqs = [], []
    edit_starts, ref_lens, alt_lens = [], [], []
    skipped = 0
    for i, row in variants.iterrows():
        try:
            ref_w, mut_w, edit_start = extract_window(genome, row["pos"], row["ref"], row["alt"], args.window)
            ref_seqs.append(ref_w)
            mut_seqs.append(mut_w)
            edit_starts.append(edit_start)
            ref_lens.append(len(row["ref"]))
            alt_lens.append(len(row["alt"]))
        except ValueError as e:
            print(f"  SKIP variant {i} (pos={row['pos']}): {e}")
            skipped += 1

    if not ref_seqs:
        sys.exit("ERROR: no variants successfully processed")
    if skipped:
        print(f"  Skipped {skipped} variants due to reference mismatch")

    # ── Save ───────────────────────────────────────────────────────────────────
    from datetime import datetime
    out_dir = Path(args.out_dir)
    p = args.prefix
    date_str = datetime.now().strftime("%Y%m%d")

    ref_forward = np.array(ref_seqs)
    mut_forward = np.array(mut_seqs)
    ref_reverse = reverse_complement(ref_forward)
    mut_reverse = reverse_complement(mut_forward)
    for fname, arr in [
        (f"{p}ref_seq_DNA_forward_{date_str}.npy", ref_forward),
        (f"{p}ref_seq_DNA_reverse_{date_str}.npy", ref_reverse),
        (f"{p}mut_seq_DNA_forward_{date_str}.npy", mut_forward),
        (f"{p}mut_seq_DNA_reverse_{date_str}.npy", mut_reverse),
        # Edit-position metadata, index-aligned with the four sequence arrays
        # above. Lets extract_embeddings.py's --pool-region downstream find
        # where the variant sits in each window without recomputing it.
        (f"{p}edit_start_{date_str}.npy", np.array(edit_starts, dtype=np.int64)),
        (f"{p}ref_len_{date_str}.npy", np.array(ref_lens, dtype=np.int64)),
        (f"{p}alt_len_{date_str}.npy", np.array(alt_lens, dtype=np.int64)),
    ]:
        out_path = out_dir / fname
        np.save(out_path, arr)
        print(f"  Saved {len(arr)} sequences → {out_path}")
    print(f"\nDone. {len(ref_forward)} sequence pairs written to {out_dir}/")


if __name__ == "__main__":
    main()
