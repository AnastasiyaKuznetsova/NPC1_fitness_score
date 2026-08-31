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

Saves five files to a timestamped subdirectory of --out_dir
(--out_dir/YYYYMMDD_HHMMSS/), created fresh on every run:

  Four sequence-window .npy arrays, shape (N,) of strings (or (N*K,) with
  --jitter — see below), row i is the same variant as row i of
  variant_meta.csv below:

    ref_seq_DNA_forward.npy   ref_seq_DNA_reverse.npy
    mut_seq_DNA_forward.npy   mut_seq_DNA_reverse.npy

  One variant_meta.csv, row-aligned with the four arrays above, with
  columns pos/ref/alt (the variant as parsed) and edit_start (0-based
  index within the forward window where the allele begins — same for
  ref_seq and mut_seq, since both windows share the same left context).
  extract_embeddings.py's --pool-region downstream reads this to find
  where each variant's allele sits in its window (ref_len/alt_len are
  just len(ref)/len(alt) from this file).

  --jitter augmentation: also adds 'variant_id' (0-based, matches
  df_preprocessed.csv's row order) and 'offset' columns, letting
  regressor.py's --variant-meta-file group jittered copies of one variant
  back together for evaluation (train on every offset as an independent
  sample; average predictions per variant_id before scoring test folds).

--window is required and is baked into every output filename as '<window>bp'
(e.g. ref_seq_DNA_forward_8192bp.npy). If --jitter is used, the offsets are
also baked in as '_jitter<off1>_<off2>...' (omitted by default). Each run
gets its own --out_dir/YYYYMMDD_HHMMSS/ subdirectory.

Examples
--------
# Original Excel workflow (unchanged):
python prepare_dataset.py --xlsx data/NPC1_mut_fitness_scores.xlsx --window 8192

# ClinVar 2-star TSV:
python prepare_dataset.py --tsv data/npc1_2star_GRCh38.tsv --window 8192

# Single variant from CLI:
python prepare_dataset.py --coords chr18:25290796:A:T --window 8192

# Multiple variants from CLI:
python prepare_dataset.py --coords chr18:25290796:A:T chr18:25291000:G:C --window 8192

"""

import argparse
import os
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from Bio import SeqIO
from Bio.Seq import Seq

OUTPUT_DIR = "output"

# Default accession → used when fetching from NCBI
ACCESSION_DEFAULT = "NC_000018.10"
GENOME_URL_TEMPLATE = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=nuccore&id={accession}&rettype=fasta&retmode=text"
)


# ── Genome loading ─────────────────────────────────────────────────────────────

def fetch_genome(accession: str, max_retries: int = 5) -> Seq:
    url = GENOME_URL_TEMPLATE.format(accession=accession)
    print(f"Fetching genome {accession} from NCBI ...")
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            record = SeqIO.read(StringIO(response.text), "fasta")
            print(f"  Genome length: {len(record.seq):,} bp")
            return record.seq
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) as e:
            last_error = e
            wait = 2 ** attempt
            print(f"  Fetch attempt {attempt}/{max_retries} failed ({e}); retrying in {wait}s ...")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch genome {accession} after {max_retries} attempts") from last_error


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

def extract_window(genome: Seq, pos: int, ref: str, alt: str, window: int,
                    offset: int = 0) -> tuple[str, str, int]:
    """Return (ref_window, mut_window, edit_start) centered on pos (1-based), shifted
    by `offset` bp (jitter augmentation — see --jitter).

    Works for SNVs, insertions, and deletions.
    The window is centered on the first base of the variant (VCF convention),
    then re-centered `offset` bp away — offset=0 reproduces the plain centered
    window; offset>0 shifts the window downstream (the variant moves toward the
    window's start); offset<0 shifts it upstream (variant moves toward the end).
    ref_window always spans `window` bases (or fewer at chromosome ends).
    mut_window replaces the ref allele with the alt allele, so its length
    may differ from ref_window for indels.
    edit_start is the 0-based index of the first allele base within the
    window — identical for ref_window and mut_window since both are built
    from the same left context.
    Raises ValueError (like the reference-mismatch check) if `offset` is large
    enough to push the variant entirely outside the window.
    """
    ind = pos - 1  # 0-based index of first ref base
    actual = str(genome[ind: ind + len(ref)]).upper()
    if actual != ref:
        raise ValueError(
            f"Reference mismatch at pos={pos}: expected {ref!r}, genome has {actual!r}"
        )
    # Center the window on the midpoint of the ref allele, then apply the jitter offset
    center = ind + len(ref) // 2 + offset
    start  = max(0, center - window // 2)
    end    = min(center + window // 2, len(genome))
    ref_window = str(genome[start:end])
    target_len = len(ref_window)
    edit_start = ind - start
    if not (0 <= edit_start < target_len):
        raise ValueError(
            f"offset={offset} at pos={pos} pushes the variant outside the window "
            f"(edit_start={edit_start}, window length={target_len})"
        )

    # Build mut_window and adjust trailing context so it matches ref_window length.
    # Insertion (len(alt) > len(ref)): naive mut_window is too long — trim the right end.
    # Deletion  (len(alt) < len(ref)): naive mut_window is too short — extend the right end.
    size_delta = len(alt) - len(ref)
    mut_end    = min(end - size_delta, len(genome))
    mut_window = str(genome[start:ind]) + alt + str(genome[ind + len(ref):mut_end])
    mut_window = mut_window[:target_len]  # trim overshoot
    # Deletions near a chromosome end leave mut_window short — skip rather than pad with Ns.
    if len(mut_window) < target_len:
        raise ValueError(
            f"pos={pos}: deletion near chromosome end leaves mut_window short "
            f"({len(mut_window)} < {target_len}); skipping rather than padding with Ns"
        )
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
                        help=f"Directory for output files. Default: {OUTPUT_DIR}")
    parser.add_argument("--window", type=int, required=True,
                        help="Sequence window size in bp (centered on the variant). "
                             "Required; saved into the output filenames as '<window>bp'.")
    parser.add_argument("--prefix", default="",
                        help="Optional prefix for output filenames (e.g. 'chr18_' → "
                             "'chr18_ref_seq_DNA_forward.npy').")
    parser.add_argument("--jitter", type=int, nargs="+", default=None, metavar="BP",
                        help="Jitter augmentation: one or more bp offsets (e.g. -1000 -500 0 "
                             "500 1000) at which to re-center the window around each variant, "
                             "in addition to the plain centered window. Each (variant, offset) "
                             "pair becomes its own row, so N variants x K offsets produce up to "
                             "N*K rows (offsets that push the variant off the window edge are "
                             "skipped and logged, like a reference mismatch). variant_meta.csv "
                             "gets 'variant_id' (0-based, matches df_preprocessed.csv's row "
                             "order) and 'offset' columns so downstream training code can group "
                             "jittered copies back to their source variant. Omit for the original "
                             "single-row-per-variant behavior (offset=0 only).")

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
    offsets = args.jitter if args.jitter else [0]
    ref_seqs, mut_seqs, meta_rows = [], [], []
    skipped = 0
    for i, row in variants.iterrows():
        for offset in offsets:
            try:
                ref_w, mut_w, edit_start = extract_window(
                    genome, row["pos"], row["ref"], row["alt"], args.window, offset=offset,
                )
                ref_seqs.append(ref_w)
                mut_seqs.append(mut_w)
                meta_rows.append({
                    "variant_id": i, "offset": offset,
                    "pos": row["pos"], "ref": row["ref"], "alt": row["alt"], "edit_start": edit_start,
                })
            except ValueError as e:
                print(f"  SKIP variant {i} (pos={row['pos']}, offset={offset}): {e}")
                skipped += 1

    if not ref_seqs:
        sys.exit("ERROR: no variants successfully processed")
    if skipped:
        print(f"  Skipped {skipped} (variant, offset) pairs due to reference mismatch or out-of-window offset")

    # ── Save ───────────────────────────────────────────────────────────────────
    from datetime import datetime
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / timestamp_str
    out_dir.mkdir(parents=True, exist_ok=True)
    p = args.prefix
    window_str = f"{args.window}bp"
    jitter_str = f"_jitter{'_'.join(str(o) for o in args.jitter)}" if args.jitter else ""
    tag = f"{window_str}{jitter_str}"

    ref_forward = np.array(ref_seqs)
    mut_forward = np.array(mut_seqs)
    ref_reverse = reverse_complement(ref_forward)
    mut_reverse = reverse_complement(mut_forward)
    for fname, arr in [
        (f"{p}ref_seq_DNA_forward_{tag}.npy", ref_forward),
        (f"{p}ref_seq_DNA_reverse_{tag}.npy", ref_reverse),
        (f"{p}mut_seq_DNA_forward_{tag}.npy", mut_forward),
        (f"{p}mut_seq_DNA_reverse_{tag}.npy", mut_reverse),
    ]:
        out_path = out_dir / fname
        np.save(out_path, arr)
        print(f"  Saved {len(arr)} sequences → {out_path}")

    # Edit-position metadata, row-aligned with the four sequence arrays above.
    # Lets extract_embeddings.py's --pool-region downstream find where the
    # variant sits in each window without recomputing it.
    meta_path = out_dir / f"{p}variant_meta_{tag}.csv"
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False)
    print(f"  Saved {len(meta_rows)} rows → {meta_path}")

    print(f"\nDone. {len(ref_forward)} sequence pairs written to {out_dir}/")


if __name__ == "__main__":
    main()
