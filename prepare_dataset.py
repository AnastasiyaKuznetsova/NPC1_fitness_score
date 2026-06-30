"""
Prepare DNA sequence windows for NPC1 fitness score modeling.

Saves four .npy arrays to output/:
    ref_seq_DNA_forward.npy  — reference sequences, forward strand
    ref_seq_DNA_reverse.npy  — reference sequences, reverse complement
    mut_seq_DNA_forward.npy  — mutant sequences, forward strand
    mut_seq_DNA_reverse.npy  — mutant sequences, reverse complement

Each array has shape (N,) of strings, one per variant.
"""

import os
from io import StringIO

import numpy as np
import pandas as pd
import requests
from Bio import SeqIO
from Bio.Seq import Seq

WINDOW_SIZE = 8192
OUTPUT_DIR = "output"
GENOME_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=nuccore&id=NC_000018.10&rettype=fasta&retmode=text"
)


def fetch_genome(url: str) -> Seq:
    print("Fetching genome NC_000018.10 ...")
    response = requests.get(url)
    response.raise_for_status()
    record = SeqIO.read(StringIO(response.text), "fasta")
    print(f"Genome length: {len(record.seq):,} bp")
    return record.seq


def extract_window(seq: Seq, pos: int, new_base: str, reference_base: str, window: int):
    """Return (ref_window, mut_window) as strings centered on pos (1-based)."""
    ind = pos - 1
    assert seq[ind] == reference_base, (
        f"Reference mismatch at ind={ind}: expected {reference_base!r}, got {seq[ind]!r}"
    )
    start = max(0, ind - window // 2)
    end = min(ind + window // 2 + 1, len(seq))
    ref_window = str(seq[start:end])
    mut_window = str(seq[start:ind]) + new_base + str(seq[ind + 1:end])
    return ref_window, mut_window


def reverse_complement(sequences: np.ndarray) -> np.ndarray:
    return np.array([str(Seq(s).reverse_complement()) for s in sequences])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_excel("data/NPC1_mut_fitness_scores.xlsx", header=1)
    df = df[df["Consequence"] != "splice region"].reset_index(drop=True)
    print(f"Variants after filtering splice regions: {len(df)}")

    genome = fetch_genome(GENOME_URL)

    ref_seqs, mut_seqs = [], []
    for _, row in df.iterrows():
        ref_window, mut_window = extract_window(
            genome,
            pos=row["end"],
            new_base=row["alternate_base"],
            reference_base=row["reference_base"],
            window=WINDOW_SIZE,
        )
        ref_seqs.append(ref_window)
        mut_seqs.append(mut_window)

    ref_forward = np.array(ref_seqs)
    mut_forward = np.array(mut_seqs)
    ref_reverse = reverse_complement(ref_forward)
    mut_reverse = reverse_complement(mut_forward)

    assert len(ref_forward) == len(mut_forward) == len(df), "Shape mismatch"

    np.save(f"{OUTPUT_DIR}/ref_seq_DNA_forward.npy", ref_forward)
    np.save(f"{OUTPUT_DIR}/ref_seq_DNA_reverse.npy", ref_reverse)
    np.save(f"{OUTPUT_DIR}/mut_seq_DNA_forward.npy", mut_forward)
    np.save(f"{OUTPUT_DIR}/mut_seq_DNA_reverse.npy", mut_reverse)

    print(f"\nSaved {len(ref_forward)} sequences to {OUTPUT_DIR}/")
    print("  ref_seq_DNA_forward.npy")
    print("  ref_seq_DNA_reverse.npy")
    print("  mut_seq_DNA_forward.npy")
    print("  mut_seq_DNA_reverse.npy")


if __name__ == "__main__":
    main()
