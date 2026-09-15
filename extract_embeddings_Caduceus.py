"""
Extract Caduceus (Ph or Ps) embeddings from the ref_seq/mut_seq windows written
by prepare_dataset.py, pooled into one .npy file.

Caduceus is a bidirectional Mamba (BiMamba) DNA language model, hosted
directly on the HF Hub with trust_remote_code=True (no local repo clone
needed, unlike JanusDNA) — but it needs the `mamba_ssm` package with its
compiled CUDA kernels, same as JanusDNA.

  --variant ph   Post-hoc (Ph): NOT reverse-complement equivariant in a single
                 forward pass — the model card says to run it on a sequence
                 and its RC separately and average the two. This pipeline
                 already gets RC via prepare_dataset.py's separate forward/
                 reverse strand files (see --strand below), so run this
                 script once per strand and combine downstream, the same way
                 the other extract_embeddings_*.py scripts here do — this
                 script does not do an internal RC pass.
  --variant ps   Parameter-sharing (Ps/RCPS): RC-equivariant within a single
                 forward pass by construction. Its last_hidden_state has
                 2 * d_model channels (a forward-read half and an RC-read
                 half, tied via shared weights) — this script pools and saves
                 that full vector as-is; it does not split or recombine the
                 two halves.

Caduceus's tokenizer is one-token-per-base (CaduceusTokenizer, no BPE), so
unlike DNABERT-2 no offset-mapping is needed to locate a base position in
token space. Its HF forward() takes only input_ids (no attention_mask
parameter at all — see modeling_caduceus.py) and its Mamba backbone does not
support masking padded positions out of the state-space scan, so (like
JanusDNA) batching different-length sequences together would leak padding
into shorter sequences' representations. --batch-size therefore defaults to
1; only raise it if every sequence in your input file has the same length
(e.g. fixed-width windows from prepare_dataset.py).

Caduceus is bidirectional (not causal), so there is no "hasn't seen the edit
yet" isolation the way there is for JanusDNA's --extract-from layer mode —
every position has already seen the whole sequence. --pool-region downstream
here is purely a locality window around the edit, analogous to DNABERT-2's
downstream pooling.

Input
  --ref-file       Reference sequences .npy. Default: output/ref_seq_DNA_{strand}.npy
  --mut-file       Mutant sequences .npy.    Default: output/mut_seq_DNA_{strand}.npy
  --strand         forward or reverse — selects the default input files and labels
                    output. Default: forward.
  --batch-size     Sequences per forward pass. Default: 1 (see module docstring
                    on why padding is unsafe here). The script warns and still
                    pads (unmasked) if lengths differ within a batch.

Pooling
  --pool-region    'full' (default): pool over the whole sequence.
                    'downstream': pool only a window around the edit — requires
                    --variant-meta-file.
  --emb-type       'average': mean over the pooled region. 'last': last token —
                    only valid with --pool-region full. Default: average.
  --downstream-k   Only used with --pool-region downstream. One or more window
                    sizes to test: an integer k means positions [start, start+k]
                    inclusive, where start is the allele's own last base, or
                    'all' for everything to the sequence end. One output file
                    is saved per k, e.g. --downstream-k 0 32 128 512 all.
                    Default: all.
  --variant-meta-file
                    variant_meta.csv written by prepare_dataset.py (columns
                    pos, ref, alt, edit_start), row-aligned with --ref-file/
                    --mut-file. Required for --pool-region downstream.
                    Default: output/variant_meta.csv.

Output
  Saved under embeddings/Caduceus_{variant}_{d_model}dim_{context_window}_emb/,
  e.g. Caduceus_ph_256dim_8192bp_emb/ (context_window is parsed from the input
  filename — e.g. ref_seq_DNA_forward_8192bp.npy — so the --ref-file/
  --variant-meta-file name must contain a "<N>bp" token).
  One {ref_seq,mut_seq}_Lfinal_{emb_type}_{strand}[_ds{k}].npy per input
  (and per k, if swept), shape (N, D) — D = d_model for --variant ph,
  2 * d_model for --variant ps.

Example
-------
apptainer exec --nv --bind "$PWD":"$PWD" --pwd "$PWD" sif/caduceus.sif python3 \\
    extract_embeddings_Caduceus.py --variant ph \\
    --ref-file output/20260903_120806/ref_seq_DNA_forward_8192bp.npy \\
    --mut-file output/20260903_120806/mut_seq_DNA_forward_8192bp.npy
"""

import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from embedding_utils import parse_context_window, parse_downstream_k, pool, pool_downstream

MODEL_NAMES = {
    "ph": "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
    "ps": "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
}
MODEL_FAMILY = "Caduceus"
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def build_filename(mode: str, strand: str, region_suffix: str = "") -> str:
    suffix = f"_{region_suffix}" if region_suffix else ""
    return f"Lfinal_{mode}_{strand}{suffix}.npy"


def parse_pretrain_seqlen(model_name: str):
    """Best-effort parse of the 'seqlen-<N>k' token in Caduceus's own HF repo
    naming convention (e.g. 'caduceus-ph_seqlen-131k_d_model-256_n_layer-16'
    -> 131072), for an out-of-distribution window-length warning. Returns
    None for a --model override that doesn't follow this convention."""
    m = re.search(r"seqlen-(\d+)k", model_name)
    return int(m.group(1)) * 1024 if m else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Caduceus-Ph/Ps embeddings from DNA sequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--variant", required=True, choices=["ph", "ps"],
                        help="Caduceus-Ph (post-hoc, needs manual RC averaging — see module "
                             "docstring) or Caduceus-Ps (parameter-sharing, RC-equivariant, "
                             "doubles the embedding dim).")
    parser.add_argument("--model", default=None,
                        help=f"HF model name. Default per --variant: {MODEL_NAMES}")
    parser.add_argument("--strand", default="forward", choices=["forward", "reverse"],
                        help="Genomic strand of the input --ref-file/--mut-file — selects the "
                             "default input files and labels output. Default: forward.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Sequences per forward pass (default: 1 — see module docstring on padding).")
    parser.add_argument("--emb-type", default="average", choices=["average", "last"],
                        help="Pooling: 'average' = mean over the pooled region; 'last' = last token — "
                             "only valid with --pool-region full. (default: average)")
    parser.add_argument("--ref-file", default=None, metavar="FILE",
                        help="Path to reference sequences .npy file. Default: output/ref_seq_DNA_{strand}.npy")
    parser.add_argument("--mut-file", default=None, metavar="FILE",
                        help="Path to mutant sequences .npy file. Default: output/mut_seq_DNA_{strand}.npy")
    parser.add_argument("--pool-region", default="full", choices=["full", "downstream"],
                        help="'full' (default) pools over the whole sequence. 'downstream' pools a "
                             "window around the edit (locality window — Caduceus is bidirectional, "
                             "so there is no causal isolation) using the variant metadata CSV.")
    parser.add_argument("--downstream-k", nargs="+", default=["all"], type=parse_downstream_k, metavar="K",
                        help="Window sizes to test when --pool-region downstream (default: all).")
    parser.add_argument("--variant-meta-file", default=None, metavar="FILE",
                        help="Path to variant_meta.csv written by prepare_dataset.py. "
                             "Default: output/variant_meta.csv. Required for --pool-region downstream.")
    args = parser.parse_args()

    if args.pool_region == "downstream" and args.emb_type != "average":
        parser.error("--pool-region downstream only supports --emb-type average")
    return args


def extract_embeddings(
    sequences: list,
    tokenizer,
    model,
    df: str,
    emb_type: str,
    strand: str,
    out_dir: str,
    batch_size: int,
    pool_region: str,
    pad_id: int,
    downstream_ks: list = None,  # type: ignore
    edit_starts: list = None,  # type: ignore
) -> None:
    """Extract Caduceus's final hidden state, pooled per --pool-region/--emb-type,
    and save one .npy file per region (per k, if swept) into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    regions = downstream_ks if pool_region == "downstream" else [None]
    all_pooled = {r: [] for r in regions}

    for start in range(0, len(sequences), batch_size):
        seqs = [str(s) for s in sequences[start:start + batch_size]]
        token_ids = [tokenizer(s, add_special_tokens=False)["input_ids"] for s in seqs]
        lengths = [len(t) for t in token_ids]
        max_length = max(lengths)
        if len(set(lengths)) > 1:
            warnings.warn(
                "Batch has sequences of different lengths — padding is not masked out in "
                "Caduceus's Mamba layers, so shorter sequences' representations may be affected "
                "by padding. Use --batch-size 1 (default) if this matters."
            )
        padded = [t + [pad_id] * (max_length - len(t)) for t in token_ids]
        input_ids = torch.tensor(padded, dtype=torch.long, device=DEVICE)
        batch_starts = edit_starts[start:start + batch_size] if pool_region == "downstream" else None

        with torch.no_grad():
            hidden = model(input_ids, return_dict=True).last_hidden_state  # (B, max_length, D)

        for r in regions:
            pooled = (pool_downstream(hidden, lengths, batch_starts, r)
                      if pool_region == "downstream" else pool(hidden, lengths, emb_type))
            all_pooled[r].append(pooled.float().cpu().numpy())

        print(f"Processed {min(start + batch_size, len(sequences))}/{len(sequences)} sequences")

    for r in regions:
        combined = np.concatenate(all_pooled[r], axis=0)
        region_suffix = "" if r is None else f"ds{r}"
        fname = build_filename(emb_type, strand, region_suffix)
        out_path = os.path.join(out_dir, f"{df}_{fname}")
        np.save(out_path, combined)
        print(f"Saved {emb_type} embeddings region={region_suffix or 'full'}: "
              f"{combined.shape} -> {out_path}")


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model or MODEL_NAMES[args.variant]

    print(f"Using device: {DEVICE}")
    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(DEVICE).eval()
    print("Model loaded.\n")

    if bool(getattr(model.config, "rcps", False)) != (args.variant == "ps"):
        warnings.warn(
            f"--variant {args.variant} was requested, but {model_name}'s config.rcps="
            f"{getattr(model.config, 'rcps', None)} doesn't match (rcps=True <-> Ps, rcps=False <-> "
            "Ph) — double check --model against --variant if you overrode the default."
        )

    input_files = {
        "ref_seq": args.ref_file or f"output/ref_seq_DNA_{args.strand}.npy",
        "mut_seq": args.mut_file or f"output/mut_seq_DNA_{args.strand}.npy",
    }

    context_window = parse_context_window(input_files["ref_seq"])
    d_model = getattr(model.config, "d_model", "unknown")
    out_dir = os.path.join("embeddings", f"{MODEL_FAMILY}_{args.variant}_{d_model}dim_{context_window}_emb")

    pretrain_seqlen = parse_pretrain_seqlen(model_name)
    if pretrain_seqlen is not None:
        window_bp = int(context_window.rstrip("bp"))
        if window_bp > pretrain_seqlen:
            warnings.warn(
                f"--ref-file windows are {window_bp} bp, but {model_name} was pretrained at only "
                f"{pretrain_seqlen} bp context — embeddings from windows this much longer than "
                "pretraining are out-of-distribution for this checkpoint."
            )

    edit_start = ref_len = alt_len = None
    if args.pool_region == "downstream":
        meta = pd.read_csv(args.variant_meta_file or "output/variant_meta.csv")
        edit_start = meta["edit_start"].to_numpy()
        ref_len = meta["ref"].astype(str).str.len().to_numpy()
        alt_len = meta["alt"].astype(str).str.len().to_numpy()

    for df, path in input_files.items():
        seqs = np.load(path, allow_pickle=True)

        starts = None
        if args.pool_region == "downstream":
            allele_len = ref_len if df == "ref_seq" else alt_len
            if args.strand == "forward":
                # earliest position whose locality window is centered on the
                # allele = the allele's own last base (0-indexed).
                starts = (edit_start + allele_len - 1).tolist()  # type: ignore
            else:
                # Reverse-complementing flips reading direction, so the
                # allele's last base *in the reverse array's own order* is
                # the locus at forward position edit_start (the allele's
                # first base) — independent of allele length.
                seq_lens = np.array([len(s) for s in seqs])
                starts = (seq_lens - edit_start - 1).tolist()

        extract_embeddings(
            sequences=seqs,
            tokenizer=tokenizer,
            model=model,
            df=df,
            emb_type=args.emb_type,
            strand=args.strand,
            out_dir=out_dir,
            batch_size=args.batch_size,
            pool_region=args.pool_region,
            pad_id=tokenizer.pad_token_id,
            downstream_ks=args.downstream_k,
            edit_starts=starts,
        )
