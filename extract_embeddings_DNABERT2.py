"""
Extract DNABERT-2 embeddings from the ref_seq/mut_seq windows written by
prepare_dataset.py, pooled into one .npy file.

Unlike Evo2 (extract_embeddings_Evo2.py), DNABERT-2's trust_remote_code
forward pass only exposes its final hidden state (no per-layer
--layer/blocks.N selection), so there is a single output per emb_type/region
per input file.

DNABERT-2 tokenizes with BPE, not one-token-per-base, so downstream pooling
(--pool-region downstream) uses the tokenizer's offset mapping to translate
a base position from variant_meta.csv into the token index that first
covers it.

Input
  --ref-file       Reference sequences .npy. Default: output/ref_seq_DNA_forward.npy
  --mut-file       Mutant sequences .npy.    Default: output/mut_seq_DNA_forward.npy
                    Strand (forward/reverse) is read from the "_forward_"/"_reverse_"
                    token in --ref-file's name — it is not a separate argument.
  --batch-size     Sequences per forward pass. Default: 8.

Pooling
  --pool-region    'full' (default): pool over the whole sequence.
                    'downstream': pool only tokens after the edit — requires
                    --variant-meta-file.
  --emb-type       'average': mean over the pooled region. 'last': last token —
                    only valid with --pool-region full. Default: average.
  --downstream-k   Only used with --pool-region downstream. One or more window
                    sizes to test: an integer k means token positions
                    [start, start+k] inclusive, where start is the token that
                    first covers the allele's own last base, or 'all' for
                    everything to the sequence end. One output file is saved
                    per k, e.g. --downstream-k 0 32 128 512 all. Default: all.
  --variant-meta-file
                    variant_meta.csv written by prepare_dataset.py (columns
                    pos, ref, alt, edit_start), row-aligned with --ref-file/
                    --mut-file. Required for --pool-region downstream.
                    Default: output/variant_meta.csv.

Output
  Saved under a folder named DNABERT2_117M_{context_window}_emb/, e.g.
  DNABERT2_117M_8192bp_emb/ (context_window is parsed from the input
  filename — e.g. ref_seq_DNA_forward_8192bp.npy or
  variant_meta_8192bp.csv — so the --ref-file/--variant-meta-file name must
  contain a "<N>bp" token).
  One {ref_seq,mut_seq}_Lfinal_{emb_type}_{strand}[_ds{k}].npy per input
  (and per k, if swept), shape (N, D).

Example
-------
apptainer exec --nv --bind "$PWD":"$PWD" --pwd "$PWD" sif/dnabert2.sif python3 extract_embeddings_DNABERT2.py --ref-file output/20260831_143505/ref_seq_DNA_forward_2500bp.npy --mut-file mut_seq_DNA_forward_2500bp.npy 


"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

MODEL_NAME = "zhihan1996/DNABERT-2-117M"
MODEL_FAMILY = "DNABERT2"
PARAMS_LABEL = "117M"
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_context_window(path: str) -> str:
    """Extract the '<N>bp' context-window token from a filename written by
    prepare_dataset.py (e.g. ref_seq_DNA_forward_8192bp.npy or
    variant_meta_8192bp.csv), for use in the output folder name."""
    m = re.search(r"(\d+bp)", os.path.basename(path))
    if not m:
        raise ValueError(
            f"Could not find a '<N>bp' context-window token in filename {path!r}. "
            "Expected a name like '..._8192bp.npy' as written by prepare_dataset.py."
        )
    return m.group(1)


def parse_strand(path: str) -> str:
    """Extract the strand ('forward' or 'reverse') from a filename written by
    prepare_dataset.py (e.g. ref_seq_DNA_forward_8192bp.npy)."""
    m = re.search(r"forward|reverse", os.path.basename(path))
    if not m:
        raise ValueError(
            f"Could not find 'forward' or 'reverse' in filename {path!r}. "
            "Expected a name like '..._forward_8192bp.npy' as written by prepare_dataset.py."
        )
    return m.group(0)


def build_filename(mode: str, strand: str, region_suffix: str = "") -> str:
    suffix = f"_{region_suffix}" if region_suffix else ""
    return f"Lfinal_{mode}_{strand}{suffix}.npy"


def pool(hidden: torch.Tensor, attention_mask: torch.Tensor, emb_type: str) -> torch.Tensor:
    """Apply pooling to a (B, L, D) hidden state tensor over the whole sequence,
    respecting padding via attention_mask (B, L)."""
    mask = attention_mask.bool()
    if emb_type == "average":
        mask_expanded = mask.unsqueeze(-1).float()
        return (hidden * mask_expanded).sum(1) / mask_expanded.sum(1)
    else:  # last
        lengths = mask.sum(1)
        last_indices = (lengths - 1).clamp(min=0)
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last_indices]


def pool_downstream(hidden: torch.Tensor, attention_mask: torch.Tensor, starts: list[int], k) -> torch.Tensor:
    """Mean over token positions [start, start+k] per sequence (inclusive — k=0
    is just the token that first covers the allele's own last base; k == "all"
    means [start, length)), where `start` is a token index derived from the
    tokenizer's offset mapping (see main())."""
    B, L = hidden.shape[0], hidden.shape[1]
    lengths = attention_mask.sum(1).tolist()
    region_start = [min(s, l) for s, l in zip(starts, lengths)]
    region_end = lengths if k == "all" else [min(s + k + 1, l) for s, l in zip(starts, lengths)]
    if any(e <= s for s, e in zip(region_start, region_end)):
        print("  WARNING: empty downstream region for at least one sequence "
              "(edit sits at/past the sequence end) — that row's pooled vector will be all zeros")
    mask = torch.zeros((B, L), dtype=torch.bool, device=hidden.device)
    for b, (s, e) in enumerate(zip(region_start, region_end)):
        mask[b, s:e] = True
    mask_expanded = mask.unsqueeze(-1).float()
    return (hidden * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)


def token_start_for_position(offsets: list[tuple[int, int]], position: int) -> int:
    """Index of the first token whose span covers base `position` (0-indexed,
    end-exclusive spans; special tokens have span (0, 0) and are skipped).
    Falls back to the last real token if `position` is past every span."""
    last_real = 0
    for i, (start, end) in enumerate(offsets):
        if start == end == 0:
            continue
        last_real = i
        if end > position:
            return i
    return last_real


def _parse_k(value: str):
    if value == "all":
        return "all"
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--downstream-k values must be 'all' or an integer, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DNABERT-2 embeddings from DNA sequences.")
    parser.add_argument("--model", default=MODEL_NAME, help=f"HF model name. Default: {MODEL_NAME}")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Sequences per forward pass (default: {BATCH_SIZE})")
    parser.add_argument("--emb-type", default="average", choices=["average", "last"],
                        help="Pooling: 'average' = mean over the pooled region (the whole sequence "
                             "for --pool-region full, or [start, start+k] for downstream, where "
                             "start is the token covering the allele's own last base — use "
                             "--downstream-k 0 for just that token); 'last' = last token — only "
                             "valid with --pool-region full. (default: average)")
    parser.add_argument("--ref-file", default=None, metavar="FILE",
                        help="Path to reference sequences .npy file. "
                             "Default: output/ref_seq_DNA_forward.npy")
    parser.add_argument("--mut-file", default=None, metavar="FILE",
                        help="Path to mutant sequences .npy file. "
                             "Default: output/mut_seq_DNA_forward.npy")
    parser.add_argument("--pool-region", default="full", choices=["full", "downstream"],
                        help="'full' (default) pools over the whole sequence. 'downstream' "
                             "pools from the token covering the allele's own last base onward, "
                             "using the variant metadata CSV written by prepare_dataset.py.")
    parser.add_argument("--downstream-k", nargs="+", default=["all"], type=_parse_k, metavar="K",
                        help="Window sizes to test when --pool-region downstream: each is either "
                             "an integer k (token positions [start, start+k], inclusive) or the "
                             "literal 'all' (everything from start to the sequence end). One "
                             "output file is saved per k, e.g. --downstream-k 0 32 128 512 all. "
                             "Default: all")
    parser.add_argument("--variant-meta-file", default=None, metavar="FILE",
                        help="Path to the variant_meta.csv written by prepare_dataset.py (columns "
                             "pos, ref, alt, edit_start; row-aligned with --ref-file/--mut-file). "
                             "Default: output/variant_meta.csv. Required for --pool-region downstream.")
    args = parser.parse_args()

    if args.pool_region == "downstream" and args.emb_type != "average":
        parser.error("--pool-region downstream only supports --emb-type average "
                      "('last' isn't well-defined for a swept window — use "
                      "--downstream-k 0 for just the mutation-site embedding)")
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
    downstream_ks: list = None, # type: ignore
    edit_starts: list = None, # type: ignore
) -> None:
    """Extract DNABERT-2's final hidden state, pooled per --pool-region/--emb-type,
    and save one .npy file per region (per k, if swept) into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    regions = downstream_ks if pool_region == "downstream" else [None]  # None = full-sequence pool()
    all_pooled = {r: [] for r in regions}

    for start in range(0, len(sequences), batch_size):
        seqs = [str(s) for s in sequences[start:start + batch_size]]
        need_offsets = pool_region == "downstream"
        encoded = tokenizer(
            seqs, return_tensors="pt", padding=True,
            return_offsets_mapping=need_offsets,
        )
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)

        with torch.no_grad():
            hidden = model(input_ids, attention_mask=attention_mask)[0]  # B, L, D

        if pool_region == "downstream":
            batch_positions = edit_starts[start:start + batch_size]
            batch_starts = [
                token_start_for_position(offsets.tolist(), pos)
                for offsets, pos in zip(encoded["offset_mapping"], batch_positions)
            ]
            for r in regions:
                pooled = pool_downstream(hidden, attention_mask, batch_starts, r)
                all_pooled[r].append(pooled.float().cpu().numpy())
        else:
            pooled = pool(hidden, attention_mask, emb_type)
            all_pooled[None].append(pooled.float().cpu().numpy())

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

    print(f"Using device: {DEVICE}")
    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # DNABERT-2 ships a Triton flash-attention kernel that uses a removed API
    # (tl.dot(..., trans_b=True)) and fails to compile on Triton 2.x/3.x. Its
    # bert_layers.py takes the pure-PyTorch attention path whenever attention
    # dropout is nonzero, so set it here. eval() makes nn.Dropout an identity,
    # so embeddings are unchanged -- this only selects the attention implementation.
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.attention_probs_dropout_prob = 0.1
    model = AutoModel.from_pretrained(
        args.model, config=config, trust_remote_code=True
    ).to(DEVICE).eval()
    print("Model loaded.\n")

    input_files = {
        "ref_seq": args.ref_file or "output/ref_seq_DNA_forward.npy",
        "mut_seq": args.mut_file or "output/mut_seq_DNA_forward.npy",
    }

    strand = parse_strand(input_files["ref_seq"])
    context_window = parse_context_window(input_files["ref_seq"])
    out_dir = os.path.join("embeddings", f"{MODEL_FAMILY}_{PARAMS_LABEL}_{context_window}_emb")

    edit_start = ref_len = alt_len = None
    if args.pool_region == "downstream":
        meta = pd.read_csv(args.variant_meta_file or "output/variant_meta.csv")
        edit_start = meta["edit_start"].to_numpy()
        ref_len = meta["ref"].astype(str).str.len().to_numpy()
        alt_len = meta["alt"].astype(str).str.len().to_numpy()

    for df, path in input_files.items():
        seqs = np.load(path, allow_pickle=True)

        positions = None
        if args.pool_region == "downstream":
            if strand == "forward":
                # position = the allele's own last base (0-indexed) — ref allele
                # for ref_seq, alt allele for mut_seq, since their lengths can differ.
                allele_len = ref_len if df == "ref_seq" else alt_len
                positions = (edit_start + allele_len - 1).tolist() # type: ignore
            else:
                # Reverse-complementing flips reading direction, so the allele's
                # last base *in the reverse array's own left-to-right order* is
                # the locus at forward position edit_start (the allele's first
                # base) — independent of allele length, same for ref/mut.
                # position = seq_len - edit_start - 1
                seq_lens = np.array([len(s) for s in seqs])
                positions = (seq_lens - edit_start - 1).tolist()

        extract_embeddings(
            sequences=seqs,
            tokenizer=tokenizer,
            model=model,
            df=df,
            emb_type=args.emb_type,
            strand=strand,
            out_dir=out_dir,
            batch_size=args.batch_size,
            pool_region=args.pool_region,
            downstream_ks=args.downstream_k,
            edit_starts=positions, # type: ignore
        )
