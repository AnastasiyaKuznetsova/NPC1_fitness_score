"""
Extract zero-shot Evo2 embeddings from the ref_seq/mut_seq windows written by
prepare_dataset.py, pooled per layer into one .npy file each.

Model / layers
  --model          Evo2 variant: evo2_1b_base, evo2_7b (default), evo2_40b.
  --layer          One or more layer names (e.g. blocks.28.mlp.l3), extracted
                    together in a single forward pass. Default: blocks.28.mlp.l3.

Input
  --ref-file       Reference sequences .npy. Default: output/ref_seq_DNA_{strand}.npy
  --mut-file       Mutant sequences .npy.    Default: output/mut_seq_DNA_{strand}.npy
  --strand         forward or reverse — selects the default input files and labels
                    output. Default: forward.
  --batch-size     Sequences per forward pass. Default: 1.

Pooling
  --pool-region    'full' (default): pool over the whole sequence.
                    'downstream': pool only positions after the edit — requires
                    --variant-meta-file.
  --emb-type       'average': mean over the pooled region. 'last': last token —
                    only valid with --pool-region full. Default: average.
  --downstream-k   Only used with --pool-region downstream. One or more window
                    sizes to test: an integer k means positions [start, start+k]
                    inclusive, where start is the allele's own last base — the
                    earliest position whose causal context has fully seen the
                    edit (k=0 = just that position), or 'all' for everything to
                    the sequence end. One output file is saved per k, e.g.
                    --downstream-k 0 32 128 512 all. Default: all.
  --variant-meta-file
                    variant_meta.csv written by prepare_dataset.py (columns
                    pos, ref, alt, edit_start), row-aligned with --ref-file/
                    --mut-file. Required for --pool-region downstream.
                    Default: output/variant_meta.csv.

Output
  Saved under a folder named {model_family}_{params}_{context_window}_emb, e.g.
  Evo2_7B_8192bp_emb/ (context_window is parsed from the input filename — e.g.
  ref_seq_DNA_forward_8192bp.npy or variant_meta_8192bp.csv — not a fixed
  default, so the --ref-file/--variant-meta-file name must contain a
  "<N>bp" token).
  One {ref_seq,mut_seq}_L{layer}_{emb_type}_{strand}[_ds{k}].npy
  per layer (and per k, if swept), shape (N, D). model_family/params aren't
  repeated in the filename since they're already in the folder name.
"""

import argparse
import os
import re

import torch
import numpy as np
import pandas as pd

DEFAULT_MODEL_ID = "evo2_7b"
DEFAULT_LAYER_NAME = "blocks.28.mlp.l3"
BATCH_SIZE = 1
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

EVO2_MODEL_CONFIG = {
    "evo2_1b_base": {"num_layers": 25, "params": "1B"},
    "evo2_7b":      {"num_layers": 32, "params": "7B"},
    "evo2_40b":     {"num_layers": 50, "params": "40B"},
}

ALL_MODEL_CHOICES = list(EVO2_MODEL_CONFIG)


def layer_index(layer: str) -> str:
    m = re.search(r"(?:blocks|layers)\.(\d+)\.", layer)
    return m.group(1) if m else layer


def build_filename(layer: str, mode: str, strand: str, region_suffix: str = "") -> str:
    suffix = f"_{region_suffix}" if region_suffix else ""
    return f"L{layer_index(layer)}_{mode}_{strand}{suffix}.npy"


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


def pool(hidden: torch.Tensor, lengths: list[int], emb_type: str) -> torch.Tensor:
    """Apply pooling to a (B, L, D) hidden state tensor over the full sequence."""
    if emb_type == "average":
        mask = torch.zeros((hidden.shape[0], hidden.shape[1]), dtype=torch.bool, device=hidden.device)
        for b, l in enumerate(lengths):
            mask[b, :l] = True
        mask_expanded = mask.unsqueeze(-1).float()
        return (hidden * mask_expanded).sum(1) / mask_expanded.sum(1)
    else:  # last
        last_indices = torch.tensor([l - 1 for l in lengths], device=hidden.device)
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last_indices]


def pool_downstream(hidden: torch.Tensor, lengths: list[int], starts: list[int], k) -> torch.Tensor:
    """Mean over positions [start, start+k] per sequence (inclusive — k=0 is
    just the allele's own last base; k == "all" means [start, length)), where
    `start` is the earliest position whose causal context has fully seen the
    allele (see main() for how forward/reverse starts are derived from
    prepare_dataset.py's variant_meta.csv)."""
    B, L = hidden.shape[0], hidden.shape[1]
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


# ── Evo2 ────────────────────────────────────────────────────────────────────────

def load_evo2(model_name: str):
    """Returns (model, pad_id, params_label). model exposes .tokenizer.tokenize(seq)
    and .forward(input_ids, return_embeddings=True, layer_names=[...])."""
    from evo2 import Evo2
    from transformer_engine.common.recipe import (  # type: ignore[attr-defined]
        _OverrideLinearPrecision, DelayedScaling, Format,
    )
    torch.serialization.add_safe_globals([_OverrideLinearPrecision, DelayedScaling, Format])

    cfg = EVO2_MODEL_CONFIG[model_name]
    print(f"Loading {model_name} ...")
    model = Evo2(model_name)
    print("Model loaded.\n")
    # Evo2's tokenizer is a raw-byte tokenizer (no small fixed vocab / dedicated
    # pad id) — any unused byte works as filler since pool() masks it out by length.
    return model, 0, cfg["params"]


def load_model(model_name: str):
    """Returns (model, pad_id, params_label, model_family)."""
    if model_name in EVO2_MODEL_CONFIG:
        model, pad_id, params = load_evo2(model_name)
        return model, pad_id, params, "Evo2"
    raise ValueError(f"Unknown model {model_name!r}. Choose from: {ALL_MODEL_CHOICES}")


def _parse_k(value: str):
    if value == "all":
        return "all"
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--downstream-k values must be 'all' or an integer, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract zero-shot embeddings from DNA/RNA sequences.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, choices=ALL_MODEL_CHOICES,
                        help=f"Model variant (default: {DEFAULT_MODEL_ID}). Evo2: "
                             f"{list(EVO2_MODEL_CONFIG)}.")
    parser.add_argument("--layer", nargs="+", default=[DEFAULT_LAYER_NAME],
                        help="One or more layer names to extract. All layers are extracted in a "
                             f"single forward pass. (default: {DEFAULT_LAYER_NAME}).")
    parser.add_argument("--strand", default="forward", choices=["forward", "reverse"],
                        help="Strand direction — selects input file and labels output (default: forward)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Sequences per forward pass (default: {BATCH_SIZE})")
    parser.add_argument("--emb-type", default="average", choices=["average", "last"],
                        help="Pooling: 'average' = mean over the pooled region (the whole sequence "
                             "for --pool-region full, or [start, start+k] for downstream, where "
                             "start is the allele's own last base — use --downstream-k 0 for just "
                             "that position); 'last' = last token — only valid with --pool-region "
                             "full. (default: average)")
    parser.add_argument("--ref-file", default=None, metavar="FILE",
                        help="Path to reference sequences .npy file. "
                             "Default: output/ref_seq_DNA_{strand}.npy")
    parser.add_argument("--mut-file", default=None, metavar="FILE",
                        help="Path to mutant sequences .npy file. "
                             "Default: output/mut_seq_DNA_{strand}.npy")
    parser.add_argument("--pool-region", default="full", choices=["full", "downstream"],
                        help="'full' (default) pools over the whole sequence. 'downstream' "
                             "pools from the allele's own last base onward, using the variant "
                             "metadata CSV written by prepare_dataset.py.")
    parser.add_argument("--downstream-k", nargs="+", default=["all"], type=_parse_k, metavar="K",
                        help="Window sizes to test when --pool-region downstream: each is either "
                             "an integer k (positions [start, start+k], inclusive, where start is "
                             "the allele's own last base — the earliest position whose causal "
                             "context has fully seen the edit; k=0 means just that position) or "
                             "the literal 'all' (everything from start to the sequence end). One "
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
    sequences: list[str],
    model,
    df: str,
    emb_type: str,
    layers: list[str],
    strand: str,
    out_dir: str,
    batch_size: int = BATCH_SIZE,
    pad_id: int = 0,
    pool_region: str = "full",
    downstream_ks: list = None,
    edit_starts: list = None,
) -> None:
    """
    Extract embeddings from one or more layers in a single forward pass per batch.
    Saves one .npy file per layer (and, for pool_region="downstream", per k in downstream_ks)
    into out_dir (named {model_family}_{params}_{context_window}_emb by the caller).
    """
    os.makedirs(out_dir, exist_ok=True)
    regions = downstream_ks if pool_region == "downstream" else [None]  # None = full-sequence pool()
    all_pooled = {(layer, r): [] for layer in layers for r in regions}

    for start in range(0, len(sequences), batch_size):
        seqs = sequences[start:start + batch_size]
        token_ids = [model.tokenizer.tokenize(seq) for seq in seqs]
        lengths = [len(t) for t in token_ids]
        max_length = max(lengths)
        padded = [t + [pad_id] * (max_length - len(t)) for t in token_ids]
        input_ids = torch.tensor(padded, dtype=torch.int).to(DEVICE)
        batch_starts = edit_starts[start:start + batch_size] if pool_region == "downstream" else None

        with torch.no_grad():
            _, embeddings = model.forward(input_ids, return_embeddings=True, layer_names=layers)
            for layer in layers:
                hidden = embeddings[layer]  # B, L, D
                for r in regions:
                    pooled = (pool_downstream(hidden, lengths, batch_starts, r)
                              if pool_region == "downstream" else pool(hidden, lengths, emb_type))
                    all_pooled[(layer, r)].append(pooled.float().cpu().numpy())

        print(f"Processed {min(start + batch_size, len(sequences))}/{len(sequences)} sequences")

    for layer in layers:
        for r in regions:
            combined = np.concatenate(all_pooled[(layer, r)], axis=0)
            region_suffix = "" if r is None else f"ds{r}"
            fname = build_filename(layer, emb_type, strand, region_suffix)
            out_path = os.path.join(out_dir, f"{df}_{fname}")
            np.save(out_path, combined)
            print(f"Saved {emb_type} embeddings layer {layer_index(layer)} region={region_suffix or 'full'}: "
                  f"{combined.shape} -> {out_path}")


if __name__ == "__main__":
    args = parse_args()

    layers = args.layer

    model, pad_id, params, model_family = load_model(args.model)

    input_files = {
        "ref_seq": args.ref_file or f"output/ref_seq_DNA_{args.strand}.npy",
        "mut_seq": args.mut_file or f"output/mut_seq_DNA_{args.strand}.npy",
    }

    context_window = parse_context_window(input_files["ref_seq"])
    out_dir = f"{model_family}_{params}_{context_window}_emb"

    edit_start = ref_len = alt_len = None
    if args.pool_region == "downstream":
        meta = pd.read_csv(args.variant_meta_file or "output/variant_meta.csv")
        edit_start = meta["edit_start"].to_numpy()
        ref_len = meta["ref"].astype(str).str.len().to_numpy()
        alt_len = meta["alt"].astype(str).str.len().to_numpy()

    for df, path in input_files.items():
        seqs = np.load(path)

        starts = None
        if args.pool_region == "downstream":
            if args.strand == "forward":
                # start = the allele's own last base (the earliest position whose
                # causal context has seen the whole allele) — ref allele for
                # ref_seq, alt allele for mut_seq, since their lengths can differ.
                allele_len = ref_len if df == "ref_seq" else alt_len
                starts = (edit_start + allele_len - 1).tolist()
            else:
                # Reverse-complementing flips reading direction, so the allele's
                # last base *in the reverse array's own left-to-right order* is
                # the locus at forward position edit_start (the allele's first
                # base) — independent of allele length, same for ref/mut.
                # start = seq_len - edit_start - 1
                seq_lens = np.array([len(s) for s in seqs])
                starts = (seq_lens - edit_start - 1).tolist()

        extract_embeddings(
            sequences=seqs,
            model=model,
            df=df,
            emb_type=args.emb_type,
            layers=layers,
            out_dir=out_dir,
            strand=args.strand,
            batch_size=args.batch_size,
            pad_id=pad_id,
            pool_region=args.pool_region,
            downstream_ks=args.downstream_k,
            edit_starts=starts,
        )
