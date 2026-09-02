"""
Extract JanusDNA embeddings from the ref_seq/mut_seq windows written by
prepare_dataset.py, pooled per layer into one .npy file each.

JanusDNA is NOT on PyPI or the HuggingFace Hub (unlike Evo2/DNABERT-2), so this
script needs two things Evo2/DNABERT-2 don't:
  --janusdna-repo   A local clone of https://github.com/Qihao-Duan/JanusDNA
                    (this script imports its `janusdna` package for
                    JanusDNAConfig/JanusDNAModel — pip install its `mamba_ssm`
                    wheel per that repo's README first, since JanusDNA's Mamba
                    layers require the compiled CUDA kernels).
  --checkpoint      A local pretrained-weights file downloaded from the
                    JanusDNA Harvard Dataverse release (doi:10.7910/DVN/HDT0RN
                    — not auto-downloaded by this script). vocab_size is
                    inferred from the checkpoint's embed_tokens.weight shape;
                    everything else about the architecture comes from
                    --model-size (see MODEL_PRESETS below).

Model sizes
  JanusDNA has no "small/base/large" parameter tiers the way DNABERT-2 or Evo2
  do. The paper/repo instead release 8-layer checkpoints at 3 hidden_size
  values (32/72/144), each optionally with a periodic mid-stack attention
  sublayer ("midattn") switched on or off — 5 released configs in total (no
  144dim+midattn checkpoint). An August 2025 update additionally offers the
  32dim configs with two extra MLP layers after the fusion point ("_mlp") and
  an optional reverse-complement-augmented ("rc") variant. Use --model-size to
  select one of the 5 base architectures (see MODEL_PRESETS for exact dims and
  the approximate published parameter counts for the two sizes that reported
  one: ~2M for 32dim, ~7.7M for 72dim; 144dim's count isn't published).

Why only single, pre-fusion layers are safe to extract
  JanusDNA reads each input twice — once forward, once with the token order
  reversed (NOT reverse-complemented; the flip is only in reading direction,
  applied to the same forward-strand embedding) — concatenated along the
  sequence axis into one 2L-length tensor. For every *released* checkpoint
  (layer_fusion=False, confirmed in configs/model/janusdna.yaml and every
  scripts/pre_train/*.sh), each of the num_hidden_layers decoder layers
  (BiJanusDNAMambaSeperateWrapper / BiJanusDNAAttentionWrapper in
  janusdna/modeling_janusdna.py) keeps the forward half and the
  reversed-reading half on entirely separate weights and only concatenates
  them back together — no mixing. The forward and reversed streams are joined
  exactly once, in a dedicated FinalAttention (flex-attention) fusion module
  that runs *after* the last decoder layer, followed by final_fusion() and
  final_mlp(). So any --layer index in [0, num_hidden_layers) is a clean,
  unmerged single-direction representation; this script never calls
  final_attention at all (it raises out of the forward pass right after the
  last requested hook fires), so it also sidesteps needing a working
  torch.compile'd flex_attention. (A hypothetical layer_fusion=True checkpoint
  *would* mix directions at every layer via BiJanusDNAMambaWrapper instead —
  none of the released checkpoints use this, but if you ever load one, no
  layer is safe to extract from.)

Input
  --ref-file       Reference sequences .npy. Default: output/ref_seq_DNA_forward.npy
  --mut-file       Mutant sequences .npy.    Default: output/mut_seq_DNA_forward.npy
  --strand         forward or reverse — selects the default input files and labels
                    output. Default: forward.
  --batch-size     Sequences per forward pass. Default: 1 — JanusDNA's Mamba
                    layers don't take an attention mask, so padding a batch of
                    different-length sequences would leak into the state-space
                    scan of the shorter ones. Only raise this if every sequence
                    in your input file has the same length (e.g. fixed-width
                    windows from prepare_dataset.py); the script warns and
                    still pads (unmasked) if lengths differ within a batch.

Layers / direction
  --layer          One or more decoder-layer indices (0-indexed into
                    model.layers), extracted together in a single forward
                    pass. Default: last layer (num_hidden_layers - 1).
  --direction      'forward' (default): the causal, left-to-right reading of
                    the sequence as given. 'backward': the causal reading of
                    the reversed sequence, flipped back to the original base
                    order — i.e. each position's representation from having
                    only seen bases to its right. Both are single-direction
                    and safe; they are just two different unmerged views.

Pooling
  --pool-region    'full' (default): pool over the whole sequence.
                    'downstream': pool only positions after the edit — requires
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
  Saved under embeddings/JanusDNA_{model-size}_{context_window}_emb/, e.g.
  JanusDNA_72dim_nomidattn_2500bp_emb/ (context_window is parsed from the
  input filename — e.g. ref_seq_DNA_forward_2500bp.npy — so the
  --ref-file/--variant-meta-file name must contain a "<N>bp" token).
  One {ref_seq,mut_seq}_L{layer}_{direction}_{emb_type}_{strand}[_ds{k}].npy
  per layer (and per k, if swept), shape (N, D).

Example
-------
python extract_embeddings_JanusDNA.py \\
    --janusdna-repo /path/to/JanusDNA \\
    --checkpoint /path/to/janusdna_72dim_nomidattn.pt \\
    --model-size 72dim_nomidattn \\
    --layer 6 7 \\
    --ref-file output/20260831_143505/ref_seq_DNA_forward_2500bp.npy \\
    --mut-file output/20260831_143505/mut_seq_DNA_forward_2500bp.npy
"""

import argparse
import os
import sys
import warnings
from functools import partial

import numpy as np
import pandas as pd
import torch

from embedding_utils import parse_context_window, parse_downstream_k, pool, pool_downstream

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# hidden_size / intermediate_size / flex_attn_n_embd / attn_layer_offset come
# straight from scripts/pre_train/slurm_JanusDNA_*.sh; attn_layer_offset=100
# (vs. attn_layer_period=8) is how the "nomidattn" runs disable the periodic
# mid-stack attention sublayer entirely (i % 8 never equals 100), leaving all
# 8 layers as plain Mamba+MoE. params_label is the size self-reported in
# JanusDNA's README benchmark tables where published; "unpublished" otherwise.
MODEL_PRESETS = {
    "32dim_midattn":    dict(hidden_size=32,  intermediate_size=128, flex_attn_n_embd=64,  attn_layer_offset=4,   params_label="~1.98M"),
    "32dim_nomidattn":  dict(hidden_size=32,  intermediate_size=128, flex_attn_n_embd=64,  attn_layer_offset=100, params_label="~1.99M"),
    "72dim_midattn":    dict(hidden_size=72,  intermediate_size=288, flex_attn_n_embd=128, attn_layer_offset=4,   params_label="~7.66M"),
    "72dim_nomidattn":  dict(hidden_size=72,  intermediate_size=288, flex_attn_n_embd=128, attn_layer_offset=100, params_label="~7.75M"),
    "144dim_nomidattn": dict(hidden_size=144, intermediate_size=576, flex_attn_n_embd=256, attn_layer_offset=100, params_label="unpublished"),
}
N_LAYER = 8  # fixed across every released checkpoint

# CharacterTokenizer scheme from src/dataloaders/datasets/hg38_char_tokenizer.py
# (characters=['A','C','G','T','N'], ids assigned after 7 reserved specials).
SPECIAL_IDS = {"[CLS]": 0, "[SEP]": 1, "[BOS]": 2, "[MASK]": 3, "[PAD]": 4, "[RESERVED]": 5, "[UNK]": 6}
CHAR_IDS = {"A": 7, "C": 8, "G": 9, "T": 10, "N": 11}
PAD_ID = SPECIAL_IDS["[PAD]"]
UNK_ID = SPECIAL_IDS["[UNK]"]


class _StopForward(Exception):
    """Raised from the last requested layer's hook to skip the (unneeded,
    expensive) final-fusion attention and everything after it."""


def tokenize(seq: str) -> list[int]:
    return [CHAR_IDS.get(base, UNK_ID) for base in seq.upper()]


def build_filename(layer: int, direction: str, mode: str, strand: str, region_suffix: str = "") -> str:
    suffix = f"_{region_suffix}" if region_suffix else ""
    return f"L{layer}_{direction}_{mode}_{strand}{suffix}.npy"


def pool_upstream(hidden: torch.Tensor, lengths: list[int], starts: list[int], k) -> torch.Tensor:
    """Mirror of pool_downstream for JanusDNA's --direction backward stream:
    that stream is causal in reverse (position i has seen positions [i, end)),
    so it has seen the edit only at positions <= the edit's own first base —
    i.e. the *upstream* side. Mean over positions [start-k, start] per
    sequence (inclusive — k=0 is just the allele's own first base; k == "all"
    means [0, start])."""
    B, L = hidden.shape[0], hidden.shape[1]
    region_end = [min(s + 1, l) for s, l in zip(starts, lengths)]
    region_start = [0 if k == "all" else max(s - k, 0) for s in starts]
    if any(e <= s for s, e in zip(region_start, region_end)):
        print("  WARNING: empty upstream region for at least one sequence "
              "(edit sits at/before the sequence start) — that row's pooled vector will be all zeros")
    mask = torch.zeros((B, L), dtype=torch.bool, device=hidden.device)
    for b, (s, e) in enumerate(zip(region_start, region_end)):
        mask[b, s:e] = True
    mask_expanded = mask.unsqueeze(-1).float()
    return (hidden * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)


def build_config(model_size: str, vocab_size: int, JanusDNAConfig):
    p = MODEL_PRESETS[model_size]
    config = JanusDNAConfig(
        vocab_size=vocab_size,
        hidden_size=p["hidden_size"],
        intermediate_size=p["intermediate_size"],
        num_hidden_layers=N_LAYER,
        num_attention_heads=4,
        attn_layer_period=8,
        attn_layer_offset=p["attn_layer_offset"],
        expert_layer_period=2,
        expert_layer_offset=1,
        flex_attn_n_embd=p["flex_attn_n_embd"],
        bidirectional=True,
        bidirectional_strategy="add",
        bidirectional_weight_tie=True,
        bidirectional_attn_tie=False,
        layer_fusion=False,  # every released checkpoint — see module docstring
        final_attention=True,
        layer_fusion_strategy="pool",
        mid_single_direction_attention=True,
        final_attention_class="flex_attention",
        use_cache=False,
        gradient_checkpointing=False,
    )
    # Only used by final_attention (never invoked — see _StopForward), and by
    # mid-stack attention sublayers on *_midattn presets. sdpa avoids needing
    # flash-attn / a working torch.compile'd flex_attention just to hook
    # earlier layers.
    config._attn_implementation = "sdpa"
    return config


def load_checkpoint(model, checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    model_keys = set(model.state_dict().keys())
    if not (set(state_dict.keys()) & model_keys):
        # Common wrapper prefixes (e.g. Lightning's "model.model." or
        # DNAEmbeddingModelJanusDNA's "model.") — strip until keys line up.
        for prefix in ("model.model.", "model."):
            stripped = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
            if set(stripped.keys()) & model_keys:
                state_dict = stripped
                break

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint {checkpoint_path}: {len(missing)} missing keys, {len(unexpected)} unexpected keys.")
    if missing:
        print(f"  missing (first 5): {missing[:5]}")
    if unexpected:
        print(f"  unexpected (first 5): {unexpected[:5]}")


def infer_vocab_size(checkpoint_path: str) -> int:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    for key, tensor in state_dict.items():
        if key.endswith("embed_tokens.weight"):
            return tensor.shape[0]
    default = len(SPECIAL_IDS) + len(CHAR_IDS)
    print(f"WARNING: could not find embed_tokens.weight in checkpoint to infer vocab_size; "
          f"defaulting to {default}.")
    return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract JanusDNA embeddings from DNA sequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--janusdna-repo", required=True,
                        help="Path to a local clone of github.com/Qihao-Duan/JanusDNA (for its `janusdna` package).")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a local pretrained-weights file (see module docstring).")
    parser.add_argument("--model-size", default="72dim_nomidattn", choices=list(MODEL_PRESETS),
                        help="Architecture preset (default: 72dim_nomidattn). See MODEL_PRESETS.")
    parser.add_argument("--layer", nargs="+", type=int, default=[N_LAYER - 1],
                        help=f"Decoder-layer indices (0-indexed, valid range [0, {N_LAYER})) to extract "
                             f"together in one forward pass. Default: {N_LAYER - 1} (last layer).")
    parser.add_argument("--direction", default="forward", choices=["forward", "backward"],
                        help="'forward' (default): causal left-to-right reading. 'backward': causal "
                             "reading of the reversed sequence, flipped back to original base order.")
    parser.add_argument("--strand", default="forward", choices=["forward", "reverse"],
                        help="Genomic strand of the input --ref-file/--mut-file — selects the default "
                             "input files and labels output. Default: forward. (Not to be confused with "
                             "--direction, which is JanusDNA's internal dual-reading-direction stream.)")
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
                        help="'full' (default) pools over the whole sequence. 'downstream' pools only "
                             "the positions that have actually seen the edit in --direction's causal "
                             "order: from the allele's own last base onward for --direction forward, or "
                             "from the sequence start up to the allele's own first base for --direction "
                             "backward — using the variant metadata CSV.")
    parser.add_argument("--downstream-k", nargs="+", default=["all"], type=parse_downstream_k, metavar="K",
                        help="Window sizes to test when --pool-region downstream (default: all).")
    parser.add_argument("--variant-meta-file", default=None, metavar="FILE",
                        help="Path to variant_meta.csv written by prepare_dataset.py. "
                             "Default: output/variant_meta.csv. Required for --pool-region downstream.")
    args = parser.parse_args()

    if args.pool_region == "downstream" and args.emb_type != "average":
        parser.error("--pool-region downstream only supports --emb-type average")
    for layer in args.layer:
        if not (0 <= layer < N_LAYER):
            parser.error(f"--layer {layer} out of range [0, {N_LAYER})")
    return args


def extract_embeddings(
    sequences: list,
    model,
    df: str,
    emb_type: str,
    layers: list[int],
    direction: str,
    strand: str,
    out_dir: str,
    batch_size: int,
    pool_region: str,
    downstream_ks: list = None,
    edit_starts: list = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    max_layer = max(layers)
    regions = downstream_ks if pool_region == "downstream" else [None]
    all_pooled = {(layer, r): [] for layer in layers for r in regions}

    for start in range(0, len(sequences), batch_size):
        seqs = [str(s) for s in sequences[start:start + batch_size]]
        token_ids = [tokenize(s) for s in seqs]
        lengths = [len(t) for t in token_ids]
        max_length = max(lengths)
        if len(set(lengths)) > 1:
            warnings.warn(
                "Batch has sequences of different lengths — padding is not masked out in JanusDNA's "
                "Mamba layers, so shorter sequences' representations may be affected by padding. "
                "Use --batch-size 1 (default) if this matters."
            )
        padded = [t + [PAD_ID] * (max_length - len(t)) for t in token_ids]
        input_ids = torch.tensor(padded, dtype=torch.long, device=DEVICE)
        batch_starts = edit_starts[start:start + batch_size] if pool_region == "downstream" else None

        captured = {}

        def hook(module, inp, out, layer_idx):
            captured[layer_idx] = out[0].detach()
            if layer_idx == max_layer:
                raise _StopForward()

        handles = [model.layers[i].register_forward_hook(partial(hook, layer_idx=i)) for i in layers]
        try:
            with torch.no_grad():
                model(input_ids, return_dict=False)
        except _StopForward:
            pass
        finally:
            for h in handles:
                h.remove()

        for layer in layers:
            hidden_2l = captured[layer]  # (B, 2*max_length, D)
            if direction == "forward":
                hidden = hidden_2l[:, :max_length, :]
            else:
                hidden = hidden_2l[:, max_length:, :].flip(dims=[1])

            for r in regions:
                if pool_region == "downstream":
                    # forward stream has seen the edit at/after its own last base;
                    # backward stream has seen it at/before its own first base (see
                    # pool_upstream docstring) — edit_starts is precomputed to match.
                    pool_fn = pool_downstream if direction == "forward" else pool_upstream
                    pooled = pool_fn(hidden, lengths, batch_starts, r)
                else:
                    pooled = pool(hidden, lengths, emb_type)
                all_pooled[(layer, r)].append(pooled.float().cpu().numpy())

        print(f"Processed {min(start + batch_size, len(sequences))}/{len(sequences)} sequences")

    for layer in layers:
        for r in regions:
            combined = np.concatenate(all_pooled[(layer, r)], axis=0)
            region_suffix = "" if r is None else f"ds{r}"
            fname = build_filename(layer, direction, emb_type, strand, region_suffix)
            out_path = os.path.join(out_dir, f"{df}_{fname}")
            np.save(out_path, combined)
            print(f"Saved {emb_type} embeddings layer {layer} direction={direction} "
                  f"region={region_suffix or 'full'}: {combined.shape} -> {out_path}")


if __name__ == "__main__":
    args = parse_args()

    sys.path.insert(0, args.janusdna_repo)
    from janusdna.configuration_janusdna import JanusDNAConfig
    from janusdna.modeling_janusdna import JanusDNAModel

    print(f"Using device: {DEVICE}")
    vocab_size = infer_vocab_size(args.checkpoint)
    config = build_config(args.model_size, vocab_size, JanusDNAConfig)
    print(f"Building JanusDNAModel ({args.model_size}, vocab_size={vocab_size}) ...")
    model = JanusDNAModel(config).to(DEVICE).eval()
    load_checkpoint(model, args.checkpoint)
    print("Model loaded.\n")

    input_files = {
        "ref_seq": args.ref_file or f"output/ref_seq_DNA_{args.strand}.npy",
        "mut_seq": args.mut_file or f"output/mut_seq_DNA_{args.strand}.npy",
    }

    context_window = parse_context_window(input_files["ref_seq"])
    out_dir = os.path.join("embeddings", f"JanusDNA_{args.model_size}_{context_window}_emb")

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
            allele_len = ref_len if df == "ref_seq" else alt_len
            seq_lens = np.array([len(s) for s in seqs])
            if args.direction == "forward":
                # earliest position whose forward-causal context has fully seen
                # the allele = the allele's own last base, in this array's own
                # left-to-right order.
                if args.strand == "forward":
                    starts = (edit_start + allele_len - 1).tolist()
                else:
                    # Reverse-complementing flips reading direction, so the
                    # allele's last base *in the reverse array's own order* is
                    # the locus at forward position edit_start (the allele's
                    # first base) — independent of allele length.
                    starts = (seq_lens - edit_start - 1).tolist()
            else:
                # backward stream sees positions [i, end) in this array's own
                # order, so it has fully seen the allele only once i is at/before
                # the allele's own first base.
                if args.strand == "forward":
                    starts = edit_start.tolist()
                else:
                    starts = (seq_lens - edit_start - allele_len).tolist()

        extract_embeddings(
            sequences=seqs,
            model=model,
            df=df,
            emb_type=args.emb_type,
            layers=args.layer,
            direction=args.direction,
            strand=args.strand,
            out_dir=out_dir,
            batch_size=args.batch_size,
            pool_region=args.pool_region,
            downstream_ks=args.downstream_k,
            edit_starts=starts,
        )
