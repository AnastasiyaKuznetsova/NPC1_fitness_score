"""
Extract AlphaGenome embeddings from the ref_seq/mut_seq windows written by
prepare_dataset.py, pooled per extraction point into one .npy file each.

AlphaGenome needs local weights, not the hosted API
---------------------------------------------------
The public AlphaGenome API (`pip install alphagenome`, dna_client) only returns
*track predictions* and *variant scores* — it never returns hidden states. To
get embeddings you must run the model yourself from the research release:

    pip install git+https://github.com/google-deepmind/alphagenome_research.git

and pull the weights (non-commercial terms, immediate approval) from either
HuggingFace (google/alphagenome-all-folds) or Kaggle (google/alphagenome/jax/
all_folds). This script wraps
`alphagenome_research.model.dna_model.create_from_{huggingface,kaggle}` /
`create(checkpoint_path)`, so you need JAX + an H100-class GPU (DeepMind's own
stated minimum); --device cpu works for a handful of sequences but is slow.

Unlike Evo2/JanusDNA this is a JAX/Haiku model, so there is no `--layer`
flag indexing into a stack of identical decoder blocks. AlphaGenome is a U-Net
(conv encoder -> transformer tower -> conv decoder) and it exposes three
*named* embedding tensors, selected with --extract-from.


WHICH BLOCKS THE EMBEDDINGS COME FROM
=====================================
Traced from alphagenome_research/model/model.py (AlphaGenome.forward_trunk)
and .../model/embeddings.py. Shapes below are for an input of S bases.

  one-hot DNA (B, S, 4)
    |
    | -- SequenceEncoder (model.py:SequenceEncoder) ------------------------
    |      convolutions.DnaEmbedder        -> (B, S,      768)   1bp bins
    |      pool /2 + DownResBlock 0        -> (B, S/2,    896)   2bp bins
    |      pool /2 + DownResBlock 1        -> (B, S/4,   1024)   4bp bins
    |      pool /2 + DownResBlock 2        -> (B, S/8,   1152)   8bp bins
    |      pool /2 + DownResBlock 3        -> (B, S/16,  1280)  16bp bins
    |      pool /2 + DownResBlock 4        -> (B, S/32,  1408)  32bp bins
    |      pool /2 + DownResBlock 5        -> (B, S/64,  1536)  64bp bins
    |      pool /2                         -> (B, S/128, 1536) 128bp bins
    |      (each DownResBlock output is also kept as a U-Net skip)
    |    + organism embedding (human=0 / mouse=1) added to the trunk
    v
  TransformerTower (model.py:TransformerTower) — 9 blocks at 128bp resolution,
  each = MHABlock + MLPBlock, with a PairUpdateBlock/AttentionBiasBlock
  injected on every even block. Returns:
      trunk          (B, S/128, 1536)        <- 1-D sequence representation
      pair_x         (B, S/2048, S/2048, F)  <- 2-D pairwise representation
    |
    +---- OutputEmbedder(trunk)    == [--extract-from 128bp] ===============
    |       Linear(2*1536) -> RMSBatchNorm -> +organism emb -> gelu
    |       => embeddings_128bp    (B, S/128, 3072)
    |       This is the transformer tower's output — the deepest, most
    |       long-range-aware 1-D representation in the model, and the
    |       recommended default for variant-effect regressors.
    |
    +---- SequenceDecoder (model.py:SequenceDecoder) ----------------------
    |       7 x convolutions.UpResBlock, upsampling 128 -> 64 -> 32 -> 16
    |       -> 8 -> 4 -> 2 -> 1 bp, each adding back the matching encoder
    |       skip => (B, S, 768)
    |     then OutputEmbedder(x, skip_x=embeddings_128bp)
    |                                == [--extract-from 1bp] ===============
    |       Linear(2*768) + upsampled Linear(embeddings_128bp) ->
    |       RMSBatchNorm -> +organism emb -> gelu
    |       => embeddings_1bp      (B, S, 1536)
    |       Base-resolution; carries both the local conv detail and (via the
    |       skip) the transformer context. Use when you care about the exact
    |       mutated base rather than its 128bp neighbourhood.
    |
    +---- OutputPair(pair_x)         == [--extract-from pair] ==============
            symmetrise -> RMS LayerNorm -> +organism emb -> gelu
            => embeddings_pair     (B, S/2048, S/2048, 128)
            2-D contact-style representation at 2048bp resolution. Only
            informative for large windows — an 8192bp window is 4 bins, so
            this is here for completeness, not as a default.

  (The heads — RNA_SEQ, ATAC, CAGE, CHIP_*, SPLICE_*, CONTACT_MAPS — are
  *downstream* of these three tensors and are not touched by this script.)

The three tensors above are the only ones AlphaGenome's supported trunk API
returns. The encoder/transformer intermediates in the diagram are local
variables inside the Haiku modules, so --extract-from internal reaches them
with a `hk.experimental.intercept_methods` tap instead — see below.

  --extract-from internal + --internal-module SUBSTR
      Captures the output of every Haiku module whose name contains SUBSTR
      during the trunk forward pass, e.g.
        dna_embedder      -> (B, S,     768)   1bp   pre-pool
        downres_block_0   -> (B, S/2,   896)   2bp
        ...
        downres_block_5   -> (B, S/64, 1536)  64bp
        mha_block         -> per-transformer-block attention output
      This is NOT part of AlphaGenome's public API — module paths come from
      Haiku's auto-naming and can change between releases. The tap cannot run
      under jax.jit (an interceptor inside a traced function only ever sees
      tracers), so internal mode runs eagerly and is markedly slower. If
      SUBSTR matches nothing, the script prints every module name it saw so
      you can pick one. Bin size is inferred as model_seq_length // S_tap, so
      cropping/pooling stays correct whatever resolution you tap.


Input length: AlphaGenome takes a FIXED-LENGTH sequence
-------------------------------------------------------
The model only accepts 16KB / 100KB / 500KB / 1MB inputs (2**14, 2**17, 2**19,
2**20; see alphagenome.models.dna_client.SUPPORTED_SEQUENCE_LENGTHS). Your
prepare_dataset.py windows (2500bp, 8192bp, ...) are shorter, so each window is
centred inside the model input and the flanks are padded with 'N'.

  CAVEAT: those N flanks are real model input, not masked-out padding the way
  Evo2's pad tokens are — convolutions and attention do see them, so the window
  is embedded "in a sea of N" rather than in its true genomic context. If you
  care about AlphaGenome's long-range modelling at all, regenerate the windows
  at a native length instead:

      python prepare_dataset.py --tsv ... --window 16384

  and the script will use them unpadded. Only positions inside the original
  window are pooled either way (the N flanks are cropped out after the forward
  pass), so the output dimensionality is unaffected.


Model / weights
  --model-version  Checkpoint version, e.g. all_folds (default).
  --source         'huggingface' (default) or 'kaggle' — where to fetch
                    weights. Ignored if --checkpoint is given. Both need you
                    to have accepted the non-commercial terms and to be logged
                    in (huggingface_hub.login() / kagglehub.login()).
  --checkpoint     Local path to an already-downloaded Orbax checkpoint
                    directory; skips the HF/Kaggle download.
  --organism       human (default) or mouse — selects the organism embedding
                    added inside the trunk and both OutputEmbedders.
  --device         JAX device: 'gpu' (default), 'tpu' or 'cpu'.

Extraction point
  --extract-from   '128bp' (default), '1bp', 'pair', or 'internal'. See the
                    block diagram above.
  --internal-module  Module-name substring, only with --extract-from internal.

Input
  --ref-file       Reference sequences .npy. Default: output/ref_seq_DNA_{strand}.npy
  --mut-file       Mutant sequences .npy.    Default: output/mut_seq_DNA_{strand}.npy
  --strand         forward or reverse — selects the default input files and
                    labels output. Default: forward. NOTE AlphaGenome's own
                    predict_*/score_* helpers average forward and reverse
                    complement internally; forward_trunk does not, so
                    --strand reverse here means "embed the reverse-complemented
                    window", matching the other extract_embeddings_* scripts.
  --batch-size     Sequences per forward pass. Default: 1. Safe to raise —
                    every sequence is padded to the same model length, so
                    there is no cross-sequence masking issue; you are only
                    limited by GPU memory (1MB x 1bp x 1536 floats is ~6.4GB
                    per sequence).
  --model-seq-length
                    Force the model input length instead of picking the
                    smallest supported length that fits the window.

Pooling
  --pool-region    'full' (default): pool over the whole window.
                    'downstream': pool only positions from the allele onward —
                    requires --variant-meta-file. NOTE AlphaGenome is fully
                    bidirectional (conv + non-causal attention), so unlike
                    Evo2 this does NOT isolate "positions that have seen the
                    edit" — every position has seen the whole input. Here it
                    is purely a locality window around the variant, which is
                    still the useful thing: --downstream-k 0 gives the
                    embedding of the mutated base itself.
  --emb-type       'average': mean over the pooled region. 'last': last
                    position — only valid with --pool-region full.
                    Default: average.
  --downstream-k   Only used with --pool-region downstream. One or more window
                    sizes: an integer k means positions [start, start+k]
                    inclusive (k=0 = just the allele's last base), or 'all'
                    for everything to the window end. One output file per k,
                    e.g. --downstream-k 0 32 128 512 all. Default: all.
                    Values are in *bases* and are converted to the extraction
                    point's own bin size, so k=0 at --extract-from 128bp means
                    the single 128bp bin containing the allele.
  --variant-meta-file
                    variant_meta.csv written by prepare_dataset.py (columns
                    pos, ref, alt, edit_start), row-aligned with --ref-file/
                    --mut-file. Required for --pool-region downstream.
                    Default: output/variant_meta.csv.

Output
  Saved under embeddings/AlphaGenome_{model_version}_{context_window}_emb/,
  e.g. AlphaGenome_all_folds_8192bp_emb/ (context_window is parsed from the
  input filename — e.g. ref_seq_DNA_forward_8192bp.npy — so the --ref-file/
  --variant-meta-file name must contain a "<N>bp" token).
  One {ref_seq,mut_seq}_L{tag}_{emb_type}_{strand}[_ds{k}].npy per k, shape
  (N, D), where tag is 128bp / 1bp / pair / the internal module's short name.

Examples
--------
# Default: transformer-tower output (128bp bins), mean-pooled over the window
python extract_embeddings_AlphaGenome.py \
    --ref-file output/20260831_133902/ref_seq_DNA_forward_16384bp.npy \
    --mut-file output/20260831_133902/mut_seq_DNA_forward_16384bp.npy

# Base-resolution decoder output, swept around the mutated base
python extract_embeddings_AlphaGenome.py --extract-from 1bp \
    --pool-region downstream --downstream-k 0 32 128 512 all \
    --variant-meta-file output/20260831_133902/variant_meta_16384bp.csv \
    --ref-file output/20260831_133902/ref_seq_DNA_forward_16384bp.npy \
    --mut-file output/20260831_133902/mut_seq_DNA_forward_16384bp.npy

# Tap the last conv encoder block (64bp bins) instead
python extract_embeddings_AlphaGenome.py --extract-from internal \
    --internal-module downres_block_5 \
    --ref-file output/20260831_133902/ref_seq_DNA_forward_16384bp.npy \
    --mut-file output/20260831_133902/mut_seq_DNA_forward_16384bp.npy
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from embedding_utils import parse_context_window, parse_downstream_k, pool, pool_downstream

DEFAULT_MODEL_VERSION = "all_folds"
DEFAULT_EXTRACT_FROM = "128bp"
BATCH_SIZE = 1

# alphagenome.models.dna_client.SUPPORTED_SEQUENCE_LENGTHS
SUPPORTED_SEQUENCE_LENGTHS = (2**14, 2**17, 2**19, 2**20)  # 16KB, 100KB, 500KB, 1MB

# The encoder pools 7 times (1 -> 2 -> ... -> 128bp) and the pair track is at
# 128 * 16 = 2048bp, so a valid model input length must divide by 2048.
LENGTH_GRANULARITY = 2048

# Bin size in bases of each supported extraction point (see the block diagram
# in the module docstring). "internal" is inferred per-tap at runtime.
BIN_SIZE = {"1bp": 1, "128bp": 128, "pair": 2048}


def build_filename(tag: str, mode: str, strand: str, region_suffix: str = "") -> str:
    suffix = f"_{region_suffix}" if region_suffix else ""
    return f"L{tag}_{mode}_{strand}{suffix}.npy"


# ── AlphaGenome ─────────────────────────────────────────────────────────────────

def load_model(model_version: str, source: str, checkpoint: str | None, device_kind: str):
    """Returns (alphagenome_model, jax_device).

    The model object exposes the private `_trunk_apply_fn(params, state,
    dna_sequence, organism_index) -> Embeddings` built by
    dna_model.create_model(); there is no public "give me embeddings" method on
    AlphaGenomeModel, so we call that directly.
    """
    import jax
    from alphagenome_research.model import dna_model as ag_dna_model

    try:
        devices = jax.devices(device_kind)
    except RuntimeError:
        devices = []
    if not devices:
        raise RuntimeError(
            f"No JAX device of kind {device_kind!r} available (saw: "
            f"{sorted({d.platform for d in jax.devices()})}). Pass --device cpu to force CPU."
        )
    device = devices[0]
    if device.platform == "cpu":
        print("WARNING: running AlphaGenome on CPU — expect minutes per sequence.")

    if checkpoint:
        print(f"Loading AlphaGenome from local checkpoint {checkpoint} ...")
        model = ag_dna_model.create(checkpoint, device=device)
    elif source == "kaggle":
        print(f"Loading AlphaGenome {model_version} from Kaggle ...")
        model = ag_dna_model.create_from_kaggle(model_version, device=device)
    else:
        print(f"Loading AlphaGenome {model_version} from HuggingFace ...")
        model = ag_dna_model.create_from_huggingface(model_version, device=device)

    if getattr(model, "_trunk_apply_fn", None) is None:
        raise RuntimeError(
            "This AlphaGenomeModel was built without a trunk apply function, so "
            "embeddings can't be extracted from it. Expected dna_model.create() "
            "to supply one — check your alphagenome_research version."
        )
    print("Model loaded.\n")
    return model, device


def choose_model_seq_length(window_len: int, forced: int | None) -> int:
    """Smallest AlphaGenome input length that fits `window_len`."""
    if forced is not None:
        if forced % LENGTH_GRANULARITY:
            raise ValueError(
                f"--model-seq-length must be a multiple of {LENGTH_GRANULARITY} "
                f"(the encoder pools 128x and the pair track is 2048bp), got {forced}"
            )
        if forced < window_len:
            raise ValueError(f"--model-seq-length {forced} < window length {window_len}")
        if forced not in SUPPORTED_SEQUENCE_LENGTHS:
            print(
                f"WARNING: model_seq_length={forced} is shape-valid but is not one of "
                f"AlphaGenome's trained lengths {SUPPORTED_SEQUENCE_LENGTHS} — "
                "representations at untrained lengths are not calibrated."
            )
        return forced

    for length in SUPPORTED_SEQUENCE_LENGTHS:
        if length >= window_len:
            return length
    raise ValueError(
        f"Window length {window_len} exceeds AlphaGenome's maximum input of "
        f"{SUPPORTED_SEQUENCE_LENGTHS[-1]}bp (1MB)."
    )


def pad_left_for(window_len: int, model_seq_length: int) -> int:
    """Centre the window in the model input, rounding the left pad down to a
    whole 2048bp bin so every extraction point's bins line up with the window
    boundary (and the post-hoc crop is exact at 1bp, 128bp and 2048bp alike)."""
    total_pad = model_seq_length - window_len
    return (total_pad // 2 // LENGTH_GRANULARITY) * LENGTH_GRANULARITY


def encode_batch(seqs: list[str], model, model_seq_length: int, pad_left: int) -> np.ndarray:
    """One-hot encode and N-pad a batch to (B, model_seq_length, 4).

    'N' (and any other non-ACGT character) maps to an all-zero row in
    AlphaGenome's DNAOneHotEncoder, which is what the model sees for unknown
    bases — it is *not* masked out of the computation.
    """
    encoder = model._one_hot_encoder  # DNAOneHotEncoder, set up in AlphaGenomeModel.__init__
    out = np.zeros((len(seqs), model_seq_length, 4), dtype=np.float32)
    for i, seq in enumerate(seqs):
        seq = str(seq)
        out[i, pad_left:pad_left + len(seq)] = encoder.encode(seq)
    return out


def trunk_embeddings(model, dna_onehot, organism_index, extract_from: str, internal_module: str | None):
    """Runs AlphaGenome's trunk and returns (array, bin_size).

    For 1bp/128bp/pair the array is the corresponding field of the Embeddings
    dataclass returned by AlphaGenome.forward_trunk. For 'internal' it is the
    captured output of the requested Haiku module.
    """
    import jax

    if extract_from != "internal":
        embeddings = model._jit_trunk(model._params, model._state, dna_onehot, organism_index)
        array = {
            "1bp": embeddings.embeddings_1bp,
            "128bp": embeddings.embeddings_128bp,
            "pair": embeddings.embeddings_pair,
        }[extract_from]
        # forward_trunk runs under a bfloat16 compute policy, so upcast on the way out.
        return np.asarray(jax.device_get(array), dtype=np.float32), BIN_SIZE[extract_from]

    # ── internal tap ────────────────────────────────────────────────────────
    # Haiku interceptors see real arrays only outside jit (inside a traced
    # function they would capture tracers), so this path runs eagerly.
    import haiku as hk

    captured: dict[str, object] = {}

    def interceptor(next_fn, args, kwargs, context):
        out = next_fn(*args, **kwargs)
        if context.method_name == "__call__":
            captured[context.module.module_name] = out
        return out

    with hk.experimental.intercept_methods(interceptor):
        model._trunk_apply_fn(model._params, model._state, dna_onehot, organism_index)

    matches = [name for name in captured if internal_module in name]
    if not matches:
        raise ValueError(
            f"--internal-module {internal_module!r} matched no Haiku module. "
            "Modules seen during the trunk forward pass:\n  "
            + "\n  ".join(sorted(captured))
        )
    if len(matches) > 1:
        print(
            f"NOTE: {internal_module!r} matched {len(matches)} modules; using the "
            f"first in Haiku call order: {matches[0]}"
        )
    array = np.asarray(jax.device_get(captured[matches[0]]), dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(
            f"Module {matches[0]} produced a {array.ndim}-D output {array.shape}; "
            "internal taps must be (B, S, D). Use --extract-from pair for the "
            "2-D pairwise track."
        )
    bin_size = dna_onehot.shape[1] // array.shape[1]
    return array, bin_size


def crop_to_window(array: np.ndarray, pad_left: int, window_len: int, bin_size: int) -> np.ndarray:
    """Drop the N-padded flanks so pooling happens in the original window's own
    coordinates (identical contract to the Evo2/JanusDNA scripts)."""
    start = pad_left // bin_size
    end = start + -(-window_len // bin_size)  # ceil division
    return array[:, start:end]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract AlphaGenome embeddings from DNA sequence windows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION,
                        help=f"AlphaGenome checkpoint version (default: {DEFAULT_MODEL_VERSION})")
    parser.add_argument("--source", default="huggingface", choices=["huggingface", "kaggle"],
                        help="Where to fetch weights from (default: huggingface). "
                             "Ignored if --checkpoint is given.")
    parser.add_argument("--checkpoint", default=None, metavar="DIR",
                        help="Local Orbax checkpoint directory; skips the HF/Kaggle download.")
    parser.add_argument("--organism", default="human", choices=["human", "mouse"],
                        help="Organism embedding added inside the trunk (default: human)")
    parser.add_argument("--device", default="gpu", choices=["gpu", "tpu", "cpu"],
                        help="JAX device kind (default: gpu)")
    parser.add_argument("--extract-from", default=DEFAULT_EXTRACT_FROM,
                        choices=["128bp", "1bp", "pair", "internal"],
                        help="Which AlphaGenome block to read embeddings from — see the "
                             f"block diagram at the top of this file (default: {DEFAULT_EXTRACT_FROM}, "
                             "the transformer tower output)")
    parser.add_argument("--internal-module", default=None, metavar="SUBSTR",
                        help="Haiku module-name substring to tap, e.g. downres_block_5 or "
                             "dna_embedder. Only with --extract-from internal.")
    parser.add_argument("--strand", default="forward", choices=["forward", "reverse"],
                        help="Strand direction — selects input file and labels output (default: forward)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Sequences per forward pass (default: {BATCH_SIZE})")
    parser.add_argument("--model-seq-length", type=int, default=None, metavar="N",
                        help="Force the model input length instead of the smallest supported "
                             f"length that fits the window. Supported: {SUPPORTED_SEQUENCE_LENGTHS}")
    parser.add_argument("--emb-type", default="average", choices=["average", "last"],
                        help="Pooling: 'average' = mean over the pooled region; 'last' = last "
                             "position — only valid with --pool-region full. (default: average)")
    parser.add_argument("--ref-file", default=None, metavar="FILE",
                        help="Path to reference sequences .npy file. "
                             "Default: output/ref_seq_DNA_{strand}.npy")
    parser.add_argument("--mut-file", default=None, metavar="FILE",
                        help="Path to mutant sequences .npy file. "
                             "Default: output/mut_seq_DNA_{strand}.npy")
    parser.add_argument("--pool-region", default="full", choices=["full", "downstream"],
                        help="'full' (default) pools over the whole window. 'downstream' pools "
                             "from the allele onward — a locality window, not a causal one, "
                             "since AlphaGenome is bidirectional.")
    parser.add_argument("--downstream-k", nargs="+", default=["all"], type=parse_downstream_k, metavar="K",
                        help="Window sizes in bases when --pool-region downstream: an integer k "
                             "(positions [start, start+k], inclusive; k=0 = just the allele's "
                             "last base) or the literal 'all'. One output file per k. Default: all")
    parser.add_argument("--variant-meta-file", default=None, metavar="FILE",
                        help="Path to variant_meta.csv written by prepare_dataset.py (columns "
                             "pos, ref, alt, edit_start; row-aligned with --ref-file/--mut-file). "
                             "Default: output/variant_meta.csv. Required for --pool-region downstream.")
    args = parser.parse_args()

    if args.pool_region == "downstream" and args.emb_type != "average":
        parser.error("--pool-region downstream only supports --emb-type average "
                     "(use --downstream-k 0 for just the mutation-site embedding)")
    if args.extract_from == "internal" and not args.internal_module:
        parser.error("--extract-from internal requires --internal-module (e.g. downres_block_5)")
    if args.extract_from != "internal" and args.internal_module:
        parser.error("--internal-module is only meaningful with --extract-from internal")
    if args.extract_from == "pair" and args.pool_region == "downstream":
        parser.error(
            "--extract-from pair is a 2-D (bin x bin) map, so the 1-D downstream sweep "
            "doesn't apply — use --pool-region full, which means over both bin axes"
        )
    return args


def extract_embeddings(
    sequences,
    model,
    df: str,
    emb_type: str,
    strand: str,
    out_dir: str,
    tag: str,
    extract_from: str,
    internal_module: str | None,
    organism_index,
    model_seq_length: int,
    window_len: int,
    pad_left: int,
    batch_size: int = BATCH_SIZE,
    pool_region: str = "full",
    downstream_ks: list | None = None,
    edit_starts: list | None = None,
) -> None:
    """One forward pass per batch; saves one .npy per k (or a single file for
    pool_region='full') into out_dir."""
    import jax

    os.makedirs(out_dir, exist_ok=True)
    regions = downstream_ks if pool_region == "downstream" else [None]  # None = full-window pool()
    all_pooled = {r: [] for r in regions}

    for start in range(0, len(sequences), batch_size):
        seqs = sequences[start:start + batch_size]
        onehot = encode_batch(list(seqs), model, model_seq_length, pad_left)
        onehot = jax.device_put(onehot, model._emb_device)
        org_idx = jax.device_put(
            np.full((len(seqs),), organism_index, dtype=np.int32), model._emb_device
        )

        array, bin_size = trunk_embeddings(model, onehot, org_idx, extract_from, internal_module)

        if extract_from == "pair":
            # (B, bins, bins, 128) at 2048bp — crop both bin axes to the window,
            # then mean over both: a single interaction-profile vector per row.
            cropped = crop_to_window(array, pad_left, window_len, bin_size)
            cropped = crop_to_window(cropped.transpose(0, 2, 1, 3), pad_left, window_len, bin_size)
            all_pooled[None].append(cropped.mean(axis=(1, 2)).astype(np.float32))
        else:
            cropped = crop_to_window(array, pad_left, window_len, bin_size)
            hidden = torch.from_numpy(np.ascontiguousarray(cropped))
            # Every row is the same length here (fixed-width windows), but pool()
            # takes explicit lengths so the contract matches the other scripts.
            lengths = [hidden.shape[1]] * hidden.shape[0]
            batch_starts = None
            if pool_region == "downstream":
                # edit_starts are base offsets into the window; convert to this
                # extraction point's bins.
                batch_starts = [s // bin_size for s in edit_starts[start:start + batch_size]]
            for r in regions:
                if pool_region == "downstream":
                    k = r if r == "all" else max(r // bin_size, 0)
                    pooled = pool_downstream(hidden, lengths, batch_starts, k)
                else:
                    pooled = pool(hidden, lengths, emb_type)
                all_pooled[r].append(pooled.float().numpy())

        print(f"Processed {min(start + batch_size, len(sequences))}/{len(sequences)} sequences")

    for r in regions:
        combined = np.concatenate(all_pooled[r], axis=0)
        region_suffix = "" if r is None else f"ds{r}"
        fname = build_filename(tag, emb_type, strand, region_suffix)
        out_path = os.path.join(out_dir, f"{df}_{fname}")
        np.save(out_path, combined)
        print(f"Saved {emb_type} embeddings from {tag} region={region_suffix or 'full'}: "
              f"{combined.shape} -> {out_path}")


if __name__ == "__main__":
    args = parse_args()

    try:
        import jax
        from alphagenome.models import dna_model as ag_public_dna_model
        from alphagenome_research.model import dna_model as ag_dna_model
    except ImportError as exc:
        raise SystemExit(
            f"{exc}\n\nAlphaGenome embeddings need the research release (the hosted API "
            "does not return hidden states):\n"
            "    pip install git+https://github.com/google-deepmind/alphagenome_research.git\n"
            "plus a JAX build for your accelerator (e.g. pip install -U 'jax[cuda12]')."
        ) from exc

    model, device = load_model(args.model_version, args.source, args.checkpoint, args.device)
    # Stash the device and a jitted trunk on the model so extract_embeddings()
    # doesn't have to thread them through every call.
    model._emb_device = device
    model._jit_trunk = jax.jit(model._trunk_apply_fn)

    organism = (ag_public_dna_model.Organism.HOMO_SAPIENS if args.organism == "human"
                else ag_public_dna_model.Organism.MUS_MUSCULUS)
    organism_index = ag_dna_model.convert_to_organism_index(organism)

    input_files = {
        "ref_seq": args.ref_file or f"output/ref_seq_DNA_{args.strand}.npy",
        "mut_seq": args.mut_file or f"output/mut_seq_DNA_{args.strand}.npy",
    }

    context_window = parse_context_window(input_files["ref_seq"])
    out_dir = os.path.join("embeddings", f"AlphaGenome_{args.model_version}_{context_window}_emb")

    tag = args.internal_module if args.extract_from == "internal" else args.extract_from

    edit_start = ref_len = alt_len = None
    if args.pool_region == "downstream":
        meta = pd.read_csv(args.variant_meta_file or "output/variant_meta.csv")
        edit_start = meta["edit_start"].to_numpy()
        ref_len = meta["ref"].astype(str).str.len().to_numpy()
        alt_len = meta["alt"].astype(str).str.len().to_numpy()

    for df, path in input_files.items():
        seqs = np.load(path)
        window_len = max(len(str(s)) for s in seqs)
        if min(len(str(s)) for s in seqs) != window_len:
            raise ValueError(
                f"{path} contains sequences of differing lengths; AlphaGenome needs a "
                "fixed input length, so regenerate the windows with a single --window."
            )
        model_seq_length = choose_model_seq_length(window_len, args.model_seq_length)
        pad_left = pad_left_for(window_len, model_seq_length)
        if model_seq_length != window_len:
            print(f"{df}: padding {window_len}bp windows to AlphaGenome's {model_seq_length}bp "
                  f"input with N ({pad_left}bp left / {model_seq_length - window_len - pad_left}bp "
                  "right); flanks are cropped out again before pooling.")

        starts = None
        if args.pool_region == "downstream":
            # start = the allele's own last base, in window coordinates — same
            # convention as the Evo2/JanusDNA scripts so the k sweeps line up.
            if args.strand == "forward":
                allele_len = ref_len if df == "ref_seq" else alt_len
                starts = (edit_start + allele_len - 1).tolist()
            else:
                # Reverse-complementing flips reading direction, so the allele's
                # last base in the reverse array's own left-to-right order is the
                # locus at forward position edit_start.
                starts = (window_len - edit_start - 1).tolist()

        extract_embeddings(
            sequences=seqs,
            model=model,
            df=df,
            emb_type=args.emb_type,
            strand=args.strand,
            out_dir=out_dir,
            tag=tag,
            extract_from=args.extract_from,
            internal_module=args.internal_module,
            organism_index=organism_index,
            model_seq_length=model_seq_length,
            window_len=window_len,
            pad_left=pad_left,
            batch_size=args.batch_size,
            pool_region=args.pool_region,
            downstream_ks=args.downstream_k,
            edit_starts=starts,
        )
