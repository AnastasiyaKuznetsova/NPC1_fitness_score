"""
Score NPC1 variants zero-shot with JanusDNA's own bidirectional log-likelihood,
analogous to evo2_zero_shot.py's use of Evo2's score_sequences.

Computes delta = log-likelihood(mut) - log-likelihood(ref) for each variant and
saves per-variant scores to --out. No training involved. --ref-file/--mut-file
must be row-aligned with each other (as produced together by prepare_dataset.py,
before any --jitter augmentation).

Model / checkpoint
  Same requirements as extract_embeddings_JanusDNA.py: --janusdna-repo (a local
  clone of github.com/Qihao-Duan/JanusDNA) and --checkpoint (a local weights
  file — see that script's docstring for exactly which files are available on
  JanusDNA's Harvard Dataverse release). No --model-size flag: the
  architecture is always read from the sibling "<checkpoint-stem>_model_
  config.json" Dataverse ships next to every .ckpt — the script errors if
  that file is missing. Pretraining context length isn't part of that config
  (it's training metadata, not an architecture field), so it's looked up by
  reverse-mapping the loaded config back to one of 5 known configs
  (janusdna_utils.MODEL_PRESETS); windows much longer than that are
  out-of-distribution for the checkpoint — the script warns if your data
  exceeds it, or skips the check if the config doesn't match a known preset.

Input
  --ref-file       Reference sequence windows (.npy). Default: output/ref_seq_DNA_forward.npy
  --mut-file       Mutant sequence windows (.npy).    Default: output/mut_seq_DNA_forward.npy
  --batch-size     Sequences per forward pass. Default: 1 — JanusDNA's Mamba
                    layers don't take an attention mask, so padding a batch of
                    different-length sequences would leak into the state-space
                    scan of the shorter ones. Only raise this if every sequence
                    in your input file has the same length.

Example
-------
python janusdna_zero_shot.py \\
    --janusdna-repo /path/to/JanusDNA \\
    --checkpoint /path/to/72_without_midattn.ckpt \\
    --ref-file output/ref_seq_DNA_forward.npy \\
    --mut-file output/mut_seq_DNA_forward.npy \\
    --out output/janusdna_zero_shot_scores.csv
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd
import torch

from janusdna_utils import (
    MODEL_PRESETS, PAD_ID,
    build_config, find_sibling_config_json, infer_preset_key, infer_vocab_size, load_checkpoint,
    patch_janusdna_causal_mask_bug, tokenize,
)

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def score_sequences(model, sequences: list, batch_size: int) -> list[float]:
    """Mean per-token log-likelihood under JanusDNA's bidirectional objective,
    excluding each sequence's first and last position (see module docstring)."""
    scores = []
    for start in range(0, len(sequences), batch_size):
        seqs = [str(s) for s in sequences[start:start + batch_size]]
        token_ids = [tokenize(s) for s in seqs]
        lengths = [len(t) for t in token_ids]
        max_length = max(lengths)
        if len(set(lengths)) > 1:
            warnings.warn(
                "Batch has sequences of different lengths — padding is not masked out in JanusDNA's "
                "Mamba layers, so shorter sequences' scores may be affected by padding. "
                "Use --batch-size 1 (default) if this matters."
            )
        padded = [t + [PAD_ID] * (max_length - len(t)) for t in token_ids]
        input_ids = torch.tensor(padded, dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            logits = model(input_ids=input_ids, return_dict=True).logits
        token_logp = torch.log_softmax(logits, dim=-1).gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

        for b, length in enumerate(lengths):
            valid = token_logp[b, :length]
            scores.append((valid[1:-1].mean() if length > 2 else valid.mean()).item())

        print(f"Scored {min(start + batch_size, len(sequences))}/{len(sequences)} sequences")
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Score NPC1 variants zero-shot with JanusDNA's bidirectional log-likelihood.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--janusdna-repo", required=True,
                        help="Path to a local clone of github.com/Qihao-Duan/JanusDNA (for its `janusdna` package).")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a local pretrained-weights file — requires a sibling "
                             "'<checkpoint-stem>_model_config.json' in the same directory (see module "
                             "docstring). No --model-size flag: architecture is read from that file.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Sequences per forward pass (default: 1 — see module docstring on padding).")
    parser.add_argument("--ref-file", default="output/ref_seq_DNA_forward.npy",
                        help="Reference sequence windows (.npy). Default: output/ref_seq_DNA_forward.npy")
    parser.add_argument("--mut-file", default="output/mut_seq_DNA_forward.npy",
                        help="Mutant sequence windows (.npy). Default: output/mut_seq_DNA_forward.npy")
    parser.add_argument("--out", default="output/janusdna_zero_shot_scores.csv",
                        help="Output CSV path. Default: output/janusdna_zero_shot_scores.csv")
    args = parser.parse_args()

    ref_seqs = np.load(args.ref_file)
    var_seqs = np.load(args.mut_file)
    if len(ref_seqs) != len(var_seqs):
        sys.exit(f"ERROR: {len(ref_seqs)} reference vs {len(var_seqs)} mutant sequences — must match")

    sys.path.insert(0, args.janusdna_repo)
    import janusdna.modeling_janusdna as modeling_janusdna
    from janusdna.configuration_janusdna import JanusDNAConfig
    from janusdna.modeling_janusdna import JanusDNAForCausalLM
    patch_janusdna_causal_mask_bug(modeling_janusdna)

    print(f"Using device: {DEVICE}")
    vocab_size = infer_vocab_size(args.checkpoint)
    config_json = find_sibling_config_json(args.checkpoint)
    if config_json is None:
        sys.exit(
            f"ERROR: no sibling '<checkpoint-stem>_model_config.json' found next to {args.checkpoint!r} "
            "— place the config JSON Dataverse ships with this checkpoint in the same directory."
        )
    config = build_config(vocab_size, JanusDNAConfig, config_json)
    preset_key = infer_preset_key(config)
    label = preset_key or f"{config.hidden_size}dim"
    print(f"Building JanusDNAForCausalLM ({label}, vocab_size={vocab_size}) ...")
    model = JanusDNAForCausalLM(config).to(DEVICE).eval()
    load_checkpoint(model, args.checkpoint)
    print("Model loaded.\n")

    if preset_key is None:
        print(f"NOTE: {label} doesn't match a known preset in MODEL_PRESETS, so its pretraining "
              "context length is unknown — skipping the out-of-distribution window-length check.")
    else:
        pretrain_context_length = MODEL_PRESETS[preset_key]["context_length"]
        max_window = max(len(s) for s in ref_seqs)
        if max_window > pretrain_context_length:
            warnings.warn(
                f"Input windows are up to {max_window} bp, but {label} was pretrained at only "
                f"{pretrain_context_length} bp context — scores from windows this much longer than "
                f"pretraining are out-of-distribution for this checkpoint. Use 144dim_nomidattn "
                f"(pretrained at 131,072 bp) for long windows."
            )

    print(f"Scoring likelihoods of {len(ref_seqs)} reference sequences with JanusDNA ({label})...")
    ref_scores = score_sequences(model, ref_seqs, args.batch_size)

    print(f"Scoring likelihoods of {len(var_seqs)} variant sequences with JanusDNA ({label})...")
    var_scores = score_sequences(model, var_seqs, args.batch_size)

    delta_scores = np.array(var_scores) - np.array(ref_scores)

    out_df = pd.DataFrame({
        "ref_score": ref_scores,
        "var_score": var_scores,
        "janusdna_delta_score": delta_scores,
    })
    out_df.to_csv(args.out, index=False)
    print(f"\nSaved {len(out_df)} scores -> {args.out}")


if __name__ == "__main__":
    main()
