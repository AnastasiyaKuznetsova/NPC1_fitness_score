"""
Score NPC1 variants zero-shot with DNABERT-2's pseudo-log-likelihood (PLL).

DNABERT-2 is a masked language model, not autoregressive like Evo2, so its
zero-shot score is the sequence pseudo-log-likelihood: each non-special
token is masked one at a time and its log-probability under the model is
summed across all positions (Salazar et al., 2020 "Masked LM Scoring").
delta = PLL(mut) - PLL(ref), matching evo2_zero_shot.py's sign convention
(var - ref) and output shape, so both plug into benchmark_zero_shot.py the
same way.

--ref-file/--mut-file must be row-aligned with each other and with --df (as
produced together by prepare_dataset.py, before any --jitter augmentation).

Note: PLL needs one forward pass per token position, so this is much slower
per-sequence than Evo2's single-pass scoring, even though DNABERT-2 itself
is a much smaller model. Runs on CPU or GPU (auto-detected).

Example
-------
python dnabert2_zero_shot.py \\
    --ref-file output/20260831_133902/ref_seq_DNA_forward_2500bp.npy \\
    --mut-file output/20260831_133902/mut_seq_DNA_forward_2500bp.npy \\
    --out output/dnabert2_zero_shot_scores.csv
"""

import argparse
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_NAME = "zhihan1996/DNABERT-2-117M"


def pseudo_log_likelihood(seq: str, tokenizer, model, device, batch_size: int) -> float:
    input_ids = tokenizer(seq, return_tensors="pt")["input_ids"][0]
    special_mask = torch.tensor(
        tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True),
        dtype=torch.bool,
    )
    positions = torch.nonzero(~special_mask, as_tuple=True)[0]
    if len(positions) == 0:
        return 0.0

    total_log_prob = 0.0
    for start in range(0, len(positions), batch_size):
        batch_positions = positions[start:start + batch_size]
        batch_input = input_ids.unsqueeze(0).repeat(len(batch_positions), 1).clone()
        for i, pos in enumerate(batch_positions):
            batch_input[i, pos] = tokenizer.mask_token_id
        batch_input = batch_input.to(device)

        with torch.no_grad():
            outputs = model(batch_input)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        log_probs = torch.log_softmax(logits, dim=-1)

        for i, pos in enumerate(batch_positions):
            true_id = input_ids[pos]
            total_log_prob += log_probs[i, pos, true_id].item()

    return total_log_prob


def main():
    parser = argparse.ArgumentParser(
        description="Score NPC1 variants zero-shot with DNABERT-2 pseudo-log-likelihood.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ref-file", default="output/ref_seq_DNA_forward_20260825.npy",
                        help="Reference sequence windows (.npy). Default: output/ref_seq_DNA_forward_20260825.npy")
    parser.add_argument("--mut-file", default="output/mut_seq_DNA_forward_20260825.npy",
                        help="Mutant sequence windows (.npy). Default: output/mut_seq_DNA_forward_20260825.npy")
    parser.add_argument("--model", default=MODEL_NAME,
                        help=f"HF model name. Default: {MODEL_NAME}")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Masked positions scored per forward pass. Default: 16")
    parser.add_argument("--out", default="output/dnabert2_zero_shot_scores.csv",
                        help="Output CSV path. Default: output/dnabert2_zero_shot_scores.csv")

    args = parser.parse_args()

    ref_seqs = np.load(args.ref_file, allow_pickle=True)
    var_seqs = np.load(args.mut_file, allow_pickle=True)
    if len(ref_seqs) != len(var_seqs):
        sys.exit(f"ERROR: {len(ref_seqs)} reference vs {len(var_seqs)} mutant sequences — must match")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model, trust_remote_code=True).to(device).eval()

    print(f"Scoring {len(ref_seqs)} reference sequences with DNABERT-2 PLL...")
    ref_scores = []
    for i, seq in enumerate(ref_seqs):
        ref_scores.append(pseudo_log_likelihood(str(seq), tokenizer, model, device, args.batch_size))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(ref_seqs)}")

    print(f"Scoring {len(var_seqs)} variant sequences with DNABERT-2 PLL...")
    var_scores = []
    for i, seq in enumerate(var_seqs):
        var_scores.append(pseudo_log_likelihood(str(seq), tokenizer, model, device, args.batch_size))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(var_seqs)}")

    delta_scores = np.array(var_scores) - np.array(ref_scores)

    out_df = pd.DataFrame({
        "ref_score": ref_scores,
        "var_score": var_scores,
        "dnabert2_delta_score": delta_scores,
    })
    out_df.to_csv(args.out, index=False)
    print(f"\nSaved {len(out_df)} scores -> {args.out}")


if __name__ == "__main__":
    main()
