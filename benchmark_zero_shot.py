"""
Benchmark zero-shot variant-effect predictor scores against the NPC1
fitness-score dataset (Function Score in df_preprocessed.csv), reporting
Spearman correlation. Reads precomputed score files from output/ for each
predictor — it does not run any model itself — and saves one CSV with a
row per model.

Models (--model, default: all three):
  evo2      Reads evo2_delta_score from output/evo2_zero_shot_scores.csv
            (see evo2_zero_shot.py). Row-aligned with --df.
  cadd      Reads CADD_RawScore from output/cadd_scores.csv (see
            cadd_score.py). Matched to --df by (pos, ref, alt).
  dnabert2  Reads dnabert2_delta_score from
            output/dnabert2_zero_shot_scores.csv (see dnabert2_zero_shot.py).
            Row-aligned with --df.

Models whose score file is missing or empty are skipped (with a message)
rather than failing the whole comparison.

Examples
--------
python benchmark_zero_shot.py                        # all three models
python benchmark_zero_shot.py --model evo2 dnabert2   # subset
"""

import argparse

import pandas as pd
from scipy.stats import spearmanr


def _row_aligned_scores(df: pd.DataFrame, path: str, score_col: str, label: str):
    scores = pd.read_csv(path)
    if len(scores) != len(df):
        raise ValueError(
            f"{len(scores)} rows in {path} vs {len(df)} rows in --df — "
            f"{label} scores must be row-aligned (same order used when the scoring script was run)"
        )
    return df["Function Score"], scores[score_col]


def run_evo2(args, df: pd.DataFrame):
    return _row_aligned_scores(df, args.evo2_file, "evo2_delta_score", "evo2")


def run_dnabert2(args, df: pd.DataFrame):
    return _row_aligned_scores(df, args.dnabert2_file, "dnabert2_delta_score", "dnabert2")


def run_cadd(args, df: pd.DataFrame):
    cadd = pd.read_csv(args.cadd_file)
    merged = df.merge(
        cadd,
        left_on=["end", "reference_base", "alternate_base"],
        right_on=["pos", "ref", "alt"],
        how="inner",
    )
    print(f"Matched {len(merged)}/{len(df)} variants to CADD scores in {args.cadd_file}")
    merged = merged.dropna(subset=["CADD_RawScore"])
    return merged["Function Score"], merged["CADD_RawScore"]


MODELS = {"evo2": run_evo2, "cadd": run_cadd, "dnabert2": run_dnabert2}


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark zero-shot predictors' saved scores against NPC1 Function Score.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", nargs="+", choices=MODELS, default=list(MODELS),
                        help="Which zero-shot predictor(s) to benchmark. Default: all of them.")
    parser.add_argument("--df", default="output/df_preprocessed.csv",
                        help="Preprocessed dataframe with Function Score. Default: output/df_preprocessed.csv")
    parser.add_argument("--evo2-file", default="output/evo2_zero_shot_scores.csv",
                        help="[evo2] Output of evo2_zero_shot.py. Default: output/evo2_zero_shot_scores.csv")
    parser.add_argument("--cadd-file", default="output/cadd_scores.csv",
                        help="[cadd] Output of cadd_score.py. Default: output/cadd_scores.csv")
    parser.add_argument("--dnabert2-file", default="output/dnabert2_zero_shot_scores.csv",
                        help="[dnabert2] Output of dnabert2_zero_shot.py. Default: output/dnabert2_zero_shot_scores.csv")
    parser.add_argument("--out", default="output/benchmark_zero_shot_results.csv",
                        help="Output CSV path. Default: output/benchmark_zero_shot_results.csv")

    args = parser.parse_args()
    df = pd.read_csv(args.df)

    results = []
    for name in args.model:
        try:
            function_score, pred_score = MODELS[name](args, df)
        except (FileNotFoundError, ValueError) as e:
            print(f"[{name}] SKIPPED — {e}")
            continue
        rho, pval = spearmanr(function_score, pred_score, nan_policy="omit")
        n = len(function_score)
        print(f"[{name}] Spearman rho={rho:.4f}, p={pval:.3g}, n={n}")
        results.append({"model": name, "spearman_rho": rho, "p_value": pval, "n": n})

    if not results:
        raise SystemExit("ERROR: no model produced results — check that the score files exist")

    out_df = pd.DataFrame(results).sort_values("spearman_rho", key=abs, ascending=False)
    out_df.to_csv(args.out, index=False)
    print(f"\nSaved comparison -> {args.out}")


if __name__ == "__main__":
    main()
