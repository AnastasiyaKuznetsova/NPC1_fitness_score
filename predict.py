"""
Load a saved model and run predictions on new embeddings.

The model was trained on delta (mut - ref) or concat (mut || ref) embeddings.
This script replicates the same data loading so predictions match training.

Usage examples:

  # With ground-truth labels (computes Spearman / MSE / MAE):
  python predict.py \
      --model saved_models/SVR__delta__forward.joblib \
      --emb emb_to_test/ \
      --df output/df_preprocessed.csv \
      --strand forward --delta --layer 27 --pooling average \
      --out predictions.csv

  # Without labels (inference only):
  python predict.py \
      --model saved_models/SVR__delta__forward.joblib \
      --emb emb_7B/emb_7B \
      --strand forward --delta --layer 27 --pooling average \
      --out predictions.csv
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from regressor import _load_ref_mut, _pool, _safe_corr
from sklearn.metrics import mean_squared_error, mean_absolute_error


def main():
    parser = argparse.ArgumentParser(description="Predict fitness scores using a saved model.")

    parser.add_argument("--model", required=True,
                        help="Path to .joblib model saved by regressor.py.")
    parser.add_argument("--emb", required=True,
                        help="Directory containing ref_seq_*.npy and mut_seq_*.npy files.")
    parser.add_argument("--df", default=None,
                        help="CSV/TSV with ground-truth 'Function Score' column. "
                             "If omitted, only predictions are written (no metrics).")
    parser.add_argument("--strand", choices=["forward", "reverse", "both"], default="forward")
    parser.add_argument("--delta", action="store_true",
                        help="Use delta (mut - ref) embeddings. Must match training.")
    parser.add_argument("--layer", default=None,
                        help="Layer index (e.g. 27). Required for multi-layer directories.")
    parser.add_argument("--pooling", choices=["average", "last"], default=None,
                        help="Pooling mode. Required for multi-layer directories.")
    parser.add_argument("--out", default="predictions.csv",
                        help="Output CSV path. Default: predictions.csv")

    args = parser.parse_args()

    # ── Load model ──────────────────────────────────────────────────────────────
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    pipeline = joblib.load(model_path)
    print(f"Loaded model: {model_path}")

    # ── Load embeddings ─────────────────────────────────────────────────────────
    emb_dir = Path(args.emb)
    if args.strand == "both":
        ref_fwd, mut_fwd = _load_ref_mut(emb_dir, "forward", layer=args.layer, pooling=args.pooling)
        emb_fwd = _pool(ref_fwd, mut_fwd, args.delta)
        ref_rev, mut_rev = _load_ref_mut(emb_dir, "reverse", layer=args.layer, pooling=args.pooling)
        emb_rev = _pool(ref_rev, mut_rev, args.delta)
        X = np.concatenate([emb_fwd, emb_rev], axis=1)
    else:
        ref, mut = _load_ref_mut(emb_dir, args.strand, layer=args.layer, pooling=args.pooling)
        X = _pool(ref, mut, args.delta)

    print(f"Embedding shape: {X.shape}")

    # ── Predict ─────────────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X)

    # ── Optionally evaluate ─────────────────────────────────────────────────────
    results = pd.DataFrame({"y_pred": y_pred})

    if args.df is not None:
        sep = "\t" if args.df.endswith(".tsv") else ","
        df = pd.read_csv(args.df, sep=sep)

        if "Function Score" not in df.columns:
            print("WARNING: 'Function Score' column not found in df — skipping metrics.",
                  file=sys.stderr)
        else:
            y_true = df["Function Score"].to_numpy()
            if len(y_true) != len(y_pred):
                print(f"ERROR: df has {len(y_true)} rows but embeddings have {len(y_pred)} samples.",
                      file=sys.stderr)
                sys.exit(1)

            spearman_r = _safe_corr(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)

            print(f"\nMetrics vs ground truth:")
            print(f"  Spearman r : {spearman_r:.4f}")
            print(f"  MSE        : {mse:.4f}")
            print(f"  MAE        : {mae:.4f}")

            results["y_true"] = y_true

            # Add any identifying columns from df if present
            for col in ("sequence", "Protein Annotation", "variant", "id"):
                if col in df.columns:
                    results.insert(0, col, df[col].values)

    # ── Save ────────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\nPredictions saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
