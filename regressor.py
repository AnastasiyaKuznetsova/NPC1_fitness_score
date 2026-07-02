"""
Group-aware train/test split by protein annotation + linear regression.

Expected DataFrame columns:
  - 'sequence'   : protein sequence string
  - 'label'      : function score

All sequences sharing a cluster_id are kept together in either train or test —
no data leakage from identical sequences across splits.

Embedding modes:
  --emb_mode rna          : single .npy file of RNA embeddings (original behaviour)
  --emb_mode dna --delta  : delta embeddings (mut - ref)
  --emb_mode dna          : concat embeddings (mut || ref)
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import lightgbm as lgb


# ── 0. Logging ────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str = "logs") -> Path:
    """
    Configure root logger to write to both stdout and a timestamped log file.
    File name: logs/run_YYYYMMDD_HHMMSS.log
    Returns the log file path.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = Path(log_dir) / f"run_{timestamp}.log"

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(sh)

    logging.info(f"Log file: {log_path.resolve()}")
    return log_path


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_data_rna(path_to_df: str, path_to_emb: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Original RNA path: single embedding matrix."""
    sep = "\t" if path_to_df.endswith(".tsv") else ","
    df  = pd.read_csv(path_to_df, sep=sep)
    emb = np.load(path_to_emb)
    return df, emb


def load_data_dna(
    path_to_df: str,
    path_ref: str,
    path_mut: str,
    delta: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    DNA path: separate reference and mutant embedding matrices.

    Returns (df, emb) where emb is either:
      delta  = mut - ref        (N, D)   when delta=True
      concat = [mut || ref]     (N, 2D)  when delta=False
    """
    sep = "\t" if path_to_df.endswith(".tsv") else ","
    df  = pd.read_csv(path_to_df, sep=sep)

    ref = np.load(path_ref)  # (N, D) or (N, 1, D) from Evo2
    mut = np.load(path_mut)  # (N, D) or (N, 1, D) from Evo2

    # Evo2 saves embeddings with an extra length dim → squeeze to (N, D)
    if ref.ndim == 3:
        ref = ref.squeeze(1)
    if mut.ndim == 3:
        mut = mut.squeeze(1)

    assert ref.shape == mut.shape, (
        f"Reference and mutant embedding shapes must match: "
        f"{ref.shape} vs {mut.shape}"
    )

    logging.info(f"Loaded {ref.shape[0]} variants | embedding dim {ref.shape[1]}")

    if delta:
        emb = mut - ref
        logging.info(f"  delta  shape : {emb.shape}")
    else:
        emb = np.concatenate([mut, ref], axis=1)
        logging.info(f"  concat shape : {emb.shape}")

    return df, emb


# ── 2. Group-aware split ──────────────────────────────────────────────────────

def group_train_test_split(
    emb: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Single hold-out split that keeps all members of a cluster_id
    entirely in train or entirely in test.

    Returns: X_train, X_test, y_train, y_test, train_idx, test_idx
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                            random_state=random_state)
    y = np.asarray(y)
    train_idx, test_idx = next(gss.split(emb, y, groups=groups))

    train_clusters = set(groups[train_idx])
    test_clusters  = set(groups[test_idx])
    overlap = train_clusters & test_clusters
    assert len(overlap) == 0, f"Cluster leakage detected: {overlap}"

    logging.info(f"Train: {len(train_idx)} samples | {len(train_clusters)} clusters")
    logging.info(f"Test : {len(test_idx)}  samples | {len(test_clusters)}  clusters")

    return (emb[train_idx], emb[test_idx],
            y[train_idx],   y[test_idx],
            train_idx,      test_idx)


def group_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    linear: bool = False,
) -> list[dict]:
    """
    Group K-Fold cross-validation — no cluster appears in both
    the fold's train and validation sets.

    Returns list of per-fold metric dicts.
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []

    for fold, (tr, val) in enumerate(gkf.split(X, y, groups=groups)):
        pipe = build_pipeline(linear=linear)
        pipe.fit(X[tr], y[tr])
        y_pred = pipe.predict(X[val])
        metrics = {
            "fold": fold + 1,
            "r2":   r2_score(y[val], y_pred),
            "rmse": np.sqrt(mean_squared_error(y[val], y_pred)),
            "mae":  mean_absolute_error(y[val], y_pred),
        }
        fold_metrics.append(metrics)
        logging.info(f"Fold {fold+1}: R²={metrics['r2']:.3f}  "
                     f"RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}")

    logging.info(f"Mean CV R²  : {np.mean([m['r2']   for m in fold_metrics]):.3f}")
    logging.info(f"Mean CV RMSE: {np.mean([m['rmse'] for m in fold_metrics]):.4f}")
    return fold_metrics


# ── 3. Model ──────────────────────────────────────────────────────────────────

def build_pipeline(linear: bool = False, use_ridge: bool = True) -> Pipeline:
    """
    StandardScaler → Ridge / LinearRegression / LGBMRegressor pipeline.
    Ridge is recommended when features are collinear or n_features > n_samples.
    """
    if not linear:
        model = lgb.LGBMRegressor(verbose=-1)   # silence LightGBM split warnings
        logging.info("Using LGBM Regressor")
    else:
        model = Ridge() if use_ridge else LinearRegression()
        logging.info("Using linear model")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    model),
    ])


# ── 4. Evaluate ───────────────────────────────────────────────────────────────

def evaluate(
    pipe: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tag: str = "",
) -> dict:
    y_pred = pipe.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0, 1]

    label = f" [{tag}]" if tag else ""
    logging.info(f"── Hold-out test metrics{label} " + "─" * max(0, 40 - len(label)))
    logging.info(f"R²       : {r2:.4f}")
    logging.info(f"RMSE     : {rmse:.4f}")
    logging.info(f"MAE      : {mae:.4f}")
    logging.info(f"Pearson r: {corr:.4f}")

    return {"tag": tag, "r2": r2, "rmse": rmse, "mae": mae, "pearson_r": corr}


# ── 5. Run one embedding matrix end-to-end ────────────────────────────────────

def run_single(
    emb: np.ndarray,
    df: pd.DataFrame,
    linear: bool,
    tag: str = "",
) -> dict:
    """Full pipeline for one embedding matrix. Returns hold-out metrics dict."""
    logging.info("=" * 60)
    logging.info(f"  {tag}")
    logging.info("=" * 60)

    groups = df["Protein Annotation"].to_numpy()
    y      = df["Function Score"].to_numpy()

    logging.info("=== Hold-out split ===")
    X_train, X_test, y_train, y_test, tr_idx, _ = group_train_test_split(
        emb, y, groups, test_size=0.2
    )

    groups_train = groups[tr_idx]

    logging.info("=== Group K-Fold CV ===")
    group_kfold_cv(X_train, y_train, groups_train, n_splits=3, linear=linear)

    pipe = build_pipeline(linear=linear)
    pipe.fit(X_train, y_train)
    metrics = evaluate(pipe, X_test, y_test, tag=tag)
    return metrics


# ── 6. Main ───────────────────────────────────────────────────────────────────

def main(args):
    log_path = setup_logging(log_dir=args.log_dir)

    # Log the full invocation so the file is self-contained
    logging.info("Run arguments: " + " ".join(sys.argv))

    results = []

    if args.emb_mode == "rna":
        if not args.emb:
            raise ValueError("--emb is required for --emb_mode rna")

        df, emb = load_data_rna(args.df, args.emb)
        m = run_single(emb, df, linear=args.linear, tag="RNA embeddings")
        results.append(m)

    elif args.emb_mode == "dna":
        if not args.emb_ref or not args.emb_mut:
            raise ValueError(
                "--emb_ref and --emb_mut are required for --emb_mode dna"
            )

        tag = "DNA | delta (mut - ref)" if args.delta else "DNA | concat (mut || ref)"
        df, emb = load_data_dna(args.df, args.emb_ref, args.emb_mut, args.delta)
        m = run_single(emb, df, linear=args.linear, tag=tag)
        results.append(m)

    # ── Summary table ──────────────────────────────────────────────────────
    logging.info("\n" + "=" * 60)
    logging.info("  SUMMARY")
    logging.info("=" * 60)
    logging.info(f"{'Tag':<35} {'R²':>6} {'RMSE':>7} {'Pearson r':>10}")
    logging.info("-" * 60)
    for m in results:
        logging.info(f"{m['tag']:<35} {m['r2']:>6.4f} {m['rmse']:>7.4f} {m['pearson_r']:>10.4f}")

    logging.info(f"\nFull log saved to: {log_path.resolve()}")


# ── 7. CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train regressor on RNA or DNA (delta/concat) embeddings."
    )

    parser.add_argument("--df", required=True,
                        help="Path to preprocessed CSV/TSV with Function Score "
                             "and Protein Annotation columns.")

    parser.add_argument("--emb_mode", choices=["rna", "dna"], default="rna",
                        help="Embedding mode: 'rna' (single .npy) or "
                             "'dna' (ref + mut .npy files). Default: rna")

    # RNA args
    parser.add_argument("--emb",
                        help="[RNA mode] Path to RNA embedding .npy file.")

    # DNA args
    parser.add_argument("--emb_ref",
                        help="[DNA mode] Path to reference embedding .npy file.")
    parser.add_argument("--emb_mut",
                        help="[DNA mode] Path to mutant embedding .npy file.")
    parser.add_argument("--delta", action="store_true",
                        help="[DNA mode] Use delta (mut - ref) embeddings. "
                             "Default without this flag is concat (mut || ref).")

    # Model args
    parser.add_argument("--linear", action="store_true",
                        help="Use Ridge regression instead of LightGBM.")

    # Logging
    parser.add_argument("--log_dir", default="logs",
                        help="Directory for log files. Default: logs/")

    args = parser.parse_args()
    main(args)


# ── Usage examples ────────────────────────────────────────────────────────────
#
# RNA (original behaviour):
#   python regressor.py \
#       --df output/df_preprocessed.csv \
#       --emb_mode rna \
#       --emb output/rna_embeddings.npy \
#       --linear
#
# DNA delta (mut - ref), LightGBM:
#   python regressor.py \
#       --df output/df_preprocessed.csv \
#       --emb_mode dna --delta \
#       --emb_ref output/ref_embeddings.npy \
#       --emb_mut output/mut_embeddings.npy
#
# DNA concat (mut || ref), Ridge:
#   python regressor.py \
#       --df output/df_preprocessed.csv \
#       --emb_mode dna \
#       --emb_ref output/ref_embeddings.npy \
#       --emb_mut output/mut_embeddings.npy \
#       --linear