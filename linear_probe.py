"""
Performance for LighGBM:
── Hold-out test metrics ──────────────────────────────
R²  : 0.4790
RMSE: 0.2993
MAE : 0.2293
Pearson r: 0.6957

Performance for linear
── Hold-out test metrics ──────────────────────────────
R²  : 0.3659
RMSE: 0.3302
MAE : 0.2490
Pearson r: 0.6473

These models were trained on RNA embeddings, I want to use genomic embeddings!
"""

"""
Group-aware train/test split by protein annotation + linear regression.

Expected DataFrame columns:
  - 'sequence'   : protein sequence string
  - 'label'      : function score

All sequences sharing a cluster_id are kept together in either train or test —
no data leakage from identical sequences across splits.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import lightgbm as lgb


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_data(path_to_df, path_to_emb: str):
    """Load a CSV/TSV with at minimum: sequence, label, cluster_id columns."""
    sep = "\t" if path_to_df.endswith(".tsv") else ","
    df = pd.read_csv(path_to_df, sep=sep)
    emb = np.load(path_to_emb)
    return df, emb


# ── 3. Group-aware split ──────────────────────────────────────────────────────

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

    # Verify no cluster leakage
    train_clusters = set(groups[train_idx])
    test_clusters  = set(groups[test_idx])
    overlap = train_clusters & test_clusters
    assert len(overlap) == 0, f"Cluster leakage detected: {overlap}"

    print(f"Train: {len(train_idx)} samples | {len(train_clusters)} clusters")
    print(f"Test : {len(test_idx)}  samples | {len(test_clusters)}  clusters\n")

    return (emb[train_idx], emb[test_idx],
            y[train_idx], y[test_idx],
            train_idx, test_idx)


def group_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    linear: bool = False
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
            "fold":  fold + 1,
            "r2":    r2_score(y[val], y_pred),
            "rmse":  np.sqrt(mean_squared_error(y[val], y_pred)),
            "mae":   mean_absolute_error(y[val], y_pred),
        }
        fold_metrics.append(metrics)
        print(f"Fold {fold+1}: R²={metrics['r2']:.3f}  "
              f"RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}")
 
    print(f"\nMean CV R²  : {np.mean([m['r2']   for m in fold_metrics]):.3f}")
    print(f"Mean CV RMSE: {np.mean([m['rmse'] for m in fold_metrics]):.4f}\n")
    return fold_metrics


# ── 4. Model ──────────────────────────────────────────────────────────────────

def build_pipeline(linear: bool = False, use_ridge: bool = True) -> Pipeline:
    """
    StandardScaler → LinearRegression (or Ridge) pipeline.
    Ridge is recommended when features are collinear or n_features > n_samples.
    """
    if not linear:
        model = lgb.LGBMRegressor()
        print("Using LGBM Regressor")
    else:
        model = Ridge() if use_ridge else LinearRegression()
        print("Using linear model")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    model),
    ])


# ── 5. Evaluate ───────────────────────────────────────────────────────────────

def evaluate(pipe: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> None:
    y_pred = pipe.predict(X_test)
 
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
 
    print("── Hold-out test metrics ──────────────────────────────")
    print(f"R²  : {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
 
    # Pearson correlation between predicted and actual
    corr = np.corrcoef(y_test, y_pred)[0, 1]
    print(f"Pearson r: {corr:.4f}")


# ── 6. Main ───────────────────────────────────────────────────────────────────

def main(path_to_df: str,
         path_to_emb: str,
         linear: bool = False):
    df, emb     = load_data(path_to_df, path_to_emb)
    y      = df["Function Score"]
    groups = df["Protein Annotation"]


    # ── (b) Final hold-out evaluation ──
    print("=== Hold-out evaluation ===")
    X_train, X_test, y_train, y_test, groups_train, groups_test = group_train_test_split(
        emb, df['Function Score'], df['Protein Annotation'], test_size=0.2
    )

    # ── (a) Cross-validation on train dataset ──
    print("=== Group K-Fold CV ===")
    group_kfold_cv(X_train, y_train, groups_train, n_splits=3, linear = linear)

    pipe = build_pipeline(linear= linear)
    pipe.fit(X_train, y_train)
    evaluate(pipe, X_test, y_test)

    return pipe


if __name__ == "__main__":
    path_to_df = 'output/df_preprocessed.csv'
    path_to_emb = 'output/embeddings.npy'
    main(path_to_df, path_to_emb, True)