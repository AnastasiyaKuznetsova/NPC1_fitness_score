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

 The val_r works because predictions vary slightly across inner folds (each fold predicts a different mean). 
 For train/test with dummy, predictions are constant → NaN. The user just wants the raw computed value — remove the NaN→0 
 fallback entirely:

Now it returns whatever Spearman computes — NaN for constant predictions (dummy train/test), a real number otherwise. 
The NaN will show up in the summary and CSV as nan, which is the honest result. 

python3 regressor.py --emb_mode dna --emb emb_1B --strand forward --delta \
    --pca 50 --models Ridge --model_dir saved_models/

"""

import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic, DotProduct
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline

MODELS = [
    "Ridge", "Lasso", "ElasticNet",
    "KernelRidge", "SVR", "PLS", "GaussianProcess", "kNN",
    "RandomForest", "DecisionTree", "Dummy",
]

_GP_BOUNDS = (1e-7, 1e6)  # wider than sklearn's (1e-5, 1e5) default on every tunable
                           # kernel hyperparameter, so the internal MLE optimizer has
                           # room to move off the bound instead of raising ConvergenceWarning

GP_KERNELS = [
    RBF(length_scale_bounds=_GP_BOUNDS) + WhiteKernel(noise_level_bounds=_GP_BOUNDS),
    Matern(nu=1.5, length_scale_bounds=_GP_BOUNDS) + WhiteKernel(noise_level_bounds=_GP_BOUNDS),
    Matern(nu=2.5, length_scale_bounds=_GP_BOUNDS) + WhiteKernel(noise_level_bounds=_GP_BOUNDS),
    RationalQuadratic(length_scale_bounds=_GP_BOUNDS, alpha_bounds=_GP_BOUNDS)
        + WhiteKernel(noise_level_bounds=_GP_BOUNDS),
    DotProduct(sigma_0_bounds=_GP_BOUNDS) + WhiteKernel(noise_level_bounds=_GP_BOUNDS),
]

PARAM_GRIDS = {
    "Ridge":        {"reg__alpha": [0.01, 0.1, 1, 10, 100, 1000, 10000]},
    "Lasso":        {"reg__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]},
    "ElasticNet":   {"reg__alpha": [0.001, 0.01, 0.1], "reg__l1_ratio": [0.2, 0.5, 0.8]},
    "KernelRidge":  {"reg__alpha": [0.01, 0.1, 1, 10], "reg__gamma": [None, 0.001, 0.01, 0.1]},
    "SVR":          {"reg__C": [0.1, 1, 10, 100], "reg__gamma": ["scale", "auto"]},
    "PLS":          {"reg__n_components": [5, 10, 20, 30, 50]},
    "GaussianProcess":     {"reg__kernel": GP_KERNELS},
    "GaussianProcess-pca": {"dr__n_components": [10, 20, 50, 100], "reg__kernel": GP_KERNELS},
    "GaussianProcess-pls": {"dr__n_components": [5, 10, 20, 50],   "reg__kernel": GP_KERNELS},
    "kNN":          {"reg__n_neighbors": [3, 5, 10, 20, 50]},
    "RandomForest": {"reg__max_depth": [3, 5, 7], "reg__min_samples_leaf": [5, 10, 20]},
    "DecisionTree": {"reg__max_depth": [3, 5, 7], "reg__min_samples_leaf": [5, 10, 20]},
    "Dummy":        {},
}


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

    # Route Python warnings to the log file only (not stdout)
    logging.captureWarnings(True)
    warnings.simplefilter("always")
    logging.getLogger("py.warnings").handlers = [fh]
    logging.getLogger("py.warnings").propagate = False

    return log_path


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_data_rna(path_to_df: str, path_to_emb: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Original RNA path: single embedding matrix."""
    sep = "\t" if path_to_df.endswith(".tsv") else ","
    df  = pd.read_csv(path_to_df, sep=sep)
    emb = np.load(path_to_emb)
    return df, emb


def _load_ref_mut(emb_dir, strand: str, layer: str = None, pooling: str = None) -> tuple[np.ndarray, np.ndarray]:
    """Load refs_*{strand}*.npy and muts_*{strand}*.npy from a directory, squeeze to (N, D)."""
    if not hasattr(emb_dir, "glob"):
        emb_dir = Path(emb_dir)

    if layer is not None and pooling is not None:
        ref_files = sorted(emb_dir.rglob(f"ref_seq_*L{layer}*{pooling}*{strand}*.npy"))
        mut_files = sorted(emb_dir.rglob(f"mut_seq_*L{layer}*{pooling}*{strand}*.npy"))
    else:
        ref_files = sorted(emb_dir.glob(f"ref_seq_*{strand}*.npy"))
        mut_files = sorted(emb_dir.glob(f"mut_seq_*{strand}*.npy"))

    # Fall back to any ref/mut file if no strand-specific files found
    # (older directories where strand is indicated by folder name, not filename)
    if not ref_files:
        ref_files = sorted(emb_dir.glob("ref_seq_*.npy"))
    if not mut_files:
        mut_files = sorted(emb_dir.glob("mut_seq_*.npy"))

    if not ref_files:
        raise FileNotFoundError(f"No ref_seq_*.npy files found in {str(emb_dir)}")
    if not mut_files:
        raise FileNotFoundError(f"No mut_seq_*.npy files found in {str(emb_dir)}")
    if len(ref_files) != 1 or len(mut_files) != 1:
        raise ValueError(
            f"Expected exactly one refs and one muts file for strand '{strand}', "
            f"found refs: {ref_files} and muts: {mut_files}"
        )

    logging.info(f"  ref: {ref_files[0]}")
    logging.info(f"  mut: {mut_files[0]}")

    ref = np.load(ref_files[0])
    mut = np.load(mut_files[0])

    if ref.ndim == 3:
        ref = ref.squeeze(1)
    if mut.ndim == 3:
        mut = mut.squeeze(1)

    assert ref.shape == mut.shape, (
        f"ref/mut shape mismatch: {ref.shape} vs {mut.shape}"
    )
    return ref, mut


def _pool(ref: np.ndarray, mut: np.ndarray, delta: bool) -> np.ndarray:
    """Apply delta or concat pooling to a ref/mut pair."""
    if delta:
        return mut - ref
    return np.concatenate([mut, ref], axis=1)


def load_data_dna(
    path_to_df: str,
    emb: str,
    delta: bool,
    use_reverse: bool = False,
    strand: str = "forward",
    layer: str = None,
    pooling: str = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    DNA path: loads strand embeddings from emb directory.
      strand='forward'  → forward only
      strand='reverse'  → reverse only
      use_reverse=True  → forward + reverse concatenated
    """
    sep = "\t" if path_to_df.endswith(".tsv") else ","
    df  = pd.read_csv(path_to_df, sep=sep)

    if use_reverse:
        logging.info("Loading forward-strand embeddings:")
        ref_fwd, mut_fwd = _load_ref_mut(emb, "forward", layer=layer, pooling=pooling)
        emb_fwd = _pool(ref_fwd, mut_fwd, delta)
        logging.info(f"  forward pool shape: {emb_fwd.shape}")
        logging.info("Loading reverse-strand embeddings:")
        ref_rev, mut_rev = _load_ref_mut(emb, "reverse", layer=layer, pooling=pooling)
        emb_rev = _pool(ref_rev, mut_rev, delta)
        logging.info(f"  reverse pool shape: {emb_rev.shape}")
        emb_out = np.concatenate([emb_fwd, emb_rev], axis=1)
        logging.info(f"  combined shape: {emb_out.shape}")
    else:
        logging.info(f"Loading {strand}-strand embeddings:")
        ref, mut = _load_ref_mut(emb, strand, layer=layer, pooling=pooling)
        emb_out = _pool(ref, mut, delta)
        logging.info(f"  {strand} pool shape: {emb_out.shape}")

    logging.info(f"Loaded {emb_out.shape[0]} variants | final dim {emb_out.shape[1]}")
    return df, emb_out


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation. .statistic added in scipy 1.9; falls back to .correlation."""
    result = spearmanr(a, b)
    corr = getattr(result, "statistic", None)
    if corr is None:
        corr = getattr(result, "correlation", float("nan"))
    return float(corr)


_spearman_scorer = make_scorer(_safe_corr)


def _best_params_repr(model_name: str, gs: GridSearchCV, best_pipe: Pipeline, dr_mode: str = None) -> str:
    """Build a readable string of the selected hyperparameters for logging.

    For GaussianProcess variants:
      - kernel family is selected by inner CV (from gs.best_params_)
      - kernel hyperparameters (length_scale, noise) are optimised internally by MLE
        and only visible on the fitted kernel_ attribute — so we log that, not the prototype
      - DR component count (if PCA/PLS) comes from gs.best_params_
    """
    parts = []

    if model_name == "GaussianProcess":
        gp = best_pipe.named_steps["reg"]
        fitted_kernel = gp.kernel_
        kernel_family = type(fitted_kernel.k1).__name__ if hasattr(fitted_kernel, "k1") else type(fitted_kernel).__name__

        # DR component count (only present for pca/pls variants)
        if dr_mode in ("pca", "pls") and "dr__n_components" in gs.best_params_:
            parts.append(f"n_components={gs.best_params_['dr__n_components']}")

        parts.append(f"kernel={kernel_family}")
        parts.append(f"fitted_kernel={fitted_kernel}")
        return " | ".join(parts)

    return str(gs.best_params_)


# ── 2. Cross-validation ───────────────────────────────────────────────────────

def simple_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    pca_components: int = None,
    dr_mode: str = None,
) -> tuple[list[dict], object]:
    """2-fold group cross-validation (no inner loop) for when only 2 groups exist."""
    cv = GroupKFold(n_splits=2)
    fold_metrics = []
    best_model = None
    best_test_corr = -np.inf

    for fold, (tr, te) in enumerate(cv.split(X, y, groups=groups), 1):
        pipe = build_pipeline(model_name=model_name, pca_components=pca_components, dr_mode=dr_mode)
        pipe.fit(X[tr], y[tr])

        y_train_pred = pipe.predict(X[tr])
        y_test_pred  = pipe.predict(X[te])

        train_corr = _safe_corr(y[tr], y_train_pred)
        test_corr  = _safe_corr(y[te], y_test_pred)
        test_mse   = mean_squared_error(y[te], y_test_pred)
        test_mae   = mean_absolute_error(y[te], y_test_pred)
        train_mse  = mean_squared_error(y[tr], y_train_pred)
        train_mae  = mean_absolute_error(y[tr], y_train_pred)

        logging.info(
            f"Fold {fold}/2 | train r={train_corr:.3f} | "
            f"test r={test_corr:.3f} MSE={test_mse:.4f} MAE={test_mae:.4f}"
        )

        if not np.isnan(test_corr) and test_corr > best_test_corr:
            best_test_corr = test_corr
            best_model = pipe

        fold_metrics.append({
            "outer_fold": fold,
            "train_corr": float(train_corr), "train_mse": float(train_mse), "train_mae": float(train_mae),
            "val_corr": float("nan"),         "val_mse":   float("nan"),     "val_mae":   float("nan"),
            "test_corr": float(test_corr),   "test_mse":  float(test_mse),  "test_mae":  float(test_mae),
        })

    logging.info(f"2-fold CV mean test corr: {np.mean([m['test_corr'] for m in fold_metrics]):.3f}")
    return fold_metrics, best_model


def nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    outer_splits: int = 5,
    inner_splits: int = 3,
    pca_components: int = None,
    dr_mode: str = None,
) -> list[dict]:
    """
    Nested group K-Fold cross-validation.

    Outer loop (outer_splits folds): unbiased performance estimate —
      each outer test fold is never seen during inner loop or model fit.
    Inner loop (inner_splits folds): validation on the outer train split —
      reports inner CV metrics alongside the outer test metrics.

    Returns list of per-outer-fold dicts with both inner CV and outer test metrics.
    """
    outer_cv = GroupKFold(n_splits=outer_splits)
    inner_cv = GroupKFold(n_splits=inner_splits)
    param_grid_key = f"{model_name}-{dr_mode}" if dr_mode else model_name
    param_grid = PARAM_GRIDS.get(param_grid_key, PARAM_GRIDS.get(model_name, {}))
    gs_scoring = {
        "spearman": _spearman_scorer,
        "neg_mse":  "neg_mean_squared_error",
        "neg_mae":  "neg_mean_absolute_error",
    }
    fold_metrics = []
    best_model = None
    best_test_corr = -np.inf

    for outer_fold, (outer_tr, outer_te) in enumerate(outer_cv.split(X, y, groups=groups), 1):
        X_outer_tr, X_outer_te = X[outer_tr], X[outer_te]
        y_outer_tr, y_outer_te = y[outer_tr], y[outer_te]
        g_outer_tr = groups[outer_tr]

        # ── Inner loop: hyperparameter tuning via GridSearchCV ────────────────
        pipe = build_pipeline(model_name=model_name, pca_components=pca_components, dr_mode=dr_mode)
        if param_grid:
            gs = GridSearchCV(
                pipe, param_grid,
                cv=inner_cv,
                scoring=gs_scoring,
                refit="spearman",
                n_jobs=-1,
            )
            gs.fit(X_outer_tr, y_outer_tr, groups=g_outer_tr)
            best_pipe = gs.best_estimator_
            best_idx  = gs.best_index_
            val_corr  = float(gs.cv_results_["mean_test_spearman"][best_idx])
            val_mse   = float(-gs.cv_results_["mean_test_neg_mse"][best_idx])
            val_mae   = float(-gs.cv_results_["mean_test_neg_mae"][best_idx])
            best_params = _best_params_repr(model_name, gs, best_pipe, dr_mode=dr_mode)
            logging.info(f"  Best params: {best_params}")
        else:
            # No grid to search — fit once and get val metrics via manual inner CV
            val_y_true, val_y_pred = [], []
            for inner_tr, inner_val in inner_cv.split(X_outer_tr, y_outer_tr, groups=g_outer_tr):
                p = build_pipeline(model_name=model_name, pca_components=pca_components, dr_mode=dr_mode)
                p.fit(X_outer_tr[inner_tr], y_outer_tr[inner_tr])
                val_y_true.append(y_outer_tr[inner_val])
                val_y_pred.append(p.predict(X_outer_tr[inner_val]))
            val_y_true = np.concatenate(val_y_true)
            val_y_pred = np.concatenate(val_y_pred)
            val_mse  = mean_squared_error(val_y_true, val_y_pred)
            val_mae  = mean_absolute_error(val_y_true, val_y_pred)
            val_corr = _safe_corr(val_y_true, val_y_pred)
            pipe.fit(X_outer_tr, y_outer_tr)
            best_pipe = pipe

        # ── Outer test ────────────────────────────────────────────────────────

        y_train_pred = best_pipe.predict(X_outer_tr)
        y_outer_pred = best_pipe.predict(X_outer_te)

        train_mse  = mean_squared_error(y_outer_tr, y_train_pred)
        train_mae  = mean_absolute_error(y_outer_tr, y_train_pred)
        train_corr = _safe_corr(y_outer_tr, y_train_pred)

        test_mse  = mean_squared_error(y_outer_te, y_outer_pred)
        test_mae  = mean_absolute_error(y_outer_te, y_outer_pred)
        test_corr = _safe_corr(y_outer_te, y_outer_pred)

        best_params_str = f" | best={_best_params_repr(model_name, gs, best_pipe, dr_mode=dr_mode)}" if param_grid else ""
        logging.info(
            f"Outer fold {outer_fold}/{outer_splits} | "
            f"train r={train_corr:.3f} | "
            f"val r={val_corr:.3f} MSE={val_mse:.4f} | "
            f"test r={test_corr:.3f} MSE={test_mse:.4f} MAE={test_mae:.4f}"
            f"{best_params_str}"
        )

        if not np.isnan(test_corr) and test_corr > best_test_corr:
            best_test_corr = test_corr
            best_model = best_pipe

        fold_metrics.append({
            "outer_fold": outer_fold,
            "train_corr": float(train_corr),
            "train_mse":  float(train_mse),
            "train_mae":  float(train_mae),
            "val_corr":   float(val_corr),
            "val_mse":    float(val_mse),
            "val_mae":    float(val_mae),
            "test_corr":  float(test_corr),
            "test_mse":   float(test_mse),
            "test_mae":   float(test_mae),
        })

    logging.info(f"Nested CV mean test corr: {np.mean([m['test_corr'] for m in fold_metrics]):.3f}")
    logging.info(f"Nested CV mean test MSE : {np.mean([m['test_mse']  for m in fold_metrics]):.4f}")
    logging.info(f"Nested CV mean test MAE : {np.mean([m['test_mae']  for m in fold_metrics]):.4f}")
    return fold_metrics, best_model


# ── 3. Model ──────────────────────────────────────────────────────────────────

def build_pipeline(model_name: str, pca_components: int = None, dr_mode: str = None) -> Pipeline:
    """Build a StandardScaler [+ DR] + model pipeline.

    dr_mode: None | 'pca' | 'pls'
      - 'pca' : unsupervised PCA before the model
      - 'pls' : supervised PLSRegression used as a transformer (fitted on X and y)
      - None  : no dimensionality reduction
    Only used when pca_components is set. For GaussianProcess, all three are run separately.
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {MODELS}")

    if model_name == "Dummy":
        logging.info("Using Dummy Regressor (mean baseline)")
        return Pipeline([("reg", DummyRegressor(strategy="mean"))])

    tree_based = model_name in ("RandomForest", "DecisionTree", "LightGBM")
    scaler_step = [] if tree_based else [("scaler", StandardScaler())]

    # Dimensionality reduction step
    if pca_components and dr_mode == "pca" and model_name != "PLS":
        dr_step = [("dr", PCA(n_components=pca_components, random_state=42))]
        dr_label = f"PCA({pca_components}) + "
    elif pca_components and dr_mode == "pls" and model_name != "PLS":
        dr_step = [("dr", PLSRegression(n_components=min(pca_components, 20)))]
        dr_label = f"PLS-DR({pca_components}) + "
    elif pca_components and model_name != "PLS" and dr_mode is None:
        # Legacy behaviour for non-GP models: --pca uses PCA by default
        dr_step = [("dr", PCA(n_components=pca_components, random_state=42))]
        dr_label = f"PCA({pca_components}) + "
    else:
        dr_step = []
        dr_label = ""

    use_pca = bool(pca_components) and model_name != "PLS"
    pca_label = dr_label

    regressors = {
        "Ridge":       Ridge(alpha=100.0),
        "Lasso":       Lasso(alpha=0.01, max_iter=5000),
        "ElasticNet":  ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        "KernelRidge": KernelRidge(kernel="rbf", alpha=1.0),
        "SVR":         SVR(kernel="rbf", C=1.0),
        "PLS":         PLSRegression(n_components=min(pca_components or 20, 20)),
        "GaussianProcess": GaussianProcessRegressor(
            kernel=RBF(length_scale_bounds=_GP_BOUNDS) + WhiteKernel(noise_level_bounds=_GP_BOUNDS),
            random_state=42, normalize_y=True, n_restarts_optimizer=5,
        ),
        "kNN":          KNeighborsRegressor(n_neighbors=5, weights="distance"),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=10,
            max_features="sqrt", random_state=42, n_jobs=-1,
        ),
        "DecisionTree": DecisionTreeRegressor(
            max_depth=5, min_samples_leaf=10, random_state=42,
        ),
    }

    logging.info(f"Using {pca_label}{model_name}")
    return Pipeline([
        *scaler_step,
        *dr_step,
        ("reg", regressors[model_name]),
    ])
  


# ── 4. Evaluate ───────────────────────────────────────────────────────────────

# ── 5. Run one embedding matrix end-to-end ────────────────────────────────────

def run_single(
    emb: np.ndarray,
    df: pd.DataFrame,
    model_name: str,
    tag: str = "",
    outer_splits: int = 5,
    inner_splits: int = 3,
    pca_components: int = None,
    model_dir: Path = None,
    fold_by: str = "Protein Annotation",
    dr_mode: str = None,
) -> dict:
    """Full pipeline for one embedding matrix using nested CV. Returns mean ± std metrics."""
    logging.info("=" * 60)
    logging.info(f"  {tag}")

    if model_name == "Dummy":
        y = df["Function Score"].to_numpy()
        y_pred = np.full_like(y, y.mean())
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        corr = _safe_corr(y, y_pred)
        logging.info(f"Dummy baseline: mean={y.mean():.4f} | corr={corr} MSE={mse:.4f} MAE={mae:.4f}")
        result = {"tag": tag}
        for metric in ("train_corr", "train_mse", "train_mae",
                       "val_corr",   "val_mse",   "val_mae",
                       "test_corr",  "test_mse",  "test_mae"):
            result[metric]          = corr if "corr" in metric else (mse if "mse" in metric else mae)
            result[f"{metric}_std"] = float("nan")
        return result
    logging.info("=" * 60)

    if fold_by not in df.columns:
        raise ValueError(f"--fold_by column '{fold_by}' not found in dataframe. "
                         f"Available columns: {list(df.columns)}")
    groups = df[fold_by].to_numpy()
    y      = df["Function Score"].to_numpy()
    n_groups = len(np.unique(groups))
    logging.info(f"Grouping by: '{fold_by}' ({n_groups} unique groups)")

    if n_groups == 2:
        fold_metrics, best_model = simple_cv(emb, y, groups, model_name=model_name,
                                             pca_components=pca_components, dr_mode=dr_mode)
    else:
        fold_metrics, best_model = nested_cv(emb, y, groups, model_name=model_name,
                                             outer_splits=outer_splits, inner_splits=inner_splits,
                                             pca_components=pca_components, dr_mode=dr_mode)

    if model_dir is not None and best_model is not None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        safe_tag = tag.replace(" ", "_").replace("|", "").replace("/", "-")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = model_dir / f"{safe_tag}_{timestamp}.joblib"
        joblib.dump(best_model, model_path)
        logging.info(f"Best model saved to: {model_path}")

    result = {"tag": tag}
    for metric in ("train_corr", "train_mse", "train_mae",
                   "val_corr",   "val_mse",   "val_mae",
                   "test_corr",  "test_mse",  "test_mae"):
        vals = [m[metric] for m in fold_metrics]
        result[metric]          = float(np.mean(vals))
        result[f"{metric}_std"] = float(np.std(vals, ddof=1))
    return result


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
    elif args.emb_mode == "dna":
        if not args.emb:
            raise ValueError("--emb is required for --emb_mode dna")
        use_reverse = args.strand == "both"
        strand = "forward" if args.strand != "reverse" else "reverse"
        regime = "delta" if args.delta else "concat"
        df, emb = load_data_dna(args.df, args.emb, args.delta, use_reverse=use_reverse,
                                 strand=strand, layer=args.layer, pooling=args.pooling)
    else:
        raise ValueError(f"Unknown emb_mode: {args.emb_mode}")

    regime_label = "delta" if args.delta else "concat"
    for model_name in args.models:
        if model_name == "GaussianProcess":
            # Run GP with three DR options: PCA, PLS (supervised), and none
            gp_dr_options = [("pca", "pca"), ("pls", "pls"), ("none", None)]
            if not args.pca:
                gp_dr_options = [("none", None)]  # no components specified — only plain GP
            for dr_label, dr_mode in gp_dr_options:
                tag = f"GaussianProcess-{dr_label} | {regime_label} | {args.strand}"
                results.append(run_single(emb, df, model_name="GaussianProcess",
                                          tag=tag, pca_components=args.pca,
                                          model_dir=args.model_dir, fold_by=args.fold_by,
                                          dr_mode=dr_mode))
        else:
            tag = f"{model_name} | {regime_label} | {args.strand}"
            results.append(run_single(emb, df, model_name=model_name,
                                      tag=tag, pca_components=args.pca,
                                      model_dir=args.model_dir, fold_by=args.fold_by))
    results.append(run_single(emb, df, model_name="Dummy",
                              tag="Dummy (mean baseline)", pca_components=None,
                              model_dir=None, fold_by=args.fold_by))

    # ── Summary table ──────────────────────────────────────────────────────
    logging.info("\n" + "=" * 60)
    logging.info("  SUMMARY")
    logging.info("=" * 60)
    logging.info(f"{'Tag':<35} {'train_r':>8} {'stdev':>7} {'val_r':>7} {'stdev':>7} {'test_r':>7} {'stdev':>7} {'test_mse':>9} {'stdev':>7} {'test_mae':>9} {'stdev':>7}")
    logging.info("-" * 115)
    for m in results:
        logging.info(
            f"{m['tag']:<35} "
            f"{m['train_corr']:>8.4f} {m['train_corr_std']:>7.4f} "
            f"{m['val_corr']:>7.4f} {m['val_corr_std']:>7.4f} "
            f"{m['test_corr']:>7.4f} {m['test_corr_std']:>7.4f} "
            f"{m['test_mse']:>9.4f} {m['test_mse_std']:>7.4f} "
            f"{m['test_mae']:>9.4f} {m['test_mae_std']:>7.4f}"
        )

    # ── Save results CSV ───────────────────────────────────────────────────
    results_dir = Path(args.log_dir)
    results_csv = results_dir / f"results_{log_path.stem.removeprefix('run_')}.csv"
    pd.DataFrame(results).to_csv(results_csv, index=False)
    logging.info(f"Results saved to: {results_csv.resolve()}")
    logging.info(f"\nFull log saved to: {log_path.resolve()}")


# ── 7. CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train regressor on RNA or DNA (delta/concat) embeddings."
    )

    parser.add_argument("--df", default="output/df_preprocessed.csv",
                        help="Path to preprocessed CSV/TSV with Function Score "
                             "and Protein Annotation columns.")

    parser.add_argument("--emb_mode", choices=["rna", "dna"], default="dna",
                        help="Embedding mode: 'rna' (single .npy) or "
                             "'dna' (ref + mut .npy files). Default: rna")

    # RNA args
    parser.add_argument("--emb",
                        help="[RNA mode] Path to RNA embedding .npy file.")

    parser.add_argument("--strand", choices=["forward", "reverse", "both"], default="forward",
                        help="[DNA mode] Which strand(s) to use: 'forward', 'reverse', or "
                             "'both' (forward + reverse concatenated). Default: forward.")
    parser.add_argument("--delta", action="store_true",
                        help="[DNA mode] Use delta (mut - ref) embeddings. "
                             "Default without this flag is concat (mut || ref).")
    parser.add_argument("--layer", default=None,
                        help="[DNA mode] Layer index to select (e.g. 27). Required when the "
                             "embedding directory contains multiple layers.")
    parser.add_argument("--pooling", choices=["average", "last"], default=None,
                        help="[DNA mode] Pooling mode: 'average' or 'last'. Required when the "
                             "embedding directory contains multiple pooling modes.")

    parser.add_argument("--fold_by", choices=["Protein Annotation", "merged_region"],
                        default="Protein Annotation",
                        help="Column to use for group-aware fold splitting. "
                             "'Protein Annotation' (default) or 'merged_region'.")

    # Model args
    parser.add_argument("--models", nargs="+", default=["Ridge"],
                        choices=MODELS,
                        help=f"Models to run (default: Ridge). Choices: {MODELS}")

    # Logging
    parser.add_argument("--pca", type=int, default=None,
                        help="Number of PCA components before regression. "
                             "Omit to skip PCA (default: no PCA).")
    parser.add_argument("--model_dir", default=None,
                        help="Directory to save the best model per run as a .joblib file. "
                             "Omit to skip saving.")
    parser.add_argument("--log_dir", default="logs",
                        help="Directory for log files. Default: logs/")

    args = parser.parse_args()
    main(args)


# ── Usage examples ────────────────────────────────────────────────────────────
#
# RNA, Ridge:
#   python regressor.py \
#       --df output/df_preprocessed.csv \
#       --emb_mode rna \
#       --emb output/rna_embeddings.npy \
#       --linear
#
# DNA delta, forward strand, Ridge + SVR, PCA(50):
#   python regressor.py \
#       --df output/df_preprocessed.csv \
#       --emb_mode dna --delta \
#       --emb embeddings/ \
#       --strand forward --pca 50 --models Ridge SVR
#
# DNA concat, reverse strand, ElasticNet + SVR, PCA(50):
#   python regressor.py \
#       --df output/df_preprocessed.csv \
#       --emb_mode dna \
#       --emb embeddings/ \
#       --strand reverse --pca 50 --models ElasticNet SVR
#
# DNA delta, forward + reverse, all models, PCA(50):
#   python regressor.py \
#       --df output/df_preprocessed.csv \
#       --emb_mode dna --delta \
#       --emb embeddings/ \
#       --strand both --pca 50 \
#       --models Ridge Lasso ElasticNet KernelRidge SVR PLS GaussianProcess kNN