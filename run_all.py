"""
Run all regressor combinations and save a single structured metrics CSV.

Iterates over all layers found in --emb, both delta/concat, every regressor in MODELS.
Output CSV columns:
  layer, model_arch, emb_type, strand, lightweight_model, hyperparams,
  train_corr, train_mse, train_mae,
  val_corr,   val_mse,   val_mae,
  test_corr,  test_mse,  test_mae

Command-line options:
  --df         Path to the preprocessed dataframe CSV (default: output/df_preprocessed.csv).
  --emb        Directory with ref_seq_*/mut_seq_* .npy embedding files (required).
  --strand     'forward', 'reverse', or 'both' (concatenated fwd+rev) (default: forward).
  --out_dir    Root directory for results (default: all_results). Each
               (model, layer, pooling, strand[, downstream_k]) group gets its own
               subfolder named "{MODEL}_{PARAMS}_L{layer}_{pooling}_{strand}[_{k}]"
               (model/params parsed from the --emb directory name) containing that
               group's per-run .log files and a metrics.csv; a combined metrics.csv
               for the whole invocation is also written at the --out_dir root.
  --model_dir  Directory to save each run's best model as a date-stamped .joblib
               file (default: saved_models).
  --n_jobs     CPU cores per GridSearchCV inner search (default: 4). Set to a
               slice of your reserved cores when running several run_all.py
               jobs in parallel on the same node, so they don't oversubscribe
               each other.

GaussianProcess is always run twice per layer/pooling/emb_type combo — once with
PCA and once with PLS dimensionality reduction before the GP itself (component
count is tuned by the inner grid search) — no flag needed.
"""

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from regressor import load_data_dna, nested_cv, build_pipeline, parse_model_label, MODELS


def get_hyperparams(model_name: str, dr_mode: str = None) -> str:
    if model_name == "Dummy":
        return "strategy=mean"
    dr_str = f"dr={dr_mode}, " if dr_mode else ""
    pipe = build_pipeline(model_name=model_name, dr_mode=dr_mode)
    reg = pipe.named_steps["reg"]
    params = reg.get_params()
    key_params = {
        "Ridge":          ["alpha"],
        "Lasso":          ["alpha"],
        "ElasticNet":     ["alpha", "l1_ratio"],
        "KernelRidge":    ["alpha", "kernel"],
        "SVR":            ["C", "kernel"],
        "PLS":            ["n_components"],
        "GaussianProcess": ["kernel"],
        "kNN":            ["n_neighbors", "weights"],
        "RandomForest":   ["n_estimators", "max_depth", "min_samples_leaf"],
        "DecisionTree":   ["max_depth", "min_samples_leaf"],
    }.get(model_name, [])
    param_str = ", ".join(f"{k}={params[k]}" for k in key_params if k in params)
    return f"{dr_str}{param_str}" if param_str else dr_str or "default"


def extract_layer_pooling_combos(emb_dir: Path, strand: str) -> list[tuple[str, str]]:
    """Return sorted unique (layer_index, pooling_mode) pairs from refs_*{strand}*.npy filenames."""
    files = sorted(emb_dir.rglob(f"ref_seq_*{strand}*.npy"))
    combos = []
    for f in files:
        # Layer token is usually numeric (e.g. "_L27_") but can also be a name
        # like "_Lfinal_" (DNABERT-2's last-layer embeddings have no layer index).
        layer_m = re.search(r"_L(\d+|final)_", f.name)
        pool_m  = re.search(r"_L(?:\d+|final)_(average|last)_", f.name)
        if layer_m and pool_m:
            combos.append((layer_m.group(1), pool_m.group(1)))
    combos = sorted(set(combos))
    if not combos:
        raise FileNotFoundError(
            f"No ref_seq_*{strand}*.npy files with _L{{n}}_(average|last)_ pattern in {emb_dir}"
        )
    return combos


def extract_downstream_k_variants(emb_dir: Path, strand: str) -> list:
    """Return every --pool-region region found in ref_seq_*{strand}*.npy filenames:
    None for the plain full-sequence files, plus a string per '_ds{k}' suffix
    (e.g. '0', '32', 'all') written by extract_embeddings.py --pool-region downstream.
    Sorted with None (full) first, then k's in ascending numeric order ('all' last).
    """
    files = sorted(emb_dir.rglob(f"ref_seq_*{strand}*.npy"))
    variants = set()
    for f in files:
        m = re.search(r"_ds(\w+)\.npy$", f.name)
        variants.add(m.group(1) if m else None)

    def sort_key(k):
        if k is None:
            return (0, 0)
        if k == "all":
            return (2, 0)
        return (1, int(k))
    return sorted(variants, key=sort_key)


def append_result_row(csv_path: Path, row: dict) -> None:
    """Append one result row to csv_path immediately, writing the header only on
    the first write. Used so metrics.csv holds every completed run's results
    even if a later run in the sweep raises or the process is killed."""
    write_header = not csv_path.exists()
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=write_header, index=False)


def setup_run_logging(out_dir: Path, run_name: str) -> None:
    log_path = out_dir / f"{run_name}.log"
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(sh)


def run_combo(emb, df, model_name, tag, dr_mode=None, model_dir=None, run_timestamp=None, n_jobs=4) -> dict:
    groups = df["Protein Annotation"].to_numpy()
    y = df["Function Score"].to_numpy()
    logging.info(f"Grouping by: 'Protein Annotation' ({len(np.unique(groups))} unique groups)")
    fold_metrics, best_model = nested_cv(emb, y, groups, model_name=model_name,
                                         outer_splits=5, inner_splits=3,
                                         dr_mode=dr_mode, n_jobs=n_jobs)

    if model_dir is not None and best_model is not None:
        import joblib
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        safe_tag = tag.replace(" ", "_").replace("|", "").replace("/", "-")
        date_str = run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = model_dir / f"{safe_tag}_{date_str}.joblib"
        joblib.dump(best_model, model_path)
        logging.info(f"Best model saved to: {model_path}")

    result = {}
    for metric in ("train_corr", "train_mse", "train_mae",
                   "val_corr",   "val_mse",   "val_mae",
                   "test_corr",  "test_mse",  "test_mae"):
        vals = [m[metric] for m in fold_metrics]
        result[metric]          = float(np.mean(vals))
        result[f"{metric}_std"] = float(np.std(vals, ddof=1))
    return result


def main():
    parser = argparse.ArgumentParser(description="Run all regressor combinations.")
    parser.add_argument("--df", default="output/df_preprocessed.csv",
                        help="Path to the preprocessed dataframe CSV containing "
                             "'Protein Annotation' and 'Function Score' columns. "
                             "Default: output/df_preprocessed.csv.")
    parser.add_argument("--emb", required=True,
                        help="Directory with refs_*/muts_* .npy embedding files.")
    parser.add_argument("--strand", choices=["forward", "reverse", "both"], default="forward",
                        help="Which strand(s) to use: 'forward', 'reverse', or 'both' (concatenated). "
                             "Default: forward.")
    parser.add_argument("--out_dir", default="all_results",
                        help="Root directory for results; each group gets its own "
                             "subfolder (see module docstring). Default: all_results.")
    parser.add_argument("--model_dir", default="saved_models",
                        help="Directory to save the best model per run as a .joblib file. "
                             "Default: saved_models.")
    parser.add_argument("--models", nargs="+", default=["all"],
                        choices=[m for m in MODELS if m != "Dummy"] + ["all"],
                        help="Which non-Dummy models to sweep (default: all). Dummy is always "
                             "included as the baseline regardless of this flag. E.g. to skip "
                             "GaussianProcess: --models Ridge Lasso ElasticNet KernelRidge SVR "
                             "PLS kNN RandomForest DecisionTree")
    parser.add_argument("--emb-types", nargs="+", default=["delta", "concat"],
                        choices=["delta", "concat"],
                        help="Which embedding regimes to sweep (default: both). Use "
                             "'--emb-types delta' to skip concat entirely.")
    parser.add_argument("--n_jobs", type=int, default=4,
                        help="CPU cores for each GridSearchCV inner hyperparameter search "
                             "(default: 4; use -1 for all cores visible to this process). "
                             "At this dataset's size, most models fit fast enough that wide "
                             "parallelism mainly helps GaussianProcess; keeping this modest "
                             "leaves cores free to run several run_all.py jobs in parallel "
                             "on the same node/allocation without oversubscribing.")
    args = parser.parse_args()
    if "all" in args.models:
        args.models = [m for m in MODELS if m != "Dummy"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = Path(args.emb)
    model_label = parse_model_label(emb_dir)
    # Nest saved models under a subfolder named for the foundation model the
    # embeddings came from, so saved_models/ doesn't mix DNABERT-2/Evo2/etc. runs.
    model_dir = str(Path(args.model_dir) / model_label)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    delta_options = ([True] if args.emb_types == ["delta"] else
                      [False] if args.emb_types == ["concat"] else [True, False])
    model_configs = [
        {
            "model_name": model_name, "delta": delta,
            "emb_type": "delta" if delta else "concat",
            "dr_mode": dr_mode,
            "label": f"{model_name}-{dr_mode}" if dr_mode else model_name,
        }
        for model_name in args.models
        for delta in delta_options
        # GaussianProcess always runs both PCA and PLS dimensionality reduction;
        # every other model runs with no DR step.
        for dr_mode in (["pca", "pls"] if model_name == "GaussianProcess" else [None])
    ] + [{"model_name": "Dummy", "delta": True, "emb_type": "none", "dr_mode": None, "label": "Dummy"}]

    strand_configs = {
        "forward":  [("forward",  False, "forward")],
        "reverse":  [("reverse",  False, "reverse")],
        "both":     [("fwd+rev",  True,  "forward")],
    }[args.strand]
    # strand_configs entries: (label, use_reverse, ref_strand_for_layer_discovery)

    all_results = []
    root_csv_path = out_dir / "metrics.csv"
    root_csv_path.unlink(missing_ok=True)  # start each invocation with a clean combined csv

    for strand_label, use_reverse, ref_strand in strand_configs:
        layer_pool_combos = extract_layer_pooling_combos(emb_dir, ref_strand)
        downstream_k_variants = extract_downstream_k_variants(emb_dir, ref_strand)
        for layer, pooling in layer_pool_combos:
            for k in downstream_k_variants:
                k_label = "full" if k is None else f"ds{k}"
                group_name = f"{model_label}_L{layer}_{pooling}_{strand_label}"
                if k_label != "full":
                    group_name += f"_{k_label}"
                group_out_dir = out_dir / group_name
                group_out_dir.mkdir(parents=True, exist_ok=True)
                setup_run_logging(group_out_dir, group_name)
                group_csv_path = group_out_dir / "metrics.csv"
                group_csv_path.unlink(missing_ok=True)  # start each group with a clean csv

                # Load embeddings once per layer/pooling/strand/k — shared across model combos
                layer_emb = _LayerFilteredDir(emb_dir, layer, pooling)
                try:
                    df_fwd, emb_fwd = load_data_dna(
                        args.df, layer_emb, delta=False, use_reverse=use_reverse,
                        strand=ref_strand, downstream_k=k,
                    )
                    _, emb_delta = load_data_dna(
                        args.df, layer_emb, delta=True, use_reverse=use_reverse,
                        strand=ref_strand, downstream_k=k,
                    )
                except Exception as e:
                    logging.error(f"FAILED loading {group_name}: {e}")
                    continue

                for c in model_configs:
                    run_name = f"{group_name}_{c['label']}_{c['emb_type']}"
                    logging.info(f"Run: {run_name}")
                    emb_data = emb_delta if c["delta"] else emb_fwd

                    try:
                        metrics = run_combo(
                            emb_data, df_fwd, model_name=c["model_name"],
                            tag=run_name, dr_mode=c["dr_mode"],
                            model_dir=model_dir, run_timestamp=run_timestamp,
                            n_jobs=args.n_jobs,
                        )
                    except Exception as e:
                        logging.error(f"FAILED {run_name}: {e}")
                        continue

                    row = {
                        "layer":        f"L{layer}",
                        "model_arch":   model_label,
                        "pooling":      pooling,
                        "downstream_k": k_label,
                        "emb_type":     c["emb_type"],
                        "strand":       strand_label,
                        "model":        c["label"],
                        "hyperparams": get_hyperparams(c["model_name"], dr_mode=c["dr_mode"]),
                        **metrics,
                    }
                    # Written immediately so a later run's failure or a killed
                    # process still leaves every completed run's results on disk.
                    append_result_row(group_csv_path, row)
                    append_result_row(root_csv_path, row)
                    all_results.append(row)

    metrics_df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    cols = ["layer", "model_arch", "pooling", "downstream_k", "emb_type", "strand", "model",
            "train_corr", "train_corr_std", "val_corr", "val_corr_std",
            "test_corr", "test_corr_std", "test_mse", "test_mse_std", "test_mae", "test_mae_std"]
    summary_cols = [c for c in cols if c in metrics_df.columns]
    print(metrics_df[summary_cols].to_string(index=False))
    print(f"\nFull metrics saved to: {root_csv_path.resolve()}")


class _LayerFilteredDir:
    """Wraps an emb directory and restricts glob results to a specific layer and pooling mode."""
    def __init__(self, path: Path, layer: str, pooling: str):
        self._path = path
        self._layer = layer
        self._pooling = pooling

    def glob(self, pattern: str):
        # Filename format: {ref_seq|mut_seq}_*_L{layer}_{pooling}_{strand}[_ds{k}].npy
        # _load_ref_mut() already builds `pattern` as "{prefix}_*{tail}" where tail is
        # anchored at the end (e.g. "forward.npy" or "forward_ds32.npy") — forward that
        # tail unchanged and just insert the layer/pooling filter after the prefix, so
        # this stays correct regardless of whether a downstream-k suffix is present.
        prefix = "ref_seq" if pattern.startswith("ref_seq") else "mut_seq"
        tail = pattern[len(prefix) + 1:].lstrip("*")
        filtered_pattern = f"{prefix}_*L{self._layer}*{self._pooling}*{tail}"
        return self._path.rglob(filtered_pattern)

    def __str__(self):
        return str(self._path)

    def __fspath__(self):
        return str(self._path)


if __name__ == "__main__":
    main()
