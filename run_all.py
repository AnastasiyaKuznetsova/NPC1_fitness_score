"""
Run all regressor combinations and save a single structured metrics CSV.

Iterates over all layers found in --emb, both delta/concat, LightGBM/Ridge.
Output CSV columns:
  layer, model_arch, emb_type, strand, lightweight_model, hyperparams,
  train_corr, train_mse, train_mae,
  val_corr,   val_mse,   val_mae,
  test_corr,  test_mse,  test_mae

Command-line options:
  --df         Path to the preprocessed dataframe CSV (default: output/df_preprocessed.csv).
  --emb        Directory with ref_seq_*/mut_seq_* .npy embedding files (required).
  --strand     'forward', 'reverse', or 'both' (concatenated fwd+rev) (default: forward).
  --emb_mode   'dna' sweeps every layer/pooling combo in --emb; 'rna' loads RNA
               embeddings directly with no layer/pooling sweep (default: dna).
  --out_dir    Directory for per-run logs, metrics.csv, and the summary print (default: results).
  --pca        Number of PCA components to apply before regression; omit to skip PCA.
  --model_dir  Directory to save each run's best model as a date-stamped .joblib
               file (default: saved_models).
"""

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from regressor import load_data_dna, load_data_rna, nested_cv, build_pipeline, MODELS


def get_hyperparams(model_name: str, pca_components: int = None) -> str:
    if model_name == "Dummy":
        return "strategy=mean"
    pca_str = f"pca={pca_components}, " if pca_components else ""
    pipe = build_pipeline(model_name=model_name, pca_components=pca_components)
    reg = pipe.named_steps["reg"]
    params = reg.get_params()
    key_params = {
        "LightGBM":       ["n_estimators", "max_depth", "num_leaves", "learning_rate"],
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
    return f"{pca_str}{param_str}" if param_str else pca_str or "default"


def extract_layer_pooling_combos(emb_dir: Path, strand: str) -> list[tuple[str, str]]:
    """Return sorted unique (layer_index, pooling_mode) pairs from refs_*{strand}*.npy filenames."""
    files = sorted(emb_dir.rglob(f"ref_seq_*{strand}*.npy"))
    combos = []
    for f in files:
        layer_m = re.search(r"_L(\d+)_", f.name)
        pool_m  = re.search(r"_L\d+_(average|last)_", f.name)
        if layer_m and pool_m:
            combos.append((layer_m.group(1), pool_m.group(1)))
    combos = sorted(set(combos))
    if not combos:
        raise FileNotFoundError(
            f"No ref_seq_*{strand}*.npy files with _L{{n}}_(average|last)_ pattern in {emb_dir}"
        )
    return combos


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


def run_combo(emb, df, model_name, tag, pca_components=None, model_dir=None, run_timestamp=None) -> dict:
    groups = df["Protein Annotation"].to_numpy()
    y = df["Function Score"].to_numpy()
    logging.info(f"Grouping by: 'Protein Annotation' ({len(np.unique(groups))} unique groups)")
    fold_metrics, best_model = nested_cv(emb, y, groups, model_name=model_name,
                                         outer_splits=5, inner_splits=3,
                                         pca_components=pca_components)

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
    parser.add_argument("--emb_mode", choices=["rna", "dna"], default="dna",
                        help="Embedding source type: 'rna' loads RNA embeddings via load_data_rna "
                             "(no layer/pooling sweep), 'dna' sweeps every layer/pooling combo found "
                             "in --emb via load_data_dna. Default: dna.")
    parser.add_argument("--out_dir", default="results",
                        help="Directory to write per-run log files, the summary "
                             "metrics.csv, and the printed run summary. Default: results.")
    parser.add_argument("--pca", type=int, default=None,
                        help="Number of PCA components before regression. "
                             "Omit to skip PCA (default: no PCA).")
    parser.add_argument("--model_dir", default="saved_models",
                        help="Directory to save the best model per run as a .joblib file. "
                             "Default: saved_models.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = Path(args.emb)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    non_dummy_models = [m for m in MODELS if m not in ("Dummy", "LightGBM")]
    model_configs = [
        {"model_name": model_name, "delta": delta, "emb_type": "delta" if delta else "concat"}
        for model_name in non_dummy_models
        for delta in [True, False]
    ] + [{"model_name": "Dummy", "delta": True, "emb_type": "none"}]

    strand_configs = {
        "forward":  [("forward",  False, "forward")],
        "reverse":  [("reverse",  False, "reverse")],
        "both":     [("fwd+rev",  True,  "forward")],
    }[args.strand]
    # strand_configs entries: (label, use_reverse, ref_strand_for_layer_discovery)

    all_results = []

    if args.emb_mode == "rna":
        df_data, emb_data = load_data_rna(args.df, args.emb)
        for c in model_configs:
            run_name = f"RNA_{c['model_name']}"
            setup_run_logging(out_dir, run_name)
            logging.info(f"Run: {run_name}")
            metrics = run_combo(emb_data, df_data, model_name=c["model_name"], tag=run_name)
            all_results.append({
                "layer": "none", "model_arch": "RNA", "emb_type": "none",
                "strand": "none", "model": c["model_name"],
                "hyperparams": get_hyperparams(c["model_name"]),
                **metrics,
            })

    else:  # dna
        for strand_label, use_reverse, ref_strand in strand_configs:
            layer_pool_combos = extract_layer_pooling_combos(emb_dir, ref_strand)
            for layer, pooling in layer_pool_combos:
                group_name = f"L{layer}_{pooling}_{strand_label}"
                setup_run_logging(out_dir, group_name)

                # Load embeddings once per layer/pooling/strand — shared across model combos
                layer_emb = _LayerFilteredDir(emb_dir, layer, pooling)
                try:
                    df_fwd, emb_fwd = load_data_dna(
                        args.df, layer_emb, delta=False, use_reverse=use_reverse, strand=ref_strand,
                    )
                    _, emb_delta = load_data_dna(
                        args.df, layer_emb, delta=True, use_reverse=use_reverse, strand=ref_strand,
                    )
                except Exception as e:
                    logging.error(f"FAILED loading {group_name}: {e}")
                    continue

                sample_files = sorted(emb_dir.rglob(f"ref_seq_*{ref_strand}*L{layer}*{pooling}*.npy"))
                model_arch = "Evo2"
                if sample_files:
                    m = re.search(r"_(Evo2)_(\w+)_", sample_files[0].name)
                    if m:
                        model_arch = f"{m.group(1)}_{m.group(2)}"

                for c in model_configs:
                    run_name = f"{group_name}_{c['model_name']}_{c['emb_type']}"
                    logging.info(f"Run: {run_name}")
                    emb_data = emb_delta if c["delta"] else emb_fwd

                    try:
                        metrics = run_combo(
                            emb_data, df_fwd, model_name=c["model_name"],
                            tag=run_name, pca_components=args.pca,
                            model_dir=args.model_dir, run_timestamp=run_timestamp,
                        )
                    except Exception as e:
                        logging.error(f"FAILED {run_name}: {e}")
                        continue

                    all_results.append({
                        "layer":      f"L{layer}",
                        "model_arch": model_arch,
                        "pooling":    pooling,
                        "emb_type":   c["emb_type"],
                        "strand":     strand_label,
                        "model":      c["model_name"],
                        "hyperparams": get_hyperparams(c["model_name"], pca_components=args.pca),
                        **metrics,
                    })

    metrics_df = pd.DataFrame(all_results)
    csv_path = out_dir / "metrics.csv"
    metrics_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    cols = ["layer", "model_arch", "pooling", "emb_type", "strand", "model",
            "train_corr", "train_corr_std", "val_corr", "val_corr_std",
            "test_corr", "test_corr_std", "test_mse", "test_mse_std", "test_mae", "test_mae_std"]
    summary_cols = [c for c in cols if c in metrics_df.columns]
    print(metrics_df[summary_cols].to_string(index=False))
    print(f"\nFull metrics saved to: {csv_path.resolve()}")


class _LayerFilteredDir:
    """Wraps an emb directory and restricts glob results to a specific layer and pooling mode."""
    def __init__(self, path: Path, layer: str, pooling: str):
        self._path = path
        self._layer = layer
        self._pooling = pooling

    def glob(self, pattern: str):
        # Filename format: {ref_seq|mut_seq}_*_L{layer}_{pooling}_{strand}.npy
        # Extract prefix (ref_seq / mut_seq) and strand from caller's pattern.
        prefix = "ref_seq" if pattern.startswith("ref_seq") else "mut_seq"
        m = re.search(r"\*(forward|reverse)\*", pattern)
        strand = m.group(1) if m else "*"
        filtered_pattern = f"{prefix}_*L{self._layer}*{self._pooling}*{strand}*.npy"
        return self._path.rglob(filtered_pattern)

    def __str__(self):
        return str(self._path)

    def __fspath__(self):
        return str(self._path)


if __name__ == "__main__":
    main()
