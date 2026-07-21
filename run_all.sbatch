#!/bin/bash
#SBATCH --job-name=run_all
#SBATCH --output=logs/run_all_%j.out
#SBATCH --error=logs/run_all_%j.err
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH  --cpus-per-task=16
#SBATCH  --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

# Anchor to the submission directory explicitly — don't rely on ambient $PWD,
# since that's what "output/df_preprocessed.csv" and --emb are resolved relative to.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Fail fast with a clear message instead of every layer/pooling combo silently
# logging "FAILED loading ..." and leaving metrics.csv empty.
[ -f "output/df_preprocessed.csv" ] || { echo "Missing output/df_preprocessed.csv under $PWD" >&2; exit 1; }
[ -d "emb_7B" ] || { echo "Missing emb_7B/ under $PWD" >&2; exit 1; }

# Keep BLAS/OpenMP thread pools from oversubscribing against the CPUs SLURM gave us
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
apptainer run \
    --bind "$PWD":"$PWD" \
    --pwd "$PWD" \
    "run_all.sif" \
    run_all.py \
        --emb "emb_7B" \
        --out_dir "results" \
        --model_dir "saved_models"