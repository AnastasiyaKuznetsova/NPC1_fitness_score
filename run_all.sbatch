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