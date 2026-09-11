#!/bin/bash
#SBATCH --job-name=run_all_janus
#SBATCH --output=logs/run_all_janus_%j.out
#SBATCH --error=logs/run_all_janus_%j.err
#SBATCH --reservation=balch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00

set -euo pipefail

# Anchor to the submission directory explicitly — don't rely on ambient $PWD,
# since that's what "output/df_preprocessed.csv" and the embeddings dirs are resolved relative to.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

[ -f "output/df_preprocessed.csv" ] || { echo "Missing output/df_preprocessed.csv under $PWD" >&2; exit 1; }

# Discover every JanusDNA embedding folder (one per label/context-window combo
# written by extract_embeddings_JanusDNA.py) instead of hardcoding names.
mapfile -t EMB_DIRS < <(find embeddings -maxdepth 1 -type d -name "JanusDNA_*_emb" | sort)
[ "${#EMB_DIRS[@]}" -gt 0 ] || { echo "No embeddings/JanusDNA_*_emb directories found under $PWD" >&2; exit 1; }
echo "Found ${#EMB_DIRS[@]} JanusDNA embedding dir(s): ${EMB_DIRS[*]}"

# Keep BLAS/OpenMP thread pools from oversubscribing against the CPUs SLURM gave us
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

for emb_dir in "${EMB_DIRS[@]}"; do
    label="$(basename "$emb_dir")"
    echo "=== Running run_all.py on $emb_dir ==="
    apptainer exec \
        --bind "$PWD":"$PWD" \
        --pwd "$PWD" \
        "sif/run_all.sif" \
        python run_all.py \
            --emb "$emb_dir" \
            --out_dir "results_${label}" \
            --model_dir "saved_models_${label}" \
            --n_jobs "$SLURM_CPUS_PER_TASK"
done
