#!/bin/bash
#SBATCH --job-name=run_all_caduceus
#SBATCH --output=logs/run_all_caduceus_%j.out
#SBATCH --error=logs/run_all_caduceus_%j.err
#SBATCH --reservation=balch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=05-00:00:00

set -euo pipefail

cd "${$SLURM_SUBMIT_DIR:-$PWD}"

[ -f "output/df_preprocessed.csv" ] || {echo "Missing output/df_preprocessed.csv under $PWD}" >&2; exit 1; }

mapfile -t EMB_DIRS < <(find embeddings -maxdepth 1 -type d -name "Caduceus_*_emb" | sort)
[ "${#EMB_DIRS[@]}" -gt 0 ] || {echo "No embeddings Caduceus_*_emb were found under $PWD" >&2; exit 1; }
echo "Found ${#EMB_DIRS[@]} Caduceus embedding dir(s): ${EMB_DIRS[*]}"

for emb_dir in "${EMB_DIRS[@]}"; do
    label="$(basename "$emb_dir")"
    echo "=== Running run_all.py on $emb_dir ==="
    apptainer exec \
        --bind "$PWD":"$PWD" \
        --pwd "$PWD" \
        "sif/run_all.sif" \
        python run_all.py \
            --emb "$emb_dir" \
            --out_dir "results/${label}" \
            --model_dir "saved_models/${label}" \
            --n_jobs "$SLURM_CPUS_PER_TASK"
done
