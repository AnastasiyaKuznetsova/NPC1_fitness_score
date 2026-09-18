#!/bin/bash
#SBATCH --job-name=extract_emb_ntv2
#SBATCH --output=logs/extract_emb_ntv2_%j.out
#SBATCH --error=logs/extract_emb_ntv2_%j.err
#SBATCH --partition=gpu
#SBATCH --reservation=balch
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00

# Runs extract_embeddings_NTv2.py across all four NT-v2 model sizes.
#
# 50M/100M/250M share a 1,000-token max sequence length (~5,994bp at 6bp/token,
# leaving 1 token for the CLS special token), so they run on 5994bp windows.
# 500M was pretrained with a 2,048-token max (~12,282bp), so it can take a
# longer window -- this runs it on 8190bp windows.

set -uoe pipefail

# Anchor to the submission directory explicitly — don't rely on ambient $PWD,
# since that's what DATASET_DIR_* and LOGDIR are resolved relative to.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

SIF="sif/ntv2.sif"
LOGDIR="logs/logs_extract_emb_ntv2"

mkdir -p logs "$LOGDIR"

# Keep BLAS/OpenMP thread pools from oversubscribing against the CPUs SLURM gave us
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Folders written by prepare_dataset.py (output/<YYYYMMDD_HHMMSS>/) -- one for
# each window size used below. Set these to your actual prepare_dataset.py
# output directories before submitting, e.g.:
#   python prepare_dataset.py --xlsx data/NPC1_mut_fitness_scores.xlsx --window 5994
#   python prepare_dataset.py --xlsx data/NPC1_mut_fitness_scores.xlsx --window 8190
DATASET_DIR_5994="output/20260917_182136"
DATASET_DIR_8190="output/20260917_173539"

STRAND="forward"
EMB_TYPE="average"

# model_size:window_bp:dataset_dir
RUNS=(
    "50m:5994:${DATASET_DIR_5994}"
    "100m:5994:${DATASET_DIR_5994}"
    "250m:5994:${DATASET_DIR_5994}"
    "500m:8190:${DATASET_DIR_8190}"
)

for run in "${RUNS[@]}"; do
    IFS=":" read -r size window dataset_dir <<< "$run"
    model="InstaDeepAI/nucleotide-transformer-v2-${size}-multi-species"

    if [ ! -d "$dataset_dir" ]; then
        echo "ERROR: dataset dir '$dataset_dir' does not exist. Set DATASET_DIR_${window} to a" \
             "prepare_dataset.py output folder (output/<timestamp>/) with ${window}bp windows" \
             "before submitting." >&2
        exit 1
    fi

    mapfile -t ref_matches < <(find "$dataset_dir" -maxdepth 1 -name "ref_seq_DNA_${STRAND}_${window}bp.npy")
    mapfile -t mut_matches < <(find "$dataset_dir" -maxdepth 1 -name "mut_seq_DNA_${STRAND}_${window}bp.npy")

    if [ "${#ref_matches[@]}" -ne 1 ] || [ "${#mut_matches[@]}" -ne 1 ]; then
        echo "ERROR: expected exactly one ref_seq_DNA_${STRAND}_${window}bp.npy and one" \
             "mut_seq_DNA_${STRAND}_${window}bp.npy in $dataset_dir, found" \
             "${#ref_matches[@]} and ${#mut_matches[@]}." >&2
        exit 1
    fi
    ref_file="${ref_matches[0]}"
    mut_file="${mut_matches[0]}"

    logfile="${LOGDIR}/NTv2_${size}_${window}bp.log"
    echo "Running: model=${size} window=${window}bp"

    apptainer exec --nv --bind "$PWD":"$PWD" --pwd "$PWD" "$SIF" python3 extract_embeddings_NTv2.py \
        --model "$model" \
        --emb-type "$EMB_TYPE" \
        --ref-file "$ref_file" \
        --mut-file "$mut_file" \
        > "$logfile" 2>&1

    if [ $? -ne 0 ]; then
        echo "  FAILED (see $logfile)"
    else
        echo "  done"
    fi
done
