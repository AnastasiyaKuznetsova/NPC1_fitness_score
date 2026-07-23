#!/bin/bash
#SBATCH --job-name=regressor
#SBATCH --array=0-10
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/model_%a_%j.out
#SBATCH --error=logs/model_%a_%j.err

MODELS=(Ridge Lasso ElasticNet KernelRidge SVR PLS GaussianProcess kNN RandomForest DecisionTree Dummy)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Running model: $MODEL (task $SLURM_ARRAY_TASK_ID)"

apptainer run --bind $PWD:$PWD --pwd $PWD run_all.sif \
    python regressor.py \
        --emb emb_7B/ \
        --delta --strand forward \
        --models $MODEL \
        --model_dir saved_models/ \
        --log_dir logs/