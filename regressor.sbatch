#!/bin/bash

# Use this script to run each model on each node separately (n_arrays=0-10)
# Across nodes: different models run simultaneously (SLURM array)
# Within a node: inner CV folds run simultaneously (sklearn n_jobs=-1)

#SBATCH --job-name=regressor
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=logs/model_%a_%j.out
#SBATCH --error=logs/model_%a_%j.err


apptainer exec --bind $PWD:$PWD --pwd $PWD run_all.sif python regressor.py \
--emb emb_7B/ \
--delta --strand forward \
--models GaussianProcess \
--model_dir saved_models/ \
--log_dir logs/ \
--layer 27 \
--pooling average
