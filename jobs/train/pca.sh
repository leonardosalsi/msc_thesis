#!/bin/bash

#SBATCH --job-name=train_overlap
#SBATCH --output=/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/out/train_overlap.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --mem-per-cpu=128G
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1

source ~/.bashrc
conda activate gpu_env

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/train_model.py \
    /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/generated_datasets/multi_genome_dataset/1k \
    overlapping --compile_model --pca_embeddings --keep_in_memory