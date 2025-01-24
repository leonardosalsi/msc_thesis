#!/bin/bash

#SBATCH --job-name=train-model-multi_genome_dataset-overlappingesmtokenizerwithnskipping-1_2kbp
#SBATCH --output=out/train-model-multi_genome_dataset-overlappingesmtokenizerwithnskipping-1_2kbp.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=40G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:4

module load cudacore/.11.8.0 StdEnv/2023

source ~/.bashrc
conda activate gpu_env

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Launch the training script with torchrun for multi-GPU DDP
torchrun --nproc_per_node=4 /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/train_model.py \
    multi_genome_dataset OverlappingEsmTokenizerWithNSkipping 1200