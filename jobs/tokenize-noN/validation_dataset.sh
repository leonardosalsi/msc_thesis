#!/bin/bash

#SBATCH --job-name=tokenize_1k_validation
#SBATCH --output=out/tokenize_1k_validation.txt
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/tokenize1k-noN.py validation