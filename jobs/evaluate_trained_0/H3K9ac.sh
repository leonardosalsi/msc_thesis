#!/bin/bash

#SBATCH --job-name=H3K9ac
#SBATCH --output=/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/out/H3K9ac.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/evaluate_model_trained.py \
--model-name overlap_logan_no_ewc --checkpoint 12000 --task-id 16 --samples 3