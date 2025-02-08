#!/bin/bash

#SBATCH --job-name=eval-model-nucleotide-transformer-v2-50m-multi-species-promoter_all
#SBATCH --output=out/eval-model-nucleotide-transformer-v2-50m-multi-species-promoter_all.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/temp/msc_thesis/evaluate_model_trained.py 1 1 --samples 10