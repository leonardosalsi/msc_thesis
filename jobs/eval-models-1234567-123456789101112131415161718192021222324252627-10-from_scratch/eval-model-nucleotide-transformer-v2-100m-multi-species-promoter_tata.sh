#!/bin/bash

#SBATCH --job-name=eval-model-nucleotide-transformer-v2-100m-multi-species-promoter_tata
#SBATCH --output=out/eval-model-nucleotide-transformer-v2-100m-multi-species-promoter_tata.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/evaluate_model.py 2 2  --samples 10