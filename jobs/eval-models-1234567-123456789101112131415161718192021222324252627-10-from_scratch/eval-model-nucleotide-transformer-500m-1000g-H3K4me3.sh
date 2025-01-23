#!/bin/bash

#SBATCH --job-name=eval-model-nucleotide-transformer-500m-1000g-H3K4me3
#SBATCH --output=out/eval-model-nucleotide-transformer-500m-1000g-H3K4me3.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/evaluate_model.py 5 15  --samples 10