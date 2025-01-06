#!/bin/bash

#SBATCH --job-name=nucleotide-transformer-v2-50m-multi-species-random-weights-H3K4me2-lora
#SBATCH --output=out/nucleotide-transformer-v2-50m-multi-species-random-weights-H3K4me2-lora.txt
#SBATCH --cpus-per-task=6
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/evaluate_model_mcc.py 1 14 --random-weights 