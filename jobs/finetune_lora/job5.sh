#!/bin/bash

#SBATCH --job-name=500m-multi-species-lora
#SBATCH --output=out/finetune_lora/out-nucleotide-transformer-v2-500m-multi-species.txt
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=80G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env

python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/mcc_finetune_lora.py InstaDeepAI/nucleotide-transformer-v2-500m-multi-species