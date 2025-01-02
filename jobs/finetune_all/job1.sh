#!/bin/bash

#SBATCH --job-name=50m-multi-species-all
#SBATCH --output=out/finetune_all/out-nucleotide-transformer-v2-50m-multi-species.txt
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=80G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env

python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/mcc_finetune_all.py InstaDeepAI/nucleotide-transformer-v2-50m-multi-species