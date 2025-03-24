#!/bin/bash

#SBATCH --job-name=evo2_7b_base_gb_drosophila_enhancers_stark
#SBATCH --output=/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/out/evo2_7b_base_gb_drosophila_enhancers_stark.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/evaluate_evo2.py evo2_7b_base 22 --samples 10