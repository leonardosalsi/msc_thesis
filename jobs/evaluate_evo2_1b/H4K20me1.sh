#!/bin/bash

#SBATCH --job-name=evo2_1b_base_H4K20me1
#SBATCH --output=/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/out/evo2_1b_base_H4K20me1.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1

source ~/.bashrc
conda activate msc_thesis

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/evaluate_evo2.py evo2_1b_base 18 --samples 10