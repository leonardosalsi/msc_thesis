#!/bin/bash

#SBATCH --job-name=multi_species_train_1k
#SBATCH --output=out/multi_species_train_1k.txt
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/generate1k.py train