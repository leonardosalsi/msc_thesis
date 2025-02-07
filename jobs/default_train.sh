#!/bin/bash

#SBATCH --job-name=train-no-overlap-1_2kbp
#SBATCH --output=out/train-no-overlap-1_2kbp.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=40G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1

source ~/.bashrc
conda activate gpu_env

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/train_model.py \
    multi_genome_dataset Default 1200 --shannon 1.35 1.8 --gc 0.4 0.6