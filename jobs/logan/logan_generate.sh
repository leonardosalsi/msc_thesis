#!/bin/bash

#SBATCH --job-name=generate_logan_dataset
#SBATCH --output=/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/out/generate_logan_dataset.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=40G

source ~/.bashrc
conda activate gpu_env

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/create_logan_dataset.py