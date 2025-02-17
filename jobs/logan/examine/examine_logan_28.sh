#!/bin/bash

#SBATCH --job-name=eval_logan_28
#SBATCH --output=/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/out/eval_logan_28.txt
#SBATCH --cpus-per-task=5
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=80G

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/examine_logan.py 28