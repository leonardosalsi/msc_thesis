#!/bin/bash

#SBATCH --job-name=evaluate-mcc
#SBATCH --output=out-mcc.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=40G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env

python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/mcc.py