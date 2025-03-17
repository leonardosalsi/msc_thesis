#!/bin/bash

#SBATCH --job-name=bm_rust
#SBATCH --output=/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/out/bm_rust.txt
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=80G

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/create_logan_test.py --use_rust