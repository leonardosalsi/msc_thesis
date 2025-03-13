#!/bin/bash

#SBATCH --job-name=gen_logan_2k
#SBATCH --output=/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/out/gen_logan_2k.txt
#SBATCH --cpus-per-task=16
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=100G

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/create_logan_dataset.py --chunk_size 2200 --reverse_complement