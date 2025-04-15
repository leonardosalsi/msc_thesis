#!/bin/bash

#SBATCH --job-name=final2
#SBATCH --output=/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/out/final2.txt
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --tmp=500g

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/logan_final.py \
/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/generated_datasets/logan_separated/1k \
--use_scratch