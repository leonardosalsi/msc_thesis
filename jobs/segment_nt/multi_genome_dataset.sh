#!/bin/bash

#SBATCH --job-name=segment_nt_multi_genome_dataset
#SBATCH --output=/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/out/segment_nt_multi_genome_dataset.txt
#SBATCH --cpus-per-task=6
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=80G
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/segment_nt.py multi_genome_dataset