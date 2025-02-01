#!/bin/bash

#SBATCH --job-name=filter-downsize-chunks-multi_genome_dataset-train-1_2kbp-from_scratch
#SBATCH --output=out/filter-downsize-chunks-multi_genome_dataset-train-1_2kbp-from_scratch.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/filter_and_downsize_dataset.py multi_genome_dataset train 2200 --shannon 1.35 1.8 --gc 0.4 0.6