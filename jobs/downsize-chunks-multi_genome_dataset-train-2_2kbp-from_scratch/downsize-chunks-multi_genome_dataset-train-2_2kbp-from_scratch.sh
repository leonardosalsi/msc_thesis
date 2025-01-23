#!/bin/bash

#SBATCH --job-name=downsize-chunks-multi_genome_dataset-train-2_2kbp-from_scratch
#SBATCH --output=out/downsize-chunks-multi_genome_dataset-train-2_2kbp-from_scratch.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/downsize_dataset_chunks.py multi_genome_dataset train 2200