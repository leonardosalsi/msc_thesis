#!/bin/bash

#SBATCH --job-name=mt_gen_val_2_2_shgc
#SBATCH --output=/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/out/mt_gen_val_2_2_shgc.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/filter_and_downsize_dataset.py multi_genome_dataset validation 2200 --shannon 1.35 1.8 --gc 0.4 0.6