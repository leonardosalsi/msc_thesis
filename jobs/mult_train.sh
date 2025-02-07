#!/bin/bash

#SBATCH --job-name=train-overlap-1_2kbp
#SBATCH --output=out/train-overlap-1_2kbp.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=40G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:4

source ~/.bashrc
conda activate gpu_env

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29500 /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/train_model.py \
    multi_genome_dataset OverlappingEsmTokenizerWithNSkipping 1200 --shannon 1.35 1.8 --gc 0.4 0.6