#!/bin/bash

#SBATCH --job-name=train_overlap_sh_gc
#SBATCH --output=/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/out/train_overlap_sh_gc.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --mem-per-cpu=40G
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1

source ~/.bashrc
conda activate gpu_env

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/train_model.py \
    multi_genome_dataset OverlappingEsmTokenizerWithNSkipping --chunk_size 1200 --shannon 1.35 1.8 --gc 0.4 0.6 --freeze 0.5