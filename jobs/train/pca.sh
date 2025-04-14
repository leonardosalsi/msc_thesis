#!/bin/bash

#SBATCH --job-name=train_overlap
#SBATCH --output=/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/out/train_overlap.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --mem-per-cpu=256G
#SBATCH -p gpu
#SBATCH #SBATCH --gpus=rtx_4090:5

source ~/.bashrc
conda activate gpu_env

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

accelerate launch --num_processes 5 /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/train_model.py \
/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/generated_datasets/multi_genome_dataset/1k \
overlapping --compile_model --compile_model  --train_size 10 --eval_size 32 \
--gradient_accumulation 10 --max_workers 4 --keep_in_memory