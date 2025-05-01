#!/bin/bash

#SBATCH --job-name=train_overlap
#SBATCH --output=/cluster/home/salsil/msc_thesis_root/out/train_overlap.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus=rtx_4090:5

source ~/.bashrc
source $HOME/gpu_env/bin/activate

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_DISABLED=true
export TF_CPP_MIN_LOG_LEVEL=2

accelerate launch --num_processes 5 --num_machines 1 --mixed_precision no --dynamo_backend inductor /cluster/home/salsil/msc_thesis_root/msc_thesis/train_model.py \
--dataset /cluster/scratch/salsil/multi_genome_species/1k --tokenizer overlapping \
--compile_model --logging_steps 500 --train_size 10 --eval_size 32 \
--gradient_accumulation 10 --max_workers 4 --ewc_lambda 0.5 \
--original_dataset /cluster/scratch/salsil/multi_genome_species/1k