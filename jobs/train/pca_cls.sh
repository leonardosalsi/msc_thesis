#!/bin/bash

#SBATCH --job-name=256_cls
#SBATCH --output=/cluster/home/salsil/msc_thesis_root/out/256_cls.txt
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
--compile-model --logging-steps 500 --train-size 5 --eval-size 32 \
--gradient-accumulation 20 --max-workers 4 --pca-embeddings cls --pca-dim 256