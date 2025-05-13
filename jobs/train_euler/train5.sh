#!/bin/bash

#SBATCH --job-name=logan_ewc_25
#SBATCH --output=/cluster/home/salsil/msc_thesis_root/out/logan_ewc_25.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_4090:5

source ~/.bashrc
source $HOME/gpu_env/bin/activate

export WANDB_DISABLED=true
export TF_CPP_MIN_LOG_LEVEL=2
export HF_DATASETS_CACHE=/cluster/scratch/salsil/hf_cache

accelerate launch --num_processes 5 --num_machines 1 --mixed_precision no --dynamo_backend inductor /cluster/home/salsil/msc_thesis_root/msc_thesis/train_model.py \
--dataset /cluster/scratch/salsil/datasets/logan_6200 --tokenizer default \
--logging-steps 500 --train-size 10 --eval-size 32 \
--gradient-accumulation 10 --max-workers 4 --load-from-json --ewc-lambda 25 \
--original-dataset InstaDeepAI/multi_species_genomes