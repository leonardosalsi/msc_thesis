#!/bin/bash

#SBATCH --job-name=overlap_mean
#SBATCH --output=/cluster/home/salsil/msc_thesis_root/out/overlap_mean.txt
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
--dataset /cluster/scratch/salsil/datasets/logan_1200 --tokenizer overlapping \
--logging-steps 500 --train-size 5 --eval-size 16 \
--gradient-accumulation 20 --max-workers 4 --load-from-json --ewc-lambda 5 \
--original-dataset InstaDeepAI/multi_species_genomes --model-name overlap_logan_ewc_25_pca_mean_256 \
--pca-dim 256 --pca-embeddings mean