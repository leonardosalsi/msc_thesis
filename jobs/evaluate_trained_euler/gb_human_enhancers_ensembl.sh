#!/bin/bash

#SBATCH --output=/cluster/home/salsil/msc_thesis_root/out/%x.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1

source ~/.bashrc
conda activate gpu_env
which python

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/home/salsil/msc_thesis_root/msc_thesis/evaluate_model_trained.py \
--model-name $MODEL --checkpoint 12000 --task-id 25 $ADDITIONAL