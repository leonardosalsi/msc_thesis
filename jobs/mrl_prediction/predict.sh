#!/bin/bash

#SBATCH --output=/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/out/%x.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env
which python

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/mrl_prediction.py \
--model-name $MODEL --checkpoint 12000  $ADDITIONAL