#!/bin/bash

#SBATCH --output=/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/out/tsne.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=8G

source ~/.bashrc
conda activate gpu_env
which python

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/get_tsne.py \
--embeddings-path $MODEL