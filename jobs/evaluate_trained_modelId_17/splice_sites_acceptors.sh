#!/bin/bash

#SBATCH --job-name=18_splice_sites_acceptors
#SBATCH --output=/cluster/home/salsil/msc_thesis_root/out/18_splice_sites_acceptors.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/home/salsil/msc_thesis_root/msc_thesis/evaluate_model_trained.py 18 7 --samples 3 --pca_embeddings