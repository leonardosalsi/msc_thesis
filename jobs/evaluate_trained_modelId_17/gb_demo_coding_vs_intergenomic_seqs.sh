#!/bin/bash

#SBATCH --job-name=18_gb_demo_coding_vs_intergenomic_seqs
#SBATCH --output=/cluster/home/salsil/msc_thesis_root/out/18_gb_demo_coding_vs_intergenomic_seqs.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/home/salsil/msc_thesis_root/msc_thesis/evaluate_model_trained.py 18 24 --samples 3 --pca_embeddings