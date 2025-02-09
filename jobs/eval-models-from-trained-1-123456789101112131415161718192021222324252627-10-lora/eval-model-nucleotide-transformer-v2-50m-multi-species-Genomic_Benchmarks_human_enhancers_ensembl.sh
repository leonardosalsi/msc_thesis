#!/bin/bash

#SBATCH --job-name=Genomic_Benchmarks_human_enhancers_ensembl
#SBATCH --output=out/Genomic_Benchmarks_human_enhancers_ensembl.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/evaluate_model_trained.py 1 25 --samples 10