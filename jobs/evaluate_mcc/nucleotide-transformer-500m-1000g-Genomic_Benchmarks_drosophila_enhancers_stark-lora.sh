#!/bin/bash

#SBATCH --job-name=nucleotide-transformer-500m-1000g-Genomic_Benchmarks_drosophila_enhancers_stark-lora
#SBATCH --output=out/nucleotide-transformer-500m-1000g-Genomic_Benchmarks_drosophila_enhancers_stark-lora.txt
#SBATCH --cpus-per-task=6
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/evaluate_model_mcc.py 5 22 