#!/bin/bash

#SBATCH --job-name=final
#SBATCH --output=/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/out/final.txt
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=32G

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/home/salsil/msc_thesis_root/msc_thesis/logan_final.py \
/cluster/scratch/salsil/logan_json/logan_json