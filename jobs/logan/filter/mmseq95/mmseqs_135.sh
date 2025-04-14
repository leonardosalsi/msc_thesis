#!/bin/bash

#SBATCH --job-name=mmseqs_135
#SBATCH --output=/cluster/home/salsil/msc_thesis_root/out/mmseqs_135.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=256G

source ~/.bashrc
source $HOME/gpu_env/bin/activate

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_DISABLED=true
export TF_CPP_MIN_LOG_LEVEL=2

python /cluster/home/salsil/msc_thesis_root/msc_thesis/filter.py \
/cluster/scratch/salsil/logan_raw/random_walk_135.json \
/cluster/scratch/salsil/logan_filtered_95 \
--split_memory_limit 800 --min_seq_id 0.95