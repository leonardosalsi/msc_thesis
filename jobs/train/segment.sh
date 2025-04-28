#!/bin/bash

#SBATCH --job-name=segment_logan
#SBATCH --output=/cluster/home/salsil/msc_thesis_root/out/segment_logan.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus=rtx_4090:1

source ~/.bashrc
source $HOME/gpu_env/bin/activate

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_DISABLED=true
export TF_CPP_MIN_LOG_LEVEL=2

python  /cluster/home/salsil/msc_thesis_root/msc_thesis/segment_nt.py \
--dataset /cluster/scratch/salsil/logan_json/logan_json --load-from-json