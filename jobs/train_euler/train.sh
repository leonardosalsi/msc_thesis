#!/bin/bash

#SBATCH --job-name=logan_1k_upload
#SBATCH --output=/cluster/home/salsil/msc_thesis_root/out/logan_1k_upload.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --mem-per-cpu=64G

source ~/.bashrc
source $HOME/gpu_env/bin/activate

export WANDB_DISABLED=true
export TF_CPP_MIN_LOG_LEVEL=2

python /cluster/home/salsil/msc_thesis_root/msc_thesis/combine_logan.py