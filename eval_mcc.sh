#!/bin/bash

#SBATCH --job-name=evaluate-mcc
#SBATCH --output=out-mcc.txt
#SBATCH --cpus-per-task=30
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH -p gpu
#SBATCH --gres=gpu:0

source ~/.bashrc
enable_modules
module load python scipy-stack gcc arrow
module list

VENV=/cluster/home/salsil/venv
source $VENV/bin/activate

python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/eval_mcc.py