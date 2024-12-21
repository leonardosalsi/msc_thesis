#!/bin/bash

#SBATCH --job-name=evaluate-mcc
#SBATCH --output=out-mcc.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=2G

source ~/.bashrc
enable_modules
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/lib64:$LD_LIBRARY_PATH

module purge
module load gcc/9.3.0
module load python/3.10.2 scipy-stack/2023b arrow
module list

VENV=/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/venv
source $VENV/bin/activate

python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/eval_mcc.py