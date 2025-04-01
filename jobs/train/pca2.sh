#!/bin/bash

#SBATCH --job-name=train_overlap
#SBATCH --output=/cluster/home/salsil/msc_thesis_root/out/train_overlap.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus=rtx_4090:4

source ~/.bashrc
source $HOME/gpu_env/bin/activate

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

python /cluster/home/salsil/msc_thesis_root/msc_thesis/train_model.py    \
/cluster/scratch/salsil/multi_genome_species/1k  overlapping \
--compile_model --pca_embeddings  --train_size 20 --eval_size 128 \
--gradient_accumulation 25 --max_workers 4 --use_scratch --keep_in_memory