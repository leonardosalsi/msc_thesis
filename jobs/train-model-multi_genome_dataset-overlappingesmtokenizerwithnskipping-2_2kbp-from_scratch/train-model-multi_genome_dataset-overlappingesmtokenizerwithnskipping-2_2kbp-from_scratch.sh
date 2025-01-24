#!/bin/bash

#SBATCH --job-name=train-model-multi_genome_dataset-overlappingesmtokenizerwithnskipping-2_2kbp-from_scratch
#SBATCH --output=out/train-model-multi_genome_dataset-overlappingesmtokenizerwithnskipping-2_2kbp-from_scratch.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=40G
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx2080ti:4

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python -m torch.distributed.run --nproc_per_node=4 /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/train_model.py multi_genome_dataset OverlappingEsmTokenizerWithNSkipping 2200 --from_scratch