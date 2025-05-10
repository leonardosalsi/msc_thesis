#!/bin/bash

#SBATCH --job-name=gen_logan_2k
#SBATCH --output=/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/out/scratch_gen_logan_2k.txt
#SBATCH --cpus-per-task=6
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=128G

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/create_logan_dataset.py \
--output-path /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/generated_datasets/logan_2200 \
--fasta-files-path /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/logan/data \
--metadata-file-path /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/logan/data/metadata.csv  \
--chunk-size 2200 --reverse-complement --max-workers 6 --acc-column acc --group-id-column kmeans --identity-threshold 0.95

