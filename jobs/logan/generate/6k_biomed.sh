#!/bin/bash

#SBATCH --job-name=gen_logan_6k
#SBATCH --output=/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis//out/scratch_gen_logan_6k.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=128G

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/create_logan_dataset.py \
/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/logan/data \
--metadata_file_path /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/logan/data/metadata.csv \
--chunk_size 6200 \
--reverse_complement --max_workers 4 --acc_column acc --group_id_column kmeans --use_json

