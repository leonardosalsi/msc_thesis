#!/bin/bash

#SBATCH --job-name=mmseqs_89
#SBATCH --output=/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/out/mmseqs_89.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=16G

source ~/.bashrc
conda activate fasta_env

python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/filter.py \
/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/generated_datasets/logan_raw/random_walk_89.json \
/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/generated_datasets/logan_filtered_75 \
--split_memory_limit 500 --min_seq_id 0.75