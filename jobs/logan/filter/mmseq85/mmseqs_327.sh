#!/bin/bash

#SBATCH --job-name=mmseqs_327
#SBATCH --output=/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/out/mmseqs_327.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=128G
#SBATCH --tmp=150G

source ~/.bashrc
conda activate fasta_env

python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/filter.py \
/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/generated_datasets/logan_6200/random_walk/random_walk_327.json \
/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/generated_datasets/logan_6200/filtered \
--split_memory_limit 500 --min_seq_id 0.85