#!/bin/bash

#SBATCH --job-name=mmseqs_497
#SBATCH --output=/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/out/mmseqs_497.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --tmp=50G

source ~/.bashrc
conda activate fasta_env

python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/filter.py \
/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/generated_datasets/logan_raw/random_walk_497.json \
/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/generated_datasets/logan_filtered \
--split_memory_limit 500 --use_scratch  --min_seq_id 0.95