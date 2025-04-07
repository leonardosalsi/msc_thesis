#!/bin/bash

#SBATCH --job-name=filter
#SBATCH --output=/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/out/filter.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=80G
#SBATCH --tmp=2500g

source ~/.bashrc
conda activate fasta_env

cp /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/generated_datasets/logan_raw/*.json "$TMPDIR/"

python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/filter.py \
/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/generated_datasets/logan_raw \
/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/generated_datasets/logan_filtered \
--split_memory_limit 64 --n_smallest_files 10 --use_scratch  --min_seq_id 0.85
