#!/bin/bash

#SBATCH --job-name=gen_logan_1k
#SBATCH --output=/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/out/scratch_gen_logan_1k.txt
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=12G

source ~/.bashrc
conda activate gpu_env

export SCRATCH_FASTA_DIR="$TMPDIR/logan_data"
mkdir -p "$SCRATCH_FASTA_DIR"

cp /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/logan/data/*.fa.zst "$SCRATCH_FASTA_DIR/"
ls "$SCRATCH_FASTA_DIR"

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/create_logan_dataset.py \
"$SCRATCH_FASTA_DIR" \
--metadata_file_path /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/logan/data/metadata.csv \
--chunk_size 1200 \
--reverse_complement --max_workers 8 --acc_column acc --group_id_column kmeans