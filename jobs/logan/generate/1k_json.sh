#!/bin/bash

#SBATCH --job-name=gen_logan_1k
#SBATCH --output=/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/out/scratch_gen_logan_1k.txt
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=128G
#SBATCH --tmp=800g

source ~/.bashrc
conda activate gpu_env

cp /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/logan/data/*.fa.zst "$TMPDIR/"

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/create_logan_dataset.py \
"$TMPDIR" \
--metadata_file_path /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/logan/data/metadata.csv \
--chunk_size 1200 \
--reverse_complement --max_workers 8 --acc_column acc --group_id_column kmeans --use_scratch --use_json

export JSONDIR = "$(date +'%Y%m%d_%H%M%S')"
mkdir "/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/generated_datasets/$JSONDIR"
cp -r "$TMPDIR/*.json" "/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/generated_datasets/$JSONDIR/"
