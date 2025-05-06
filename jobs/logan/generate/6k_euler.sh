#!/bin/bash

#SBATCH --job-name=gen_logan_6k
#SBATCH --output=/cluster/home/salsil/msc_thesis_root/out/scratch_gen_logan_6k.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=128G

source ~/.bashrc
source $HOME/gpu_env/bin/activate

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/home/salsil/msc_thesis_root/msc_thesis/create_logan_dataset.py \
/cluster/scratch/salsil/logan \
--metadata_file_path /cluster/scratch/salsil/logan/metadata.csv \
--chunk_size 2200 \
--reverse_complement --max_workers 4 --acc_column acc --group_id_column kmeans --use_json

export JSONDIR="random_walk_6200"
mkdir "/cluster/scratch/salsil/$JSONDIR"
cp -r "$TMPDIR"/*.json "/cluster/scratch/salsil/$JSONDIR/"
