#!/bin/bash

#SBATCH --job-name=gen_logan_2k
#SBATCH --output=/cluster/home/salsil/msc_thesis_root/out/scratch_gen_logan_2k.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=128G
#SBATCH --tmp=800g

source ~/.bashrc
source $HOME/gpu_env/bin/activate

cp /cluster/scratch/salsil/logan/*.fa.zst "$TMPDIR/"

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/home/salsil/msc_thesis_root/msc_thesis/create_logan_dataset.py \
"$TMPDIR" \
--metadata_file_path /cluster/scratch/salsil/logan/metadata.csv \
--chunk_size 2200 \
--reverse_complement --max_workers 4 --acc_column acc --group_id_column kmeans --use_scratch --use_json

export JSONDIR="$(date +'%Y%m%d_%H%M%S')"
mkdir "/cluster/scratch/salsil/$JSONDIR"
cp -r "$TMPDIR"/*.json "/cluster/scratch/salsil/$JSONDIR/"
