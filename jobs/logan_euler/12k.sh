#!/bin/bash

#SBATCH --job-name=gen_logan_12k
#SBATCH --output=/cluster/home/salsil/msc_thesis_root/out/scratch_gen_logan_12k.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=128G

source ~/.bashrc
source $HOME/gpu_env/bin/activate

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python /cluster/home/salsil/msc_thesis_root/msc_thesis/create_logan_dataset.py \
--output-path /cluster/scratch/salsil/generated_datasets/logan_12200 \
--fasta-files-path /cluster/scratch/salsil/logan \
--metadata-file-path /cluster/scratch/salsil/logan/metadata.csv \
--chunk-size 12200 --reverse-complement --max-workers 4 --acc-column acc --group-id-column kmeans --identity_threshold 0.85