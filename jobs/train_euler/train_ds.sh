#!/bin/bash

#SBATCH --job-name=2kb_ewc_5
#SBATCH --output=/cluster/home/salsil/msc_thesis_root/out/2kb_ewc_5.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=200:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_4090:8

module load stack/2024-06 gcc/12.2.0 cuda/12.1.1

source ~/.bashrc
source $HOME/gpu_env/bin/activate

export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export WANDB_DISABLED=true
export TF_CPP_MIN_LOG_LEVEL=2

deepspeed /cluster/home/salsil/msc_thesis_root/msc_thesis/train_model.py \
--dataset /cluster/scratch/salsil/datasets/logan_12200 --tokenizer default \
--logging-steps 10 --save-steps 6000 --train-size 4 --eval-size 16 --max-steps 12000 \
--gradient-accumulation 8 --max-workers 4 --load-from-json --ewc-lambda 5 \
--original-dataset InstaDeepAI/multi_species_genomes --gradient-checkpointing \
--deepspeed-config /cluster/home/salsil/msc_thesis_root/ds_config.json --num-tokens 2000
