#!/bin/bash

#SBATCH --job-name=logan_no_ewc
#SBATCH --output=/cluster/home/salsil/msc_thesis_root/out/logan_no_ewd.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
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
--dataset /cluster/scratch/salsil/datasets/logan_6200 --tokenizer default \
--logging-steps 100000 --save-steps 100000 --train-size 32 --eval-size 32 --max-steps 600000 \
--gradient-accumulation 1 --max-workers 4 --load-from-json --ewc-lambda 2 \
--original-dataset InstaDeepAI/multi_species_genomes --gradient-checkpointing