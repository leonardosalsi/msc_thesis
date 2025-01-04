
    #!/bin/bash

    #SBATCH --job-name=nucleotide-transformer-500m-1000g-H4K20me1
    #SBATCH --output=out/nucleotide-transformer-500m-1000g-H4K20me1.txt
    #SBATCH --cpus-per-task=4
    #SBATCH --time=20:00:00
    #SBATCH --mem-per-cpu=8G
    #SBATCH -p gpu
    #SBATCH --gres=gpu:1

    source ~/.bashrc
    conda activate gpu_env

    HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/evaluate_model_mcc.py 5 18 --no-lora
    