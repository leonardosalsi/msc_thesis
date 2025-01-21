import os

from downstream_tasks import TASKS, MODELS

folder = "../jobs/evaluate_mcc"
tasks = TASKS
models = MODELS

model = models[-1]

print(model)

shellfiles = []

for task in tasks:
    content = f"""#!/bin/bash
    
#SBATCH --job-name={model['name']}-{task['alias']}-lora
#SBATCH --output=out/{model['name']}-{task['alias']}-lora.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \\
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/evaluate_model_mcc.py {model['modelId']} {task['taskId']}"""

    with open(os.path.join(folder, f"{model['name']}-{task['alias']}-lora.sh"), "w") as file:
        file.write(content)
    shellfiles.append(os.path.join(f"{model['name']}-{task['alias']}-lora.sh"))

shellcontent = ""
for shellfile in shellfiles:
    shellcontent += f"sbatch jobs/evaluate_mcc/{shellfile}\n"

shell = f"""#!/bin/bash

{shellcontent}"""

with open("../evaluate_it01.sh", "w") as file:
    file.write(shell)
