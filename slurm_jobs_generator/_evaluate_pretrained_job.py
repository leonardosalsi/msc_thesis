import os

from config import out_dir
from downstream_tasks import MODELS, TASKS, PRETRAINED_MODELS
from util import get_chunk_size_file_name, get_model_by_id, get_task_by_id, get_pretrained_model_by_id


def ask_evaluation_trained(parent_folder):
    model_ids = [model['modelId'] for model in PRETRAINED_MODELS]
    model_names = [model['data_alias'] for model in PRETRAINED_MODELS]
    print("What models do you want to logan? [Multiple selections possible, separate by space. Type 0 for all.]")
    for i in range(len(model_ids)):
        print(f"[{model_ids[i]}]\t{model_names[i]}")
    while True:
        model_id = input("> ").lower()
        try:
            if not int(model_id) in model_ids:
                success = False
                print(f"{model_id} is not a valid model ID")
            else:
                success = True
        except ValueError:
            success = False
            print(f"{model_id} is not a valid model ID. Please only input numeric values.")
        if success:
            ask_task(parent_folder, model_id)
            break

def ask_task(parent_folder, model_id):
    task_ids = [task['taskId'] for task in TASKS]
    task_names = [task['data_alias'] for task in TASKS]
    print("On what tasks do you want to logan the models? [Multiple selections possible, separate by space. Type 0 for all.]")
    for i in range(len(task_ids)):
        print(f"[{task_ids[i]}]\t{task_names[i]}")
    while True:
        selected_task_ids = input("> ").lower()
        if selected_task_ids == "0":
            verified_task_ids = task_ids
            success = True
        else:
            split_selected_task_ids = selected_task_ids.split()
            verified_task_ids = []
            success = True
            for task_id in split_selected_task_ids:
                try:
                    if not int(task_id) in task_ids:
                        success = False
                        print(f"{task_id} is not a valid model ID")
                    else:
                        verified_task_ids.append(int(task_id))
                except ValueError:
                    success = False
                    print(f"{task_id} is not a valid model ID. Please only input numeric values.")
        if success:
            ask_samples(parent_folder, model_id, verified_task_ids)
            break

def ask_samples(parent_folder, model_id, task_ids):
    print("How many times do you want to train and logan per model? [1-20]")
    while True:
        ans = input("> ").lower()
        try:
            samples = int(ans)
            if samples < 1 or samples > 20:
                print("Invalid input. Select a number between 1 and 20.")
            else:
                _generate_file_for_training(parent_folder, model_id, task_ids, samples)
                break
        except ValueError:
            print("Invalid input. Select a number between 1 and 20.")

def _generate_file_for_training(parent_folder, model_id, task_ids, samples):
    shellfiles = []
    jobs_folder_local = os.path.join(os.path.dirname(os.getcwd()), f"jobs")
    jobs_folder = os.path.join(parent_folder, f"jobs")

    model = get_pretrained_model_by_id(int(model_id))
    print(model)
    slurm_jobs_folder_name = f"evaluate_trained_modelId_{model['modelId']}"
    slurm_jobs_folder_local = os.path.join(jobs_folder_local, slurm_jobs_folder_name)
    os.makedirs(slurm_jobs_folder_local, exist_ok=True)

    python_evaluate = os.path.join(parent_folder, "evaluate_model_trained.py")

    for task_id in task_ids:
        filename = f"{get_task_by_id(task_id)['alias']}"
        output_file = os.path.join(out_dir, f"{model_id}_{filename}.txt")
        content = \
f"""#!/bin/bash

#SBATCH --job-name={model_id}_{filename}
#SBATCH --output={output_file}
#SBATCH --cpus-per-task=2
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \\
python {python_evaluate} {model_id} {task_id} --samples {samples}"""

        with open(os.path.join(slurm_jobs_folder_local, f"{filename}.sh"), "w") as file:
            file.write(content)
        shellfiles.append(os.path.join("jobs", slurm_jobs_folder_name, f"{filename}.sh"))
    shellcontent = ""
    for shellfile in shellfiles:
        shellcontent += f"sbatch {shellfile}\n"

    shell = \
f"""#!/bin/bash
cd ..

{shellcontent}"""

    with open(os.path.join(jobs_folder_local, f"{slurm_jobs_folder_name}.sh"), "w") as file:
        file.write(shell)

if __name__ == "__main__":
    parent_folder = "/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis"
    model_ids = [model['modelId'] for model in PRETRAINED_MODELS]
    task_ids = [task['taskId'] for task in TASKS]
    samples = 10

    for model_id in model_ids:
        _generate_file_for_training(parent_folder, model_id, task_ids, samples)