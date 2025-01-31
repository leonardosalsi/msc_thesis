import os

from downstream_tasks import MODELS, TASKS
from util import get_chunk_size_file_name, get_model_by_id, get_task_by_id


def ask_evaluation(parent_folder):
    model_ids = [model['modelId'] for model in MODELS]
    model_names = [model['data_alias'] for model in MODELS]
    print("What models do you want to evaluate? [Multiple selections possible, separate by space. Type 0 for all.]")
    for i in range(len(model_ids)):
        print(f"[{model_ids[i]}]\t{model_names[i]}")
    while True:
        selected_model_ids = input("> ").lower()
        if selected_model_ids == "0":
            verified_model_ids = model_ids
            success = True
        else:
            split_selected_model_ids = selected_model_ids.split()
            verified_model_ids = []
            success = True
            for model_id in split_selected_model_ids:
                try:
                    if not int(model_id) in model_ids:
                        success = False
                        print(f"{model_id} is not a valid model ID")
                    else:
                        verified_model_ids.append(int(model_id))
                except ValueError:
                    success = False
                    print(f"{model_id} is not a valid model ID. Please only input numeric values.")
        if success:
            ask_task(parent_folder, verified_model_ids)
            break

def ask_task(parent_folder, verified_model_ids):
    task_ids = [task['taskId'] for task in TASKS]
    task_names = [task['data_alias'] for task in TASKS]
    print("On what tasks do you want to evaluate the models? [Multiple selections possible, separate by space. Type 0 for all.]")
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
            ask_random_weights(parent_folder, verified_model_ids, verified_task_ids)
            break

def ask_random_weights(parent_folder, model_ids, task_ids):
    print("Additionally evaluate models with random weights (where applicable)? [y|n]")
    while True:
        ans = input("> ").lower()
        random_weights = False
        if ans == "y":
            random_weights = True
            break
        elif ans == "n":
            break
        else:
            print("Invalid input. Select y or n.")
    ask_lora(parent_folder, model_ids, task_ids, random_weights)

def ask_lora(parent_folder, model_ids, task_ids, random_weights):
    print("Employ LoRA during fine-tuning? [y|n]")
    while True:
        ans = input("> ").lower()
        lora = False
        if ans == "y":
            lora = True
            break
        elif ans == "n":
            break
        else:
            print("Invalid input. Select y or n.")
    ask_samples(parent_folder, model_ids, task_ids, random_weights, lora)

def ask_samples(parent_folder, model_ids, task_ids, random_weights, lora):
    print("How many times do you want to train and evaluate per model? [1-20]")
    while True:
        ans = input("> ").lower()
        try:
            samples = int(ans)
            if samples < 1 or samples > 20:
                print("Invalid input. Select a number between 1 and 20.")
            else:
                _generate_file_for_training(parent_folder, model_ids, task_ids, random_weights, lora, samples)
                break
        except ValueError:
            print("Invalid input. Select a number between 1 and 20.")

def _generate_file_for_training(parent_folder, model_ids, task_ids, random_weights, lora, samples):
    shellfiles = []
    jobs_folder_local = os.path.join(os.path.dirname(os.getcwd()), f"jobs")
    jobs_folder = os.path.join(parent_folder, f"jobs")
    joined_model_ids = "".join(str(model_id) for model_id in model_ids)
    joined_task_ids = "".join(str(task_id) for task_id in task_ids)
    if random_weights:
        if not lora:
            slurm_jobs_folder_name = f"eval-models-{joined_model_ids}-{joined_task_ids}-{samples}-random-weights"
        else:
            slurm_jobs_folder_name = f"eval-models-{joined_model_ids}-{joined_task_ids}-{samples}-random-weights-lora"
    else:
        if not lora:
            slurm_jobs_folder_name = f"eval-models-{joined_model_ids}-{joined_task_ids}-{samples}"
        else:
            slurm_jobs_folder_name = f"eval-models-{joined_model_ids}-{joined_task_ids}-{samples}-lora"
    slurm_jobs_folder_local = os.path.join(jobs_folder_local, slurm_jobs_folder_name)
    os.makedirs(slurm_jobs_folder_local, exist_ok=True)

    python_evaluate = os.path.join(parent_folder, "evaluate_model.py")

    for model_id in model_ids:
        for task_id in task_ids:
            filename = f"eval-model-{get_model_by_id(model_id)['name']}-{get_task_by_id(task_id)['alias']}"
            content = \
f"""#!/bin/bash

#SBATCH --job-name={filename}
#SBATCH --output=out/{filename}.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \\
python {python_evaluate} {model_id} {task_id} {"--no-lora" if not lora else ""} --samples {samples}"""

            with open(os.path.join(slurm_jobs_folder_local, f"{filename}.sh"), "w") as file:
                file.write(content)
            shellfiles.append(os.path.join("jobs", slurm_jobs_folder_name, f"{filename}.sh"))
        if model_id in [1, 4]:
            for task_id in task_ids:
                filename = f"eval-model-{get_model_by_id(model_id)['name']}-{get_task_by_id(task_id)['alias']}-random-weights"
                content = \
f"""#!/bin/bash

#SBATCH --job-name={filename}
#SBATCH --output=out/{filename}.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \\
python {python_evaluate} {model_id} {task_id} --random-weights {"--no-lora" if not lora else ""} --samples {samples}"""

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