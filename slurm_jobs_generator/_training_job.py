import os
from util import get_chunk_size_file_name

def ask_training(parent_folder):
    print("Training NT-50M-v2. Continue from pretrained model (by InstaDeep)? [y|n]")
    while True:
        ans = input("> ").lower()
        from_scratch = False
        if ans == "y":
            break
        elif ans == "n":
            from_scratch = True
            break
        else:
            print("Invalid input. Select y or n.")
    _ask_dataset(parent_folder, from_scratch)

def _ask_dataset(parent_folder, from_scratch):
    print("Which dataset would you like to use?")
    print("Multi Genome Dataset [1]")
    success = False
    while True:
        dataset_name = input("> ")
        if dataset_name == "1":
            dataset_name = "multi_genome_dataset"
            success = True
            break
        elif dataset_name == "q":
            break
        else:
            print("Invalid input. Select 1.")
    if success:
        _ask_tokenizer(parent_folder, from_scratch, dataset_name)

def _ask_tokenizer(parent_folder, from_scratch, dataset_name):
    print("Which tokenizer would you like to use?")
    print("Default [1]")
    print("OverlappingEsmTokenizer [2]")
    print("OverlappingEsmTokenizerWithNSkipping [3]")
    success = False
    while True:
        tokenizer_name = input("> ")
        if tokenizer_name == "1":
            tokenizer_name = "Default"
            success = True
            break
        elif tokenizer_name == "2":
            tokenizer_name = "OverlappingEsmTokenizer"
            success = True
            break
        elif tokenizer_name == "3":
            tokenizer_name = "OverlappingEsmTokenizerWithNSkipping"
            success = True
            break
        elif tokenizer_name == "q":
            break
        else:
            print("Invalid input. Select 1, 2 or 3.")
    if success:
        _ask_chunk_size(parent_folder, from_scratch, dataset_name, tokenizer_name)

def _ask_chunk_size(parent_folder, from_scratch, dataset_name, tokenizer_name):
    print("Specify chunk size used in a dataset split process.")
    while True:
        ans = input("> ")
        try:
            chunk_size = int(ans)
            _generate_file_for_training(parent_folder, from_scratch, dataset_name, tokenizer_name, chunk_size)
            break
        except ValueError:
            print("Invalid input. Provide a whole number.")
        except RuntimeError as e:
            print(e)
            print("Something went wrong.")

def _generate_file_for_training(parent_folder, from_scratch, dataset_name, tokenizer_name, chunk_size):
    shellfiles = []
    chunk_size_filename = get_chunk_size_file_name(chunk_size)
    jobs_folder_local = os.path.join(os.path.dirname(os.getcwd()), f"jobs")
    jobs_folder = os.path.join(parent_folder, f"jobs")
    if from_scratch:
        slurm_jobs_folder_name = f"train-model-{dataset_name}-{tokenizer_name.lower()}-{chunk_size_filename}-from_scratch"
    else:
        slurm_jobs_folder_name = f"train-model-{dataset_name}-{tokenizer_name.lower()}-{chunk_size_filename}"
    slurm_jobs_folder_local = os.path.join(jobs_folder_local, slurm_jobs_folder_name)
    os.makedirs(slurm_jobs_folder_local, exist_ok=True)

    python_train = os.path.join(parent_folder, "train_model.py")


    filename = slurm_jobs_folder_name
    content = \
f"""#!/bin/bash

#SBATCH --job-name={filename}
#SBATCH --output=out/{filename}.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \\
python {python_train} {dataset_name} {tokenizer_name} {chunk_size} {"--from_scratch" if from_scratch else ""}"""

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