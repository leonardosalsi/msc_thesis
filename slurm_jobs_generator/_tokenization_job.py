import os
from config import datasets_cache_dir
from util import get_chunk_size_folder_name

def ask_tokenization(parent_folder):
    print("Prerequisite is that splits have already been created.")
    print("Specify chunk size used in a dataset split process.")
    while True:
        ans = input("> ")
        try:
            chunk_size = int(ans)
            chunk_size_filename = get_chunk_size_folder_name(chunk_size)
            dataset_path = os.path.join(datasets_cache_dir , "InstaDeepAI___multi_species_genomes/", chunk_size_filename)
            if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
                print(f"The folder {dataset_path} does not exist.")
                print(f"Make sure to generate the split dataset with the chunk size {chunk_size}.")
                break
            print("Which split should be tokenized? [train|test|validation|all]")
            success = False
            while True:
                split = input("> ")
                if split == "test":
                    if not os.path.exists(os.path.join(dataset_path, 'test')) or not os.path.isdir(os.path.join(dataset_path, 'test')):
                        print(f"The folder {os.path.join(dataset_path, 'test')} does not exist.")
                        print(f"Although dataset has been split, the test split from the original dataset was not used.")
                    else:
                        success = True
                    break
                elif split == "train":
                    if not os.path.exists(os.path.join(dataset_path, 'train')) or not os.path.isdir(os.path.join(dataset_path, 'train')):
                        print(f"The folder {os.path.join(dataset_path, 'train')} does not exist.")
                        print(f"Although dataset has been split, the train split from the original dataset was not used.")
                    else:
                        success = True
                    break
                elif split == "validation":
                    if not os.path.exists(os.path.join(dataset_path, 'validation')) or not os.path.isdir(os.path.join(dataset_path, 'validation')):
                        print(f"The folder {os.path.join(dataset_path, 'validation')} does not exist.")
                        print(f"Although dataset has been split, the validation split from the original dataset was not used.")
                    else:
                        success = True
                    break
                elif split == "all":
                    if not os.path.exists(os.path.join(dataset_path, 'train')) or not os.path.isdir(os.path.join(dataset_path, 'train')):
                        print(f"The folder {os.path.join(dataset_path, 'train')} does not exist.")
                        print(f"Although dataset has been split, the train split from the original dataset was not used.")
                    elif not os.path.exists(os.path.join(dataset_path, 'test')) or not os.path.isdir(os.path.join(dataset_path, 'test')):
                        print(f"The folder {os.path.join(dataset_path, 'test')} does not exist.")
                        print(f"Although dataset has been split, the test split from the original dataset was not used.")
                    elif not os.path.exists(os.path.join(dataset_path, 'validation')) or not os.path.isdir(os.path.join(dataset_path, 'validation')):
                        print(f"The folder {os.path.join(dataset_path, 'validation')} does not exist.")
                        print(f"Although dataset has been split, the validation split from the original dataset was not used.")
                    else:
                        success = True
                    break
                else:
                    print(f"The split {split} is not recognized. Select from test, train, validation or all.")
            if success:
                _ask_tokenizer(parent_folder, split, chunk_size)
                break
            else:
                break
        except ValueError:
            print("Invalid input. Provide a whole number.")
        except RuntimeError as e:
            print(e)
            print("Something went wrong.")

def _ask_tokenizer(parent_folder, split, chunk_size):
    print("Which tokenizer would you like to use?")
    print("Default [1]")
    print("OverlappingEsmTokenizer [2]")
    print("OverlappingEsmTokenizerWithNSkipping [3]")
    success = False
    while True:
        tokenizer = input("> ")
        if tokenizer == "1":
            tokenizer = "Default"
            success = True
            break
        elif tokenizer == "2":
            tokenizer = "OverlappingEsmTokenizer"
            success = True
            break
        elif tokenizer == "3":
            tokenizer = "OverlappingEsmTokenizerWithNSkipping"
            success = True
            break
        elif tokenizer == "q":
            break
        else:
            print("Invalid input. Select 1, 2 or 3.")
    if success:
        _generate_file_for_tokenization(parent_folder, split, tokenizer, chunk_size)

def _generate_file_for_tokenization(parent_folder, _split, tokenizer, chunk_size):
    shellfiles = []
    splits = []
    chunk_size_filename = get_chunk_size_folder_name(chunk_size)
    jobs_folder_local = os.path.join(os.path.dirname(os.getcwd()), f"jobs")
    jobs_folder = os.path.join(parent_folder, f"jobs")
    slurm_jobs_folder_name = f"tokenize-{chunk_size_filename}-{_split}-{tokenizer.lower()}"
    slurm_jobs_folder_local = os.path.join(jobs_folder_local, slurm_jobs_folder_name)
    os.makedirs(slurm_jobs_folder_local, exist_ok=True)
    if _split == "train":
        splits = ["train"]
    elif _split == "test":
        splits = ["test"]
    elif _split == "validation":
        splits = ["validation"]
    elif _split == "all":
        splits = ["train", "test", "validation"]

    python_tokenizer = os.path.join(parent_folder, "tokenize_dataset.py")

    for split in splits:
        filename = f"{chunk_size_filename}_{split}-{tokenizer.lower()}"
        content = \
f"""#!/bin/bash

#SBATCH --job-name={filename}
#SBATCH --output=out/{filename}.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \\
python {python_tokenizer} {split} {tokenizer} {chunk_size}"""

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