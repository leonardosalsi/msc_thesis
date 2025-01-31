import os

from util import get_chunk_size_file_name


def ask_downsize(parent_folder):
    print("Which dataset would you like to use generate smaller chunks from?")
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
        _ask_split(parent_folder, dataset_name)

def _ask_split(parent_folder, dataset_name):
    print("Which split should be tokenized? [train|test|validation|all]")
    success = False
    while True:
        splits = input("> ")
        if splits == "test":
            success = True
            break
        elif splits == "train":
            success = True
            break
        elif splits == "validation":
            success = True
            break
        elif splits == "all":
            success = True
            break
        else:
            print(f"The split {splits} is not recognized. Select from test, train, validation or all.")
    if success:
        _ask_chunk_size(parent_folder, dataset_name, splits)

def _ask_chunk_size(parent_folder, dataset_name, splits):
    print("Specify chunk size used in the dataset split process (including overlap).")
    while True:
        ans = input("> ")
        try:
            chunk_size = int(ans)
            _generate_file_for_downsizing(parent_folder, dataset_name, splits, chunk_size)
            break
        except ValueError:
            print("Invalid input. Provide a whole number.")
        except RuntimeError as e:
            print(e)
            print("Something went wrong.")

def _generate_file_for_downsizing(parent_folder, dataset_name, _splits, chunk_size):
    shellfiles = []
    chunk_size_filename = get_chunk_size_file_name(chunk_size)
    jobs_folder_local = os.path.join(os.path.dirname(os.getcwd()), f"jobs")
    jobs_folder = os.path.join(parent_folder, f"jobs")

    splits = []
    if _splits == "train":
        splits = ["train"]
    elif _splits == "test":
        splits = ["test"]
    elif _splits == "validation":
        splits = ["validation"]
    elif _splits == "all":
        splits = ["train", "test", "validation"]

    slurm_jobs_folder_name = f"downsize-chunks-{dataset_name}-{_splits}-{chunk_size_filename}-from_scratch"
    slurm_jobs_folder_local = os.path.join(jobs_folder_local, slurm_jobs_folder_name)
    os.makedirs(slurm_jobs_folder_local, exist_ok=True)

    python_downsize = os.path.join(parent_folder, "downsize_dataset_chunks.py")

    for split in splits:
        filename = f"downsize-chunks-{dataset_name}-{split}-{chunk_size_filename}-from_scratch"
        content = \
f"""#!/bin/bash

#SBATCH --job-name={filename}
#SBATCH --output=out/{filename}.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \\
python {python_downsize} {dataset_name} {split} {chunk_size}"""

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