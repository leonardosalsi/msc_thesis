import os

from slurm_jobs_generator._downsize_chunk_job import ask_downsize
from slurm_jobs_generator._evaluate_job import ask_evaluation
from slurm_jobs_generator._training_job import ask_training

if __name__ == "__main__":
    parent_folder = os.path.dirname(os.getcwd())
    print(f"Project in {parent_folder}")
    print(f"Want to specify different directory? [y|n]")
    while True:
        ans = input("> ")
        if ans == "y":
            print("Specify the full path to the root of the project")
            parent_folder = input("> ")
            print(f"Project in {parent_folder}")
            break
        if ans == "n":
            break
        else:
            print("Invalid input. Select either y or n.")
    print("Generate jobs for training [1], model evaliation by MCC [2] or dataset chunk downsizing [3]?")
    mode = ""
    while True:
        ans = input("> ")
        if ans == "1":
            ask_training(parent_folder)
            break
        elif ans == "2":
            ask_evaluation(parent_folder)
            break
        elif ans == "3":
            ask_downsize(parent_folder)
            break
        else:
            print("Invalid input. Select 1, 2 or 3.")
