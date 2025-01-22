import os

from config import tokenized_datasets_dir
from util import get_chunk_size_folder_name


def ask_training(parent_folder):
    print("Training NT-50M-v2. Continue from pretrained model (by InstaDeep)? [y|n]")
    while True:
        ans = input("> ").lower()
        from_scratch = False
        if ans == "y":
            from_scratch = True
            break
        elif ans == "n":
            break
        else:
            print("Invalid input. Select y or n.")
    _ask_tokenizer(parent_folder, from_scratch)

def _ask_tokenizer(parent_folder, from_scratch):
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
        tokenized_dataset_folder = os.path.join(tokenized_datasets_dir, tokenizer)
        if not os.path.exists(tokenized_dataset_folder) or not os.path.isdir(tokenized_dataset_folder):
            print(f"There is no folder {tokenized_dataset_folder}. Make sure to tokenize data before creating training jobs.")
            return
        _ask_chunk_size(tokenized_dataset_folder, from_scratch, tokenizer, tokenized_dataset_folder)


def _ask_chunk_size(parent_folder, from_scratch, tokenizer, tokenized_dataset_folder):
    print("Specify chunk size used in a dataset split process.")
    while True:
        ans = input("> ")
        try:
            chunk_size = int(ans)
            chunk_size_filename = get_chunk_size_folder_name(chunk_size)
            dataset_path = os.path.join(tokenized_dataset_folder, chunk_size_filename)
            if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
                print(f"The folder {dataset_path} does not exist.")
                print(f"Make sure to generate the split dataset with the chunk size {chunk_size}.")
                break

        except ValueError:
            print("Invalid input. Provide a whole number.")
        except RuntimeError as e:
            print(e)
            print("Something went wrong.")
