import os
import shutil
import glob
import json
from datasets import load_dataset, load_from_disk, Dataset
from config import datasets_cache_dir, generator_cache_dir
from utils.util import check_folders
import re

def json_files_generator(folder_path):
    pattern = os.path.join(folder_path, '*.json')
    for file_path in glob.glob(pattern):
        file_basename = os.path.basename(file_path)
        match = re.search(r'random_walk_(\d+)\.json$', file_basename)
        if not match:
            print(f"Skipping file {file_path}: filename does not match expected pattern.")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                array = json.load(f)
                if isinstance(array, list):
                    for it in array:
                        yield it
                else:
                    print(f"Skipping file {file_path}: Expected a JSON array but got {type(array)}")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

def get_dataset(args):
    """
    Loads and returns the training and validation datasets based on the provided arguments.

    If the tokenizer is "Default", the function loads the 'multi_species_genomes'
    dataset from the Hugging Face Hub. Otherwise, it uses local dataset paths provided in args.dataset,
    after verifying that the folders exist via the check_folders utility.

    Additionally, if args.use_scratch is True, the function copies the dataset directory from the
    specified path to the local scratch directory (as given by the TMPDIR environment variable) and then
    loads the dataset from that location.

    After loading, the function removes all columns except 'sequence' from both datasets.

    :param args: Arguments containing tokenizer type and dataset path.
    :return: A tuple (dataset_train, dataset_validation)
    """

    selected_tokenizer = args.tokenizer
    selected_dataset_path = args.dataset
    use_scratch = args.use_scratch
    keep_in_memory = args.keep_in_memory
    load_from_json = args.load_from_json

    if use_scratch:
        tmpdir = os.environ.get("TMPDIR")
        if tmpdir is None:
            raise ValueError("TMPDIR environment variable is not set, but use_scratch is True.")
        scratch_dataset_path = os.path.join(tmpdir, os.path.basename(selected_dataset_path))
        if not os.path.exists(scratch_dataset_path):
            shutil.copytree(selected_dataset_path, scratch_dataset_path)
        selected_dataset_path = scratch_dataset_path

    if 'InstaDeepAI' in selected_dataset_path:
        dataset_train = load_dataset(
            "InstaDeepAI/multi_species_genomes",
            cache_dir=datasets_cache_dir,
            split='train',
            trust_remote_code=True,
            keep_in_memory=keep_in_memory,
        )
        dataset_validation = load_dataset(
            "InstaDeepAI/multi_species_genomes",
            cache_dir=datasets_cache_dir,
            split='validation',
            trust_remote_code=True,
            keep_in_memory=keep_in_memory,
        )
    else:
        if load_from_json:
            file_list = [os.path.join(selected_dataset_path, f) for f in os.listdir(selected_dataset_path) if f.endswith('.json')]
            validation_size = 500000
            def gen():
                for file in file_list:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        for item in data:
                            yield {'sequence': item}

            dataset = Dataset.from_generator(gen, cache_dir=generator_cache_dir)
            dataset = dataset.shuffle()
            dataset_train = dataset.select(range(validation_size, len(dataset)))
            dataset_validation = dataset.select(range(validation_size))
        else:
            train_path, validation_path = check_folders(selected_dataset_path)
            dataset_train = load_from_disk(train_path, keep_in_memory=keep_in_memory)
            dataset_validation = load_from_disk(validation_path, keep_in_memory=keep_in_memory)
    print(dataset_train)
    print(dataset_validation)
    columns_to_remove = [col for col in dataset_train.column_names if col != "sequence"]
    dataset_train = dataset_train.remove_columns(columns_to_remove)
    columns_to_remove = [col for col in dataset_validation.column_names if col != "sequence"]
    dataset_validation = dataset_validation.remove_columns(columns_to_remove)

    return dataset_train, dataset_validation


def get_original_training_dataset(args):

    if 'InstaDeepAI' in args.original_dataset:
        cache_dir = datasets_cache_dir
        if args.use_scratch:
            folder_name = args.original_dataset.replace("/", "___")
            tmpdir = os.environ.get("TMPDIR")
            if tmpdir is None:
                raise ValueError("TMPDIR environment variable is not set, but use_scratch is True.")
            scratch_dataset_path = os.path.join(tmpdir, os.path.basename(folder_name))
            if not os.path.exists(scratch_dataset_path):
                shutil.copytree(os.path.join(datasets_cache_dir, folder_name), scratch_dataset_path)
            cache_dir = scratch_dataset_path
        dataset_train = load_dataset(
            "InstaDeepAI/multi_species_genomes",
            cache_dir=cache_dir,
            split='train',
            trust_remote_code=True,
        )
    else:
        train_path, _ = check_folders(args.original_dataset)
        dataset_train = load_from_disk(train_path, keep_in_memory=False)
    columns_to_remove = [col for col in dataset_train.column_names if col != "sequence"]
    dataset_train = dataset_train.remove_columns(columns_to_remove)
    return dataset_train