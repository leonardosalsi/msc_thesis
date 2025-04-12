import os
import shutil

import datasets
from datasets import load_dataset, load_from_disk
from config import datasets_cache_dir, generated_datasets_dir
from pre_train.util import check_folders


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

    if use_scratch:
        tmpdir = os.environ.get("TMPDIR")
        if tmpdir is None:
            raise ValueError("TMPDIR environment variable is not set, but use_scratch is True.")
        scratch_dataset_path = os.path.join(tmpdir, os.path.basename(selected_dataset_path))
        if not os.path.exists(scratch_dataset_path):
            shutil.copytree(selected_dataset_path, scratch_dataset_path)
        selected_dataset_path = scratch_dataset_path

    if selected_tokenizer == "Default":
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
        train_path, validation_path = check_folders(selected_dataset_path)

        dataset_train = load_from_disk(train_path, keep_in_memory=keep_in_memory)
        dataset_validation = load_from_disk(validation_path, keep_in_memory=keep_in_memory)

    columns_to_remove = [col for col in dataset_train.column_names if col != "sequence"]
    dataset_train = dataset_train.remove_columns(columns_to_remove)
    columns_to_remove = [col for col in dataset_validation.column_names if col != "sequence"]
    dataset_validation = dataset_validation.remove_columns(columns_to_remove)

    return dataset_train, dataset_validation


def get_original_training_dataset():
    train_path, validation_path = check_folders(os.path.join(generated_datasets_dir, '1k', 'train'))
    dataset_train = load_from_disk(train_path, keep_in_memory=False)
    columns_to_remove = [col for col in dataset_train.column_names if col != "sequence"]
    dataset_train = dataset_train.remove_columns(columns_to_remove)
    return dataset_train