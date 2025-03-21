import argparse
import csv
import os
from pprint import pprint

from datasets import Dataset, DatasetDict, Features, Value
from config import logan_datasets_dir, generated_datasets_dir, generator_cache_dir
import fasta_walker

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train model either from scratch or from pretrained weights with specified tokenization."
    )

    parser.add_argument(
        "fasta_files_path",
        type=str,
        help="Folder of FASTA files to be processed",
    )

    parser.add_argument(
        "--metadata_file_path",
        type=str,
        help="Path to metadata file",
    )

    parser.add_argument(
        "--acc_column",
        type=str,
        help="Column header name of accession in metadata file"
    )

    parser.add_argument(
        "--group_id_column",
        type=str,
        help="Column header name of group id of accession in metadata file"
    )

    parser.add_argument(
        "--kmer",
        type=int,
        default=31,
        help="Kmer length",
        choices=[31, 28, 25, 20]
    )

    parser.add_argument(
        "--reverse_complement",
        action="store_true",
        dest="reverse_complement",
        help="Also include reverse complements to graph."
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=0,
        help="Chunk size (defined when further splitting data)",
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Chunk size (defined when further splitting data)",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    max_workers = args.max_workers
    kmer = args.kmer
    chunk_size = args.chunk_size
    reverse_complement = args.reverse_complement
    fasta_files_path = args.fasta_files_path
    metadata_path = args.metadata_file_path
    metadata_acc_column = args.acc_column
    metadata_group_id_column = args.group_id_column

    dataset_dir = os.path.join(generated_datasets_dir, f'logan')
    cache_dir = os.path.join(generator_cache_dir, 'logan')
    os.makedirs(logan_datasets_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    if reverse_complement:
        dataset_dir = os.path.join(dataset_dir, f'kmer_{kmer}_reverse')
        generator_cache = os.path.join(cache_dir, f'kmer_{kmer}_reverse')
    else:
        dataset_dir = os.path.join(dataset_dir, f'kmer_{kmer}')
        generator_cache = os.path.join(cache_dir, f'kmer_{kmer}')
    dataset_dir = dataset_dir + f"_{chunk_size}k"
    cache_dir = cache_dir  + f"_{chunk_size}k"
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    features = Features({
        "sequence": Value("string"),
        f"{metadata_group_id_column}": Value("string"),
    })

    """
    gen = fasta_walker.create_random_walk_sequences(
        kmer,
        chunk_size,
        reverse_complement,
        fasta_files_path,
        metadata_path,
        metadata_acc_column,
        metadata_group_id_column
    )

    for g in gen:
        print(g)
    """

    new_dataset = Dataset.from_generator(
        lambda: fasta_walker.create_random_walk_sequences(
            kmer,
            chunk_size,
            reverse_complement,
            fasta_files_path,
            metadata_path,
            metadata_acc_column,
            metadata_group_id_column
        ),
        cache_dir=cache_dir,
        features=features,
    )

    split_dataset = new_dataset.train_test_split(test_size=0.2, seed=112)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })



    dataset.save_to_disk(dataset_dir, num_proc=max_workers)
    dataset_dir = dataset_dir + f"_filtered"
    cache_dir = cache_dir + f"_filtered"

    def filtered_generator(split):
        for example in split:
            if len(example["sequence"]) == chunk_size:
                yield example

    filtered_train = Dataset.from_generator(lambda: filtered_generator(dataset["train"]), cache_dir=cache_dir + "_train")
    filtered_test = Dataset.from_generator(lambda: filtered_generator(dataset["test"]), cache_dir=cache_dir + "_test")

    filtered_dataset = DatasetDict({
        "train": filtered_train,
        "test": filtered_test
    })

    filtered_dataset.save_to_disk(dataset_dir, num_proc=max_workers)
