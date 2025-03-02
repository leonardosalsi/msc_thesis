import argparse
import os
import sys
from datasets import load_from_disk, DatasetDict, Dataset

from config import generated_datasets_dir  # adjust your import as needed

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to split each 2200-length sequence into two overlapping 1200-length chunks."
    )
    parser.add_argument(
        "kmer",
        type=int,
        help="Kmer size (only when using logan)",
    )
    parser.add_argument(
        "--reverse_complement",
        action="store_true",
        dest="reverse_complement",
        help="Use dataset generated with reverse complement (only when using logan)."
    )
    return parser.parse_args()

def split_sequence_to_two(example):
    sequence = example["sequence"]
    if len(sequence) != 2200:
        return []

    first_chunk = sequence[:1200]
    second_chunk = sequence[1000:2200]

    example_first = example.copy()
    example_first["sequence"] = first_chunk

    example_second = example.copy()
    example_second["sequence"] = second_chunk

    return [example_first, example_second]

def custom_flat_map(dataset, func):
    new_examples = []
    for example in dataset:
        new_examples.extend(func(example))
    return Dataset.from_list(new_examples)

if __name__ == "__main__":
    args = parse_args()
    kmer = args.kmer
    reverse_complement = args.reverse_complement

    if not kmer:
        print("Kmer size must be specified when using logan.")
        sys.exit(1)
    dataset_name = f"kmer_{kmer}"
    if reverse_complement:
        dataset_name += "_reverse"

    dataset_path = os.path.join(generated_datasets_dir, 'logan', dataset_name + "_2k")
    dataset = load_from_disk(dataset_path)

    processed_datasets = {}
    for split, ds in dataset.items():
        processed_datasets[split] = custom_flat_map(ds, split_sequence_to_two)

    new_dataset = DatasetDict(processed_datasets)
    logan_datasets_dir = os.path.join(generated_datasets_dir, f'logan')
    os.makedirs(logan_datasets_dir, exist_ok=True)
    if reverse_complement:
        dataset_dir = os.path.join(logan_datasets_dir, f'kmer_{kmer}_reverse_1k')
    else:
        dataset_dir = os.path.join(logan_datasets_dir, f'kmer_{kmer}_1k')
    os.makedirs(dataset_dir, exist_ok=True)
    new_dataset.save_to_disk(dataset_dir)
    print(dataset)
    print(new_dataset)
