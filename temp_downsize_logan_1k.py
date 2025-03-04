import argparse
import os
import sys
from datasets import load_from_disk, DatasetDict, Dataset

from config import generated_datasets_dir  # adjust your import as needed

def split_sequence_to_two(example):
    sequence = example["sequence"]
    chunks = []
    for i in range(0, len(sequence), 1000):
        chunk = sequence[i:i+1200]
        chunks.append({"sequence": chunk})  # Wrap the chunk in a dict.
    return chunks

def custom_flat_map(dataset, func):
    new_examples = []
    for example in dataset:
        new_examples.extend(func(example))
    return Dataset.from_list(new_examples)

if __name__ == "__main__":
    dataset_path = os.path.join(generated_datasets_dir, 'logan', "kmer_31_reverse")
    dataset = load_from_disk(dataset_path)

    processed_datasets = {}
    for split, ds in dataset.items():
        processed_datasets[split] = custom_flat_map(ds, split_sequence_to_two)

    new_dataset = DatasetDict(processed_datasets)
    logan_datasets_dir = os.path.join(generated_datasets_dir, f'logan')
    os.makedirs(logan_datasets_dir, exist_ok=True)
    dataset_dir = os.path.join(logan_datasets_dir, f'kmer_31_reverse_1k_unfiltered')

    os.makedirs(dataset_dir, exist_ok=True)
    new_dataset.save_to_disk(dataset_dir)
    print(dataset)
    print(new_dataset)
