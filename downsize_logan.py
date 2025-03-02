import argparse
import json
import os
import sys
from pprint import pprint

import numpy as np
from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from config import generated_datasets_dir, models_cache_dir, datasets_cache_dir, results_dir
from util import get_filtered_dataset_name


def split_sequence(example):
    sequence = example["sequence"]
    if len(sequence) < min_size:
        return []  # Return an empty list to drop examples that are too short.
    chunks = []
    # Iterate over the sequence in steps of min_size.
    for i in range(0, len(sequence), min_size):
        chunk = sequence[i:i + min_size]
        if len(chunk) == min_size:
            new_example = example.copy()
            new_example["sequence"] = chunk
            chunks.append(new_example)
    return chunks

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to clean up the Logan dataset."
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

def flat_map(dataset, function):
    all_examples = []
    for example in dataset:
        # function should return a list of examples for each input example.
        all_examples.extend(function(example))
    return Dataset.from_list(all_examples)


if __name__ == "__main__":
    args = parse_args()
    kmer = args.kmer
    reverse_complement = args.reverse_complement
    min_size = 2200

    if not kmer:
        print("Kmer size must be specified when using logan.")
        exit(1)
    dataset_name = f"kmer_{kmer}"
    if reverse_complement:
        dataset_name += "_reverse"
    dataset_path = os.path.join(generated_datasets_dir, 'logan', dataset_name)
    dataset = load_from_disk(dataset_path)

    processed_datasets = {}
    for split, ds in dataset.items():
        processed_datasets[split] = ds.filter(lambda x: len(x["sequence"]) == min_size)

    min_dataset = DatasetDict(processed_datasets)
    logan_datasets_dir = os.path.join(generated_datasets_dir, f'logan')
    os.makedirs(logan_datasets_dir, exist_ok=True)
    if reverse_complement:
        dataset_dir = os.path.join(logan_datasets_dir, f'kmer_{kmer}_reverse_2k')
    else:
        dataset_dir = os.path.join(logan_datasets_dir, f'kmer_{kmer}_2k')
    os.makedirs(dataset_dir, exist_ok=True)
    min_dataset.save_to_disk(dataset_dir)
    print(dataset)
    print(min_dataset)
