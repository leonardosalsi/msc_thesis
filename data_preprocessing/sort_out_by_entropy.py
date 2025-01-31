import math
import os.path
import argparse
from datasets import load_dataset, Dataset
from config import datasets_cache_dir, generated_datasets_dir
from tqdm import tqdm
from util import get_chunk_size_file_name, get_entropy_file_name

def restore_original_sequence(group, overlap=200):
    chunks = [entry['sequence'] for entry in group]
    sequence = chunks[0]
    chunks.pop(0)
    chunks = [entry[overlap:] for entry in chunks]
    sequence += "".join(chunks)
    return sequence

def calculate_shannon_entropy(sequence):
    counter = {"A": 0, "T": 0, "C": 0, "G": 0}
    sequence_length = len(sequence)

    def get_part_entopy(prob):
        return - prob * math.log(prob)

    for c in sequence:
       counter[c] += 1
    prob_A = get_part_entopy(counter["A"] / sequence_length)
    prob_T = get_part_entopy(counter["T"] / sequence_length)
    prob_C = get_part_entopy(counter["C"] / sequence_length)
    prob_G = get_part_entopy(counter["G"] / sequence_length)
    return prob_A + prob_T + prob_C + prob_G


def split_dataset(dataset_split, low, high):
    i = 0
    progress_bar = tqdm(total=len(dataset_split), desc="Splitting dataset", unit="entry")
    while i < len(dataset_split):
        description = dataset_split[i]["description"]
        last_entry_idx = i
        group = []
        while last_entry_idx < len(dataset_split) and dataset_split[last_entry_idx]['description'] == description:
            group.append(dataset_split[last_entry_idx])
            last_entry_idx += 1
        i = last_entry_idx
        full_sequence = restore_original_sequence(group)
        entropy = calculate_shannon_entropy(full_sequence)
        print(entropy)
        yield
        progress_bar.update(len(group))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to size down splits of the multi_genome dataset"
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Name of the dataset",
        choices=["multi_genome_dataset"]
    )
    parser.add_argument(
        "split",
        type=str,
        help="Dataset split [train|test|validation]",
        choices=["train", "test", "validation"],
    )
    parser.add_argument(
        "low",
        type=float,
        help="Lower margin of allowed entropy"
    )
    parser.add_argument(
        "high",
        type=float,
        help="Upper margin of allowed entropy"
    )
    return parser.parse_args()

"""
We take a preexisting dataset and create a new one based on the Shannon entropy metrics.
For this, we pass (alongside the dataset and the split) the lower and upper margin of the entropy and keep
all sequences that have an entropy within this interval.
"""
if __name__ == "__main__":
    args = parse_args()
    split = args.split
    selected_dataset = args.dataset
    low = args.low
    high = args.high

    """
    Select dataset to filter by entropy
    """
    if selected_dataset == "multi_genome_dataset":
        dataset = load_dataset(
            "InstaDeepAI/multi_species_genomes",
            cache_dir=datasets_cache_dir,
            split=split,
            trust_remote_code=True
        )
    else:
        print(f"Unknown dataset {selected_dataset}")
        exit(1)

    dataset = dataset.select(range(1000))

    """
    Recreate dataset with allowed sequences
    """
    entropy_file_name = get_entropy_file_name(low, high)
    split_dataset_dir = os.path.join(generated_datasets_dir, selected_dataset, entropy_file_name, split)

    """
    Save dataset to disk
    """
    new_train_dataset = Dataset.from_generator(lambda: split_dataset(dataset, low, high))
    new_train_dataset.save_to_disk(split_dataset_dir)



