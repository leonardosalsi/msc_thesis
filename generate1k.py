import math

from datasets import load_dataset, Dataset
import textwrap
from config import datasets_cache_dir
from tqdm import tqdm
from pprint import pprint

def split_dataset(dataset_split, split_name, split_sequence_length):
    split_dataset = []
    for entry in tqdm(dataset_split, desc=f"Creating {split_name} split", unit="sequence"):
        sequence = entry["sequence"]
        description = entry["description"]
        start_pos = entry["start_pos"]
        end_pos = entry["end_pos"]
        fasta_url = entry["fasta_url"]
        ranges = list(range(start_pos, end_pos, split_sequence_length))
        split_sequences = textwrap.wrap(sequence, split_sequence_length)
        sequence_position_tuples = zip(split_sequences, ranges)
        for (s, r) in sequence_position_tuples:
            split_dataset.append({
                "sequence": s,
                "description": description,
                "start_pos": r,
                "end_pos": r + split_sequence_length,
                "fasta_url": fasta_url
            })
    return split_dataset


if __name__ == "__main__":
    multi_species_genomes = load_dataset(
        "InstaDeepAI/multi_species_genomes",
        cache_dir=datasets_cache_dir,
        trust_remote_code=True
    )

    split_dataset_dir = "/shared/datasets/InstaDeepAI___multi_species_genomes/1kbp/"

    multi_species_genomes_train = multi_species_genomes["train"]
    multi_species_genomes_test = multi_species_genomes["test"]
    multi_species_genomes_val = multi_species_genomes["validation"]

    # When doing overlapping tokenization, we end up with more tokens than maximally allowed.
    # If we apply a sliding window over a sequence of 6200 characters, we end up with 6198 tokens.
    # Splitting the original sequences and storing them in a new dataset allows for overlapping tokenization
    # without going over the maximal length allowed by the model.
    # The tokenizer will right-pad the remaining to the max_length required by the model = 2048
    split_num = 4
    sequence_length = len(multi_species_genomes_train[0]["sequence"])
    split_sequence_length = math.floor(sequence_length / split_num)

    new_train_split = split_dataset(multi_species_genomes_train, "train", split_sequence_length)
    new_train_dataset = Dataset.from_list(new_train_split)
    new_train_dataset.save_to_disk(f"{split_dataset_dir}/train")

    new_test_split = split_dataset(multi_species_genomes_test, "test", split_sequence_length)
    new_test_dataset = Dataset.from_list(new_test_split)
    new_test_dataset.save_to_disk(f"{split_dataset_dir}/test")

    new_val_split = split_dataset(multi_species_genomes_val, "validation", split_sequence_length)
    new_val_dataset = Dataset.from_list(new_val_split)
    new_val_dataset.save_to_disk(f"{split_dataset_dir}/validation")
