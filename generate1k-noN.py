import math
import os.path

from datasets import load_dataset, Dataset
from config import datasets_cache_dir
from tqdm import tqdm

def split_with_overlap(sequence, description, chunk_size, overlap_size=200):
    length = len(sequence)
    step = chunk_size - overlap_size
    total_chunks = (length + step - 1) // step
    padded_length = total_chunks * step + overlap_size
    padded_text = sequence.ljust(padded_length, "_")
    chunks = []
    for i in range(total_chunks):
        start = i * step
        end = start + chunk_size
        chunks.append({
            'sequence': padded_text[start:end],
            'description': description,
            'start_pos': start,
            'end_pos': end,
        })
        if end == length:
            break

    return chunks


def find_optimal_chunk_size(dataset_split, max_chunk_length=6300, overlap=200):
    description = dataset_split[0]["description"]
    last_entry_idx = 0
    group = []
    while last_entry_idx < len(dataset_split) and dataset_split[last_entry_idx]['description'] == description:
        group.append(dataset_split[last_entry_idx])
        last_entry_idx += 1
    L = len(restore_original_sequence(group))
    for chunk_length in range(max_chunk_length, overlap, -1):
        step = chunk_length - overlap
        num_chunks = math.ceil((L - overlap) / step)
        total_covered = (num_chunks - 1) * step + chunk_length
        if total_covered == L:
            return chunk_length
    return None

def restore_original_sequence(group, overlap=200):
    chunks = [entry['sequence'] for entry in group]
    sequence = chunks[0]
    chunks.pop(0)
    chunks = [entry[overlap:] for entry in chunks]
    sequence += "".join(chunks)
    return sequence

def split_dataset(dataset_split):
    chunk_size = 1700
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
        if not 'N' in full_sequence:
            splits = split_with_overlap(full_sequence, description, chunk_size)
            if (len(full_sequence) != splits[-1]['end_pos']):
                print(len(full_sequence), splits[-1]['end_pos'])
                exit(1)
            for entry in splits:
                yield entry
        else:
            print("NO N")
        progress_bar.update(len(group))


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to size down splits of the multi_genome dataset"
    )
    parser.add_argument(
        "split",
        type=str,
        help="Dataset split [train|test|validation]",
        choices=["train", "test", "validation"],
    )

    split = parser.parse_args().split

    multi_species_genomes = load_dataset(
        "InstaDeepAI/multi_species_genomes",
        cache_dir=datasets_cache_dir,
        split=split,
        trust_remote_code=True
    )

    # When doing overlapping tokenization, we end up with more tokens than maximally allowed.
    # If we apply a sliding window over a sequence of 6200 characters, we end up with 6198 tokens.
    # Splitting the original sequences and storing them in a new dataset allows for overlapping tokenization
    # without going over the maximal length allowed by the model.
    # The tokenizer will right-pad the remaining to the max_length required by the model = 2048
    split_dataset_dir = os.path.join(datasets_cache_dir, 'InstaDeepAI___multi_species_genomes/1kbp-noN/', split)
    new_train_dataset = Dataset.from_generator(lambda: split_dataset(multi_species_genomes))
    new_train_dataset.save_to_disk(split_dataset_dir)


