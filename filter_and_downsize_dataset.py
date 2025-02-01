import math
import os.path
import argparse
from datasets import load_dataset, Dataset
from config import datasets_cache_dir, generated_datasets_dir
from tqdm import tqdm
from util import get_chunk_size_file_name, get_filtered_dataset_name

def restore_original_sequence(group, overlap=200):
    chunks = [entry['sequence'] for entry in group]
    sequence = chunks[0]
    chunks.pop(0)
    chunks = [entry[overlap:] for entry in chunks]
    sequence += "".join(chunks)
    return sequence

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

def calculate_shannon_entropy(sequence):
    counter = {"A": 0, "T": 0, "C": 0, "G": 0, "N": 0}
    sequence_length = len(sequence)

    def get_part_entopy(prob):
        if prob == 0:
            return 0
        return - prob * math.log(prob)

    for c in sequence:
       counter[c] += 1
    prob_A = get_part_entopy(counter["A"] / sequence_length)
    prob_T = get_part_entopy(counter["T"] / sequence_length)
    prob_C = get_part_entopy(counter["C"] / sequence_length)
    prob_G = get_part_entopy(counter["G"] / sequence_length)
    prob_N = get_part_entopy(counter["N"] / sequence_length)
    return prob_A + prob_T + prob_C + prob_G + prob_N

def calculate_gc_content(sequence):
    gc_count = sequence.upper().count('G') + sequence.upper().count('C')
    return gc_count / len(sequence)

def filter_dataset(dataset_split, chunk_size, shannon, gc):
    i = 0
    progress_bar = tqdm(total=len(dataset_split), desc="Splitting dataset", unit="entry")
    result_len = 0
    def_num_groups = 0
    while i < len(dataset_split):
        description = dataset_split[i]["description"]
        last_entry_idx = i
        group = []
        while last_entry_idx < len(dataset_split) and dataset_split[last_entry_idx]['description'] == description:
            group.append(dataset_split[last_entry_idx])
            last_entry_idx += 1
        def_num_groups += 1
        i = last_entry_idx
        full_sequence = restore_original_sequence(group)
        """Evaluate if allowed"""
        shannon_ok = True
        gc_ok = True
        if shannon is not None:
            shannon_entropy = calculate_shannon_entropy(full_sequence)
            if shannon_entropy < shannon[0] or shannon_entropy > shannon[1]:
                shannon_ok = False
        if gc is not None:
            gc_content = calculate_gc_content(full_sequence)
            if gc_content < gc[0] or gc_content > gc[1]:
                gc_ok = False
        is_allowed = shannon_ok and gc_ok
        if is_allowed:
            """Create smaller chunks"""
            splits = split_with_overlap(full_sequence, description, chunk_size)
            if (len(full_sequence) != splits[-1]['end_pos']):
                print(len(full_sequence), splits[-1]['end_pos'])
                exit(1)
            for entry in splits:
                yield entry
            result_len += 1
        progress_bar.update(len(group))
    print(f"result_len: {result_len} (total {100 / def_num_groups * result_len:.2f}%)")

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
        "chunk_size",
        type=int,
        help="Size of each chunk (including overlap)"
    )
    parser.add_argument(
        "--shannon",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Lower and upper margin of allowed Shannon entropy (e.g., --shannon 1.4 1.8)"
    )
    parser.add_argument(
        "--gc",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Lower and upper margin of allowed GC content (e.g., --gc 0.4 0.6)"
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
    chunk_size = args.chunk_size
    shannon = args.shannon
    gc = args.gc
    if shannon is None and gc is None:
        print("You must specify either --shannon or --gc")
        exit(1)

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

    """
    Recreate dataset with allowed sequences
    """
    entropy_file_name = get_filtered_dataset_name(chunk_size, shannon, gc)
    split_dataset_dir = os.path.join(generated_datasets_dir, selected_dataset, entropy_file_name, split)
    """
    Save dataset to disk
    """
    new_train_dataset = Dataset.from_generator(lambda: filter_dataset(dataset, chunk_size, shannon, gc))
    new_train_dataset.save_to_disk(split_dataset_dir)



