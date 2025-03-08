import argparse
import concurrent.futures
import math
import os
import random
import glob
from io import StringIO
import numpy as np
import zstandard
from datasets import Dataset, DatasetDict
from Bio import SeqIO
from tqdm import tqdm
from collections import defaultdict
from config import results_dir, logan_datasets_dir, generated_datasets_dir

ALPHABET = {"A", "T", "C", "G"}

def chop_at_first_repeated_kmer(sequence, k):
    kmers = set()
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if kmer in kmers:
            return sequence[:i + k - 1]
        kmers.add(kmer)
    return sequence

def find_overlaps_and_build_graph(sequences, k_mer=3):
    min_overlap = k_mer - 1
    prefix_dict = defaultdict(list)
    graph = defaultdict(list)

    for i, seq in enumerate(sequences):
        prefix_dict[seq[:min_overlap]].append(i)


    for i, seq1 in enumerate(sequences):
        seq1_suffix = seq1[-min_overlap:]
        graph[i] = []
        for j in prefix_dict[seq1_suffix]:
            if i != j:
                graph[i].append(j)

    return graph

def random_dfs_path(graph, start, depth=5000):
    path = [start]
    visited = {start}
    current = start

    while len(path) < depth:
        neighbors = graph.get(current, [])
        valid_neighbors = [n for n in neighbors if n not in visited]
        if not valid_neighbors:
            break

        nxt = random.choice(valid_neighbors)
        path.append(nxt)
        visited.add(nxt)
        current = nxt
    return path


def sample_longest_dfs_path_sequence(graph, start, sequences, kmer, samples=10, depth=10000):
    best_path = None
    best_length = 0
    for _ in range(samples):
        path = random_dfs_path(graph, start, depth)
        if sequences is not None and kmer is not None:
            seq = sequences[path[0]] + "".join([sequences[p][kmer - 1:] for p in path[1:]])
            current_length = len(seq)
        else:
            current_length = len(path)  # fallback: compare by path length
        if current_length > best_length:
            best_length = current_length
            best_path = path
    sequence = sequences[best_path[0]] + "".join([sequences[p][kmer - 1:] for p in best_path[1:]])
    return sequence

def random_walk_graph_sequences(graph, sequences, kmer, list_len, chunk_size):
    random_walk_sequences = []
    for i, node in enumerate(graph):
        sequence = sample_longest_dfs_path_sequence(graph, node, sequences, kmer)
        chunks = [sequence[i:i + chunk_size] for i in range(0, len(sequence), chunk_size)]
        if chunks and len(chunks[-1]) < 50:
            chunks.pop()
        random_walk_sequences.extend(chunks)
        if i >= list_len:
            break
    return random_walk_sequences

def fasta_parsing_func(fasta_path, kmer):
    with open(fasta_path, "rb") as f:
        data = f.read()

    dctx = zstandard.ZstdDecompressor()
    data = dctx.decompress(data)

    if data is None:
        return

    decoded_lines = data.decode()
    for s in SeqIO.parse(StringIO(decoded_lines), "fasta"):
        s = str(s.seq)
        s = "".join([c for c in s if c in ALPHABET])
        s = chop_at_first_repeated_kmer(s, kmer)
        yield s

def compute_reverse_complement(seq):
    complement_map = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join(complement_map.get(base, base) for base in reversed(seq))

def add_reverse_complements(sequences):
    reversed_complements = [compute_reverse_complement(seq) for seq in sequences]
    return np.concatenate((sequences, reversed_complements))


def process_fasta_file(file, kmer, reverse_complement, chunk_size):
    acc = os.path.basename(file).split('.')[0]
    try:
        _sequences = list(fasta_parsing_func(file, kmer))
        sequences = _sequences
        if reverse_complement:
            _sequences = add_reverse_complements(_sequences)
    except FileNotFoundError:
        return

    graph = find_overlaps_and_build_graph(_sequences, kmer)
    random_walk_sequences = random_walk_graph_sequences(graph, _sequences, kmer, len(sequences), chunk_size)

    for s in random_walk_sequences:
        yield {"sequence": s, "acc": acc}



def generate_dataset(kmer, reverse_complement, chunk_size):
    logan_data = os.path.join(logan_datasets_dir, 'data')
    fasta_files = glob.glob(os.path.join(logan_data, "*.contigs.fa.zst"))

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_fasta_file, file, kmer, reverse_complement, chunk_size): file
                   for file in fasta_files}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                           desc="Processing fasta files"):
            try:
                for entry in future.result():
                    yield entry
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train model either from scratch or from pretrained weights with specified tokenization."
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
        default=1200,
        help="Chunk size (defined when further splitting data)",
        choices=[1200, 2200]
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    kmer = args.kmer
    chunk_size = args.chunk_size
    reverse_complement = args.reverse_complement

    num = math.floor(chunk_size / 1000)
    new_dataset = Dataset.from_generator(lambda: generate_dataset(kmer, reverse_complement, chunk_size))
    logan_datasets_dir = os.path.join(generated_datasets_dir, f'logan')

    os.makedirs(logan_datasets_dir, exist_ok=True)
    if reverse_complement:
        dataset_dir = os.path.join(logan_datasets_dir, f'kmer_{kmer}_reverse')
    else:
        dataset_dir = os.path.join(logan_datasets_dir, f'kmer_{kmer}')
    dataset_dir = dataset_dir + f"_{num}k"
    os.makedirs(dataset_dir, exist_ok=True)
    split_dataset = new_dataset.train_test_split(test_size=0.2, seed=112)
    train_dataset = split_dataset['train']
    test_val_dataset = split_dataset['test']
    split_dataset_2 = test_val_dataset.train_test_split(test_size=0.5, seed=112)
    val_dataset = split_dataset_2["train"]
    test_dataset = split_dataset_2["test"]
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })

    dataset.save_to_disk(dataset_dir)

    dataset_dir = dataset_dir + f"_filtered"

    def filtered_generator(split):
        for example in split:
            if len(example["sequence"]) == chunk_size:
                yield example

    filtered_train = Dataset.from_generator(lambda: filtered_generator(dataset["train"]))
    filtered_validation = Dataset.from_generator(lambda: filtered_generator(dataset["validation"]))
    filtered_test = Dataset.from_generator(lambda: filtered_generator(dataset["test"]))

    filtered_dataset = DatasetDict({
        "train": filtered_train,
        "validation": filtered_validation,
        "test": filtered_test
    })

    filtered_dataset.save_to_disk(dataset_dir)