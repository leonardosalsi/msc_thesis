import argparse
import concurrent.futures
import math
import os
import random
import glob
from io import StringIO
from pprint import pprint

import numpy as np
import zstandard
from datasets import Dataset, DatasetDict
from Bio import SeqIO
from tqdm import tqdm
from collections import defaultdict
from config import results_dir, logan_datasets_dir, generated_datasets_dir, generator_cache_dir
from Bio.Seq import Seq
import time
import logging  # For logging retry messages

from util import init_logger, LOGLEVEL

ALPHABET = {"A", "T", "C", "G"}
COMPLEMENT_MAP = str.maketrans("ATCG", "TAGC")
MAX_WORKERS = 16

def chop_at_first_repeated_kmer(sequence, k):
    kmers = set()
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if kmer in kmers:
            return sequence[:i + k - 1]
        kmers.add(kmer)
    return sequence




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
    return seq.translate(COMPLEMENT_MAP)[::-1]

def random_dfs_path(graph, start, depth):
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

def process_fasta_file(file, kmer, reverse_complement, chunk_size):
    results = []
    acc = os.path.basename(file).split('.')[0]
    try:
        sequences = list(fasta_parsing_func(file, kmer))
        len_original_sequences = len(sequences)
        if reverse_complement:
            sequences = sequences + [compute_reverse_complement(seq) for seq in sequences]
    except FileNotFoundError:
        return results

    min_overlap = kmer - 1
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

    random_walk_sequences = []
    for i, node in enumerate(graph):
        path = random_dfs_path(graph, node, depth=10000)
        sequence = sequences[path[0]] + "".join([sequences[p][kmer - 1:] for p in path[1:]])
        chunks = [sequence[i:i + chunk_size] for i in range(0, len(sequence), chunk_size)]
        if chunks and len(chunks[-1]) < 50:
            chunks.pop()
        random_walk_sequences.extend(chunks)
        if i >= len_original_sequences:
            break

    for s in random_walk_sequences:
        entry = {"sequence": s, "acc": acc}
        results.append(entry)
    return results

def generate_dataset(kmer, reverse_complement, chunk_size):
    logan_data = os.path.join(logan_datasets_dir, 'data')
    fasta_files = glob.glob(os.path.join(logan_data, "*.contigs.fa.zst"))[0:200]
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_fasta_file, file, kmer, reverse_complement, chunk_size): file
                   for file in fasta_files}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                           desc="Processing fasta files"):
            file_name = futures[future]
            acc = file_name.split('/')[-1].split('.')[0]
            try:
                for entry in future.result():
                    yield entry
            except Exception as e:
                tqdm.write(f"Error processing {file_name}: {e}")

def save_dataset_with_retry(dataset, save_dir, num_proc, max_retries=3, delay=5):
    for attempt in range(1, max_retries + 1):
        try:
            dataset.save_to_disk(save_dir, num_proc=num_proc)
            logging.info(f"Dataset saved successfully to {save_dir} on attempt {attempt}.")
            return
        except OSError as e:
            logging.warning(f"Attempt {attempt} failed to save dataset to {save_dir}. Error: {e}. Retrying in {delay} seconds.")
            time.sleep(delay)
    raise OSError(f"Failed to save dataset to {save_dir} after {max_retries} attempts.")

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
    dataset_dir = dataset_dir + f"_{num}k"
    cache_dir = cache_dir  + f"_{num}k"
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    new_dataset = Dataset.from_generator(lambda: generate_dataset(kmer, reverse_complement, chunk_size), cache_dir=cache_dir)

    split_dataset = new_dataset.train_test_split(test_size=0.2, seed=112)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    save_dataset_with_retry(dataset, dataset_dir, num_proc=MAX_WORKERS)

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

    save_dataset_with_retry(filtered_dataset, dataset_dir, num_proc=MAX_WORKERS)
