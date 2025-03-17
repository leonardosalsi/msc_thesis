import argparse
import concurrent.futures
import math
import os
import random
import glob
import threading
from io import StringIO
from pprint import pprint

import numpy as np
import psutil
import zstandard
from datasets import Dataset, DatasetDict
from Bio import SeqIO
from tqdm import tqdm
from collections import defaultdict
from config import results_dir, logan_datasets_dir, generated_datasets_dir, generator_cache_dir
from Bio.Seq import Seq
import time
import logging
import logan_compiler
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

def random_walk_graph_sequences(graph, sequences, kmer, list_len, chunk_size):
    random_walk_sequences = []
    for i, node in enumerate(graph):
        path = random_dfs_path(graph, node, depth=10000)
        sequence = sequences[path[0]] + "".join([sequences[p][kmer - 1:] for p in path[1:]])
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
    return seq.translate(COMPLEMENT_MAP)[::-1]

def add_reverse_complements(sequences):
    return sequences + [compute_reverse_complement(seq) for seq in sequences]

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


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def benchmark_function(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


def monitor_memory(interval, stop_event, peak_memory):
    process = psutil.Process(os.getpid())
    while not stop_event.is_set():
        mem = process.memory_info().rss
        if mem > peak_memory[0]:
            peak_memory[0] = mem
        time.sleep(interval)


def benchmark_function_with_memory(func, *args, **kwargs):
    stop_event = threading.Event()
    # List to hold the peak memory usage (using a list so it's mutable)
    peak_memory = [0]
    monitor_thread = threading.Thread(target=monitor_memory, args=(0.01, stop_event, peak_memory))
    monitor_thread.start()

    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    stop_event.set()
    monitor_thread.join()
    return result, end_time - start_time, peak_memory[0]


# Example usage in your benchmark:
def benchmark(kmer, reverse_complement, chunk_size):
    import glob, os
    logan_data = os.path.join(logan_datasets_dir, 'data')
    fasta_files = glob.glob(os.path.join(logan_data, "*.contigs.fa.zst"))[0:50]

    for file in tqdm(fasta_files):
        sequences =  logan_compiler.process_fasta_file(file, kmer, reverse_complement, chunk_size)
        pprint(sequences)



# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to train model either from scratch or from pretrained weights with specified tokenization."
    )
    parser.add_argument(
        "--use_rust",
        action="store_true",
        dest="use_rust",
        help="Use the Rust-variant of the code."
    )

    use_rust = parser.parse_args().use_rust
    benchmark(31, True, 1200)
