import argparse
import csv
import os
import random
import sys
import glob
from io import StringIO
from concurrent.futures import ProcessPoolExecutor
from pprint import pprint
import json
import numpy as np
import pandas as pd
import torch
import zstandard
from tqdm import tqdm
from pyfaidx import Fasta
from Bio import SeqIO
import plotnine as p9
from tqdm import tqdm
from collections import defaultdict
from config import datasets_cache_dir, results_dir, logan_datasets_dir

ALPHABET = {"A", "T", "C", "G"}
MAX_SEQ_LENGTH = 2048

LOGAN_RATIOS_FILE = os.path.join(results_dir, "logan_ratios.json")


def chop_at_first_repeated_kmer(sequence, k):
    kmers = set()
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if kmer in kmers:
            return sequence[:i + k - 1]
        kmers.add(kmer)
    return sequence  # No repeated k-mers found, return the whole sequence

def find_overlaps_and_build_graph(sequences, k_mer=3):
    min_overlap = k_mer - 1
    prefix_dict = defaultdict(list)

    for i, seq in enumerate(sequences):
        prefix_dict[seq[:min_overlap]].append(i)

    graph = defaultdict(list)

    for i, seq1 in enumerate(sequences):
        seq1_suffix = seq1[-min_overlap:]
        graph[i] = []
        for j in prefix_dict[seq1_suffix]:
            if i != j:
                graph[i].append(j)

    return graph

def dfs_paths(graph, start, path=None, all_paths=None, depth=10):
    if path is None:
        path = [start]  # Initialize the path with the starting node
    if all_paths is None:
        all_paths = []  # Initialize the list to store all paths

    # If we revisit a node in the current path, it's a cycle, so we stop
    if start in path[:-1]:
        all_paths.append(path[:-1])
        return all_paths

    # Check if the current node is a leaf (no neighbors)
    if start not in graph or not graph[start]:
        all_paths.append(path)  # Add the current path as a complete path
        return all_paths

    if len(path) >= depth:
        all_paths.append(path)
        return all_paths

    for neighbor in graph[start]:
        dfs_paths(graph, neighbor, path + [neighbor], all_paths)

    return all_paths

def random_walk_graph_sequences(graph, sequences, kmer):
    random_walk_sequences = []
    for node in graph:
        paths = dfs_paths(graph, node)
        idx = np.random.randint(len(paths))
        path = paths[idx]
        seq = sequences[path[0]] + "".join([sequences[p][kmer - 1:] for p in path[1:]])
        seq = seq[:MAX_SEQ_LENGTH]
        random_walk_sequences.append(seq)
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


def pre_select_fasta_files(sample_size=1000):
    logan_data = os.path.join(logan_datasets_dir, 'data')
    with open("./data/acc_list.txt") as filedata:
        allowed_files = filedata.read().splitlines()
    random.shuffle(allowed_files)
    allowed_files = allowed_files[:sample_size]
    to_be_processed = []
    for entry in tqdm(allowed_files, desc="Processing fasta files"):
        to_be_processed.append(os.path.join(logan_data, f"{entry}.contigs.fa.zst"))
    return to_be_processed

def get_json_content():
    if os.path.isfile(LOGAN_RATIOS_FILE):
        with open(LOGAN_RATIOS_FILE, "r") as f:
            return json.load(f)
    return []


def append_to_json_file(result):
    """Append a result dictionary to the JSON file."""
    data = get_json_content()
    data.append(result)
    with open(LOGAN_RATIOS_FILE, "w") as f:
        json.dump(data, f, indent=4)

def add_reverse_complements(sequences):
    reversed_sequences = np.array([s[::-1] for s in sequences])
    return np.concatenate((sequences, reversed_sequences))

def calculate_ratios(fasta_files, kmer, reverse_complement):
    logan_data = os.path.join(logan_datasets_dir, 'data')
    logan_data_eval = os.path.join(results_dir, 'logan_eval')
    print(logan_data_eval)
    os.makedirs(logan_data_eval, exist_ok=True)
    if reverse_complement:
        data_eval = os.path.join(logan_data_eval, f'kmer_{kmer}_reverse')
    else:
        data_eval = os.path.join(logan_data_eval, f'kmer_{kmer}')
    os.makedirs(data_eval, exist_ok=True)

    metadata_file = glob.glob(os.path.join(logan_data, "*.csv"))[0]
    metadata = pd.read_csv(metadata_file)
    metadata['kingdom'] = metadata['kingdom'].fillna('Other')
    common_names_file_path = "./data/common_names.json"
    groups_file_path = "./data/groups.json"
    with open(common_names_file_path) as json_data:
        common_names = json.load(json_data)
    with open(groups_file_path) as json_data:
        groups = json.load(json_data)

    known_files = glob.glob(os.path.join(data_eval, "*.json"))
    known_accs = [os.path.splitext(os.path.basename(x))[0] for x in known_files]

    not_found = []
    for file in tqdm(fasta_files, desc="Processing fasta files"):
        acc = os.path.basename(file).split('.')[0]
        if acc not in known_accs:
            entry = metadata.loc[metadata['acc'] == acc]
            _kingdom = entry['kingdom'].values[0]
            _kingdom = _kingdom if pd.notna(_kingdom) else 'Other'
            _organism = entry['organism'].values[0]
            _mbases = entry['mbases'].values[0]
            _mbases = _mbases if pd.notna(_mbases) else 0
            _organism_kmeans = entry['organism_kmeans'].values[0]
            _organism_kmeans = _organism_kmeans if pd.notna(_organism_kmeans) else 0

            try:
                sequences = np.array(list(fasta_parsing_func(file, kmer)))
                if reverse_complement:
                    sequences = add_reverse_complements(sequences)
            except FileNotFoundError:
                not_found.append(file)
                continue

            graph = find_overlaps_and_build_graph(sequences, kmer)
            random_walk_sequences = random_walk_graph_sequences(graph, sequences, kmer)
            result = {
                "acc": acc,
                "kingdom": _kingdom,
                "organism": _organism,
                "common_name": common_names.get(_organism, "Unknown"),
                "group": groups.get(_organism, "Unknown"),
                "mbases": int(_mbases),
                "kmeans": int(_organism_kmeans),
                "sequences": [len(x) for x in sequences],
                "random_walk_sequences": [len(x) for x in random_walk_sequences],
            }
            result_file = os.path.join(data_eval, f"{acc}.json")
            with open(result_file, "w") as f:
                json.dump(result, f, indent=4)

def get_file_content():
    if os.path.isfile(LOGAN_RATIOS_FILE):
        with open(LOGAN_RATIOS_FILE, "r") as f:
            return json.load(f)
    return None

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train model either from scratch or from pretrained weights with specified tokenization."
    )
    parser.add_argument(
        "kmer",
        type=int,
        help="Kmer length",
        choices=[31, 28, 25, 20]
    )

    parser.add_argument(
        "--reverse_complement",
        action="store_true",
        dest="reverse_complement",
        help="Also include reverse complements to graph."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fasta_files = pre_select_fasta_files(200)
    kmer = args.kmer
    reverse_complement = args.reverse_complement
    calculate_ratios(fasta_files, kmer, reverse_complement)
