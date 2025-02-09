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
from datasets import Dataset
from tqdm import tqdm
from pyfaidx import Fasta
from Bio import SeqIO
import plotnine as p9
from tqdm import tqdm
from collections import defaultdict
from config import datasets_cache_dir, results_dir, logan_datasets_dir, generated_datasets_dir

ALPHABET = {"A", "T", "C", "G"}
KMER = 31
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

def random_walk_graph_sequences(graph, sequences):
    random_walk_sequences = []
    for node in graph:
        paths = dfs_paths(graph, node)
        idx = np.random.randint(len(paths))
        path = paths[idx]
        seq = sequences[path[0]] + "".join([sequences[p][KMER - 1:] for p in path[1:]])
        seq = seq[:MAX_SEQ_LENGTH]
        random_walk_sequences.append(seq)
    return random_walk_sequences

def fasta_parsing_func(fasta_path):
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
        s = chop_at_first_repeated_kmer(s, KMER)
        yield s

def pre_select_fasta_files():
    logan_data = os.path.join(logan_datasets_dir, 'data')
    with open("./data/acc_list.txt") as filedata:
        allowed_files = filedata.read().splitlines()
    random.shuffle(allowed_files)
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


def generate_dataset(fasta_files):
    logan_data = os.path.join(logan_datasets_dir, 'data')
    metadata_file = glob.glob(os.path.join(logan_data, "*.csv"))[0]
    metadata = pd.read_csv(metadata_file)
    metadata['kingdom'] = metadata['kingdom'].fillna('Other')
    common_names_file_path = "./data/common_names.json"
    groups_file_path = "./data/groups.json"
    groups_labels_file_path = "./data/groups_labels.json"
    kingdom_labels_file_path = "./data/kingdom_labels.json"
    with open(common_names_file_path) as json_data:
        common_names = json.load(json_data)
    with open(groups_file_path) as json_data:
        groups = json.load(json_data)
    with open(groups_labels_file_path) as json_data:
        groups_labels = json.load(json_data)
    with open(kingdom_labels_file_path) as json_data:
        kingdom_labels = json.load(json_data)

    for file in tqdm(fasta_files, desc="Processing fasta files"):
        acc = os.path.basename(file).split('.')[0]
        metadata_entry = metadata.loc[metadata['acc'] == acc]
        _kingdom = metadata_entry['kingdom'].values[0]
        _kingdom = _kingdom if pd.notna(_kingdom) else 'Other'
        _organism = metadata_entry['organism'].values[0]
        _mbases = metadata_entry['mbases'].values[0]
        _mbases = _mbases if pd.notna(_mbases) else 0
        _organism_kmeans = metadata_entry['organism_kmeans'].values[0]
        _organism_kmeans = _organism_kmeans if pd.notna(_organism_kmeans) else 0

        try:
            sequences = np.array(list(fasta_parsing_func(file)))
        except FileNotFoundError:
            continue

        graph = find_overlaps_and_build_graph(sequences, KMER)
        random_walk_sequences = random_walk_graph_sequences(graph, sequences)

        for s in random_walk_sequences:
            entry = {
                "sequence": s,
                "kingdom_label": kingdom_labels.get(_kingdom, 4),
                "group_label": groups_labels.get(groups.get(_organism, "Unknown"), 18),
                "kmeans_label": int(_organism_kmeans),
            }
            yield entry


def get_file_content():
    if os.path.isfile(LOGAN_RATIOS_FILE):
        with open(LOGAN_RATIOS_FILE, "r") as f:
            return json.load(f)
    return None

if __name__ == "__main__":
    fasta_files = pre_select_fasta_files()
    logan_dataset_dir = os.path.join(generated_datasets_dir, 'logan')
    os.makedirs(logan_dataset_dir, exist_ok=True)
    new_train_dataset = Dataset.from_generator(lambda: generate_dataset(fasta_files))
    new_train_dataset.save_to_disk(logan_dataset_dir)
