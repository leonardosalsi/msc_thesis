import argparse
import os
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
KMER = 31
MAX_SEQ_LENGTH = 2048


def chop_at_first_repeated_kmer(sequence, k):
    kmers = set()
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if kmer in kmers:
            return sequence[:i + k - 1]
        kmers.add(kmer)
    return sequence  # No repeated k-mers found, return the whole sequence


# Reconstruct assembly graph
def find_overlaps_and_build_graph(sequences, k_mer=3):
    min_overlap = k_mer - 1
    prefix_dict = defaultdict(list)

    # Precompute the suffixes
    for i, seq in enumerate(sequences):
        prefix_dict[seq[:min_overlap]].append(i)

    graph = defaultdict(list)

    # Check for overlaps
    for i, seq1 in tqdm(enumerate(sequences), total=len(sequences)):
        seq1_suffix = seq1[-min_overlap:]
        graph[i] = []
        for j in prefix_dict[seq1_suffix]:
            if i != j:
                graph[i].append(j)

    return graph


# Perform random walk on the graph
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
    # Explore each neighbor recursively, ensuring no cycles
    for neighbor in graph[start]:
        dfs_paths(graph, neighbor, path + [neighbor], all_paths)

    return all_paths


def random_walk_graph_sequences(graph, sequences):
    random_walk_sequences = []
    for node in tqdm(graph):
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
        return [[]]

    sequences = []
    decoded_lines = data.decode()  # .split("\n")

    for s in tqdm(SeqIO.parse(StringIO(decoded_lines), "fasta")):
        s = str(s.seq)
        s = "".join([c for c in s if c in ALPHABET])  # make sure only ALPHABET
        s = chop_at_first_repeated_kmer(s, KMER)
        yield s

def get_results(force_recompute = False):
    data_file = os.path.join(results_dir, "logan_examined.json")
    result_exists = os.path.isfile(data_file)
    if result_exists and not force_recompute:
        with open(data_file) as f:
            results = json.load(f)
        return results
    else:
        logan_data = os.path.join(logan_datasets_dir, 'data')
        metadata_file = glob.glob(os.path.join(logan_data) + "/*.csv")[0]
        contigs_files = glob.glob(os.path.join(logan_data) + "/*.contigs.fa.zst")
        metadata = pd.read_csv(metadata_file)
        metadata['kingdom'] = metadata['kingdom'].fillna('Other')
        kingdoms = list(set(metadata['kingdom']))
        organisms = list(set(metadata['organism']))
        kingdoms_ratios = {}
        organisms_ratios = {}

        for organism in organisms:
            organisms_ratios[organism] = []

        for kingdom in kingdoms:
            kingdoms_ratios[kingdom] = []

        for file in contigs_files:
            filename = file.split('/')[-1].split('.')[0]
            entry = metadata.loc[metadata['acc'] == filename]
            _kingdom = entry['kingdom'].values[0]
            _kingdom = _kingdom if pd.notna(_kingdom) else 'Other'
            _organism = entry['organism'].values[0]
            _mbases = entry['mbases'].values[0]
            _mbases = _mbases if pd.notna(_mbases) else 0
            _organism_kmeans = entry['organism_kmeans'].values[0]
            _organism_kmeans = _organism_kmeans if pd.notna(_organism_kmeans) else 0
            sequences = np.array(list(fasta_parsing_func(file)))
            graph = find_overlaps_and_build_graph(sequences, KMER)
            random_walk_sequences = random_walk_graph_sequences(graph, sequences)
            sequences_len = np.array([len(x) for x in sequences])
            random_walk_sequences_len = np.array([len(x) for x in random_walk_sequences])
            sequence_length_ratio = float(np.mean(sequences_len / random_walk_sequences_len))
            print(_organism_kmeans)
            kingdoms_ratios[_kingdom].append(
                {"acc": filename, "ratio": sequence_length_ratio, "mbases": int(_mbases)})
            organisms_ratios[_organism].append(
                {"acc": filename, "ratio": sequence_length_ratio, "mbases": int(_mbases), "kmeans": int(_organism_kmeans)})
        _results = {"kingdoms": kingdoms_ratios, "organisms": organisms_ratios}

        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(_results, f, indent=4)
        with open(data_file) as f:
            results = json.load(f)
        return results

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to examine and visualize data on the Logan dataset."
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        dest="force_recompute",
        help="Aggregate data from new instead of reading stored variant. Will overwrite stored variant. Default is false."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    force_recompute = args.force_recompute
    results = get_results(force_recompute)
    print(results['kingdoms'])