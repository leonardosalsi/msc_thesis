import argparse
import concurrent.futures
import io
import math
import os
import random
import glob
from io import StringIO
from pprint import pprint
from collections import defaultdict, Counter
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
import csv
from collections import defaultdict
from util import init_logger, LOGLEVEL

ALPHABET = {"A", "T", "C", "G"}
COMPLEMENT_MAP = str.maketrans("ATCG", "TAGC")
MAX_WORKERS = 1

def chop_at_first_repeated_kmer(sequence, k):
    kmers = set()
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if kmer in kmers:
            return sequence[:i + k - 1]
        kmers.add(kmer)
    return sequence

def group_acc_by_organism_kmeans(metadata_path):
    groups = defaultdict(list)
    with open(metadata_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            acc = row["acc"]
            organism_kmeans = row["organism_kmeans"]
            groups[organism_kmeans].append(acc)
    return groups

def find_overlaps_and_build_graph(sequences, k_mer=3, reverse_complement=False):
    min_overlap = k_mer - 1
    prefix_dict = defaultdict(list)
    graph = defaultdict(list)

    if reverse_complement:
        for i, seq in enumerate(sequences):
            if i == 0:
                continue
            prefix = seq[:min_overlap]
            prefix_dict[prefix].append(i)
            prefix_dict[compute_reverse_complement(prefix)].append(-i)
    else:
        for i, seq in enumerate(sequences):
            if i == 0:
                continue
            prefix = seq[:min_overlap]
            prefix_dict[prefix].append(i)

    for i, seq1 in enumerate(sequences):
        if i == 0:
            continue
        seq1_suffix = seq1[-min_overlap:]
        graph[i] = []
        for j in prefix_dict[seq1_suffix]:
            if i != j:
                graph[i].append(j)
    return graph


def get_connected_components(graph):
    visited = set()
    components = []

    def dfs(node, component):
        stack = [node]
        while stack:
            n = stack.pop()
            if n not in visited:
                visited.add(n)
                component.append(n)
                for neighbor in graph[n]:
                    if neighbor not in visited:
                        stack.append(neighbor)
        return component

    for node in list(graph.keys()):
        if node not in visited:
            component = dfs(node, [])
            components.append(component)

    subgraphs = []
    for comp in components:
        subgraph = {}
        for node in comp:
            # Only include neighbors that are within the same component.
            subgraph[node] = [n for n in graph[node] if n in comp]
        subgraphs.append(subgraph)

    max_component_size = max(len(component) for component in components) if components else 0
    num_singletons = sum(1 for component in components if len(component) == 1)

    # Create a dictionary: key = component size, value = count of components with that size
    size_distribution = Counter(len(component) for component in components)

    return subgraphs, components, max_component_size, num_singletons, dict(size_distribution)


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

def random_walk_graph_sequences(graph, sequences, kmer, chunk_size):
    random_walk_sequences = []
    for i, node in enumerate(graph):
        path = random_dfs_path(graph, node, depth=10000)
        sequence = sequences[path[0]] + "".join([sequences[p][kmer - 1:] for p in path[1:]])
        chunks = [sequence[i:i + chunk_size] for i in range(0, len(sequence), chunk_size)]
        if chunks and len(chunks[-1]) < 50:
            chunks.pop()
        random_walk_sequences.extend(chunks)
    return random_walk_sequences

def fasta_parsing_func_streaming(fasta_path, kmer):
    with open(fasta_path, "rb") as f:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            # Wrap the byte stream with a text wrapper so that it can be decoded on the fly.
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            # Now, SeqIO.parse can iterate over the file without loading it all at once.
            for record in SeqIO.parse(text_stream, "fasta"):
                s = str(record.seq)
                s = "".join([c for c in s if c in ALPHABET])
                s = chop_at_first_repeated_kmer(s, kmer)
                yield s

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

def process_fasta_file(file, kmer, reverse_complement, chunk_size):
    results = []
    acc = os.path.basename(file).split('.')[0]
    try:
        sequences = ["|"] + list(fasta_parsing_func_streaming(file, kmer))
    except FileNotFoundError:
        return results

    graph = find_overlaps_and_build_graph(sequences, kmer, reverse_complement)
    subgraphs, components, max_component_size, num_singletons, size_distributions = get_connected_components(graph)
    pprint(subgraphs)
    print("NUmber of sequences:", len(sequences) - 1)
    if len(components) == 1:
        print("Graph is connected.")
    else:
        print(f"Graph has {len(components)} connected components.")
    print(components[0])
    print(f"Biggest component: {max_component_size}")
    print(f"Number of singletons: {num_singletons}")
    pprint(size_distributions)
    exit(0)
    random_walk_sequences = random_walk_graph_sequences(graph, sequences, kmer, chunk_size)

    for s in random_walk_sequences:
        entry = {"sequence": s, "acc": acc}
        results.append(entry)
    return results

def generate_dataset(kmer, reverse_complement, chunk_size):
    logan_data = os.path.join(logan_datasets_dir, 'data')
    metadata_file = os.path.join(logan_data, 'metadata.csv')
    fasta_groups = group_acc_by_organism_kmeans(metadata_file)
    for _, files in fasta_groups.items():
        print(f"Processing {files}")
    exit(0)
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
    kmer = 31
    chunk_size = 1200
    reverse_complement = True

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
