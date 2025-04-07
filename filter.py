import argparse
import shutil
import subprocess
import tempfile
from os.path import isfile, join, exists
import json
import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm


def json_to_fasta(json_folder_path, parent_folder, n_smallest_files=0, use_scratch=False):

    fasta_raw_folder = os.path.join(parent_folder, 'fasta_raw')

    if not exists(fasta_raw_folder):
        os.makedirs(fasta_raw_folder)
    json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json') and isfile(join(json_folder_path, f))]

    if n_smallest_files > 0:
        json_files.sort(key=lambda f: os.path.getsize(join(json_folder_path, f)))
        json_files = json_files[:n_smallest_files]

    fasta_paths = []
    for json_file in tqdm(json_files, desc="Converting JSON to FASTA"):
        fasta_path = os.path.join(fasta_raw_folder, f'{os.path.splitext(json_file)[0]}.fasta')
        if not exists(fasta_path):
            with open(os.path.join(json_folder_path, json_file), 'r') as f:
                sequences = json.load(f)
                records = [SeqRecord(Seq(seq), id=f"seq_{i}", description="") for i, seq in enumerate(sequences)]
                SeqIO.write(records, fasta_path, 'fasta')
        fasta_paths.append(fasta_path)
        if use_scratch:
            os.remove(os.path.join(json_folder_path, json_file))

    return fasta_paths


def run_mmseqs(fasta_paths, parent_folder, use_scratch, min_seq_id, split_memory_limit):
    fasta_filtered_folder = os.path.join(parent_folder, 'fasta_filtered')
    if not exists(fasta_filtered_folder):
        os.makedirs(fasta_filtered_folder)

    filtered_paths = []

    for fasta_file in tqdm(fasta_paths, desc="Running MMSEQS"):
        fasta_base_name = os.path.splitext(os.path.basename(fasta_file))[0]
        output_prefix = os.path.join(fasta_filtered_folder, f'{fasta_base_name}_mmseqs_out')

        cmd_create = [
            'mmseqs', 'easy-cluster',
            fasta_file,
            output_prefix,
            fasta_filtered_folder,
            '--min-seq-id', f'{min_seq_id}',
            '--split-memory-limit', f"{split_memory_limit}G"
        ]
        subprocess.run(cmd_create, check=True)

        rep_seq_path = f'{output_prefix}_rep_seq.fasta'
        filtered_paths.append(rep_seq_path)

    if use_scratch:
        print(f"Make sure to move the folder {fasta_filtered_folder} before the job ends")

    return filtered_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train model either from scratch or from pretrained weights with specified tokenization."
    )

    parser.add_argument(
        "json_folder_path",
        type=str,
        help="Folder of JSON files to be processed",
    )

    parser.add_argument(
        "fasta_out_dir",
        type=str,
        help="Folder of FASTA files to be processed",
    )

    parser.add_argument(
        "--split_memory_limit",
        type=int,
        default=8,
        help="Path to metadata file",
    )

    parser.add_argument(
        "--n_smallest_files",
        type=int,
        default=8,
        help="Path to metadata file",
    )

    parser.add_argument(
        "--min_seq_id",
        type=float,
        default=0.95,
        help="Path to metadata file",
    )

    parser.add_argument(
        "--use_scratch",
        action="store_true",
        dest="use_scratch",
        help="Pre-load everything into local scratch and load from there."
    )

    return parser.parse_args()

if __name__ == "__main__":
        args = parse_args()
        json_folder_path = args.json_folder_path
        fasta_out_dir = args.fasta_out_dir
        split_memory_limit = args.split_memory_limit
        use_scratch = args.use_scratch
        min_seq_id = args.min_seq_id
        n_smallest_files = args.n_smallest_files
        parent_folder = os.path.abspath(os.path.join(json_folder_path, os.pardir))

        if use_scratch:
            tmpdir = os.environ.get("TMPDIR")
            if tmpdir is None:
                raise ValueError("TMPDIR environment variable is not set, but use_scratch is True.")
            scratch_json_path = os.path.join(tmpdir, 'json')
            if not os.path.exists(scratch_json_path):
                shutil.copytree(json_folder_path, scratch_json_path)
            json_folder_path = scratch_json_path

        fasta_paths = json_to_fasta(json_folder_path, parent_folder, n_smallest_files, use_scratch)
        run_mmseqs(fasta_paths, parent_folder, use_scratch, min_seq_id, split_memory_limit)

