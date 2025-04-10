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

from pre_train.util import print_args


def json_to_fasta(json_file_path, use_scratch=False, min_seq_id=0.95):
    json_file = json_file_path.split('/')[-1]

    if use_scratch:
        tmpdir = os.environ.get("TMPDIR")
        if tmpdir is None:
            raise ValueError("TMPDIR environment variable is not set, but use_scratch is True.")
        scratch_json_path = os.path.join(tmpdir, 'json')
        if not os.path.exists(scratch_json_path):

            os.makedirs(scratch_json_path)
            shutil.copy(json_file_path, os.path.join(scratch_json_path, json_file))
        json_folder_path = scratch_json_path
    else:
        json_folder_path = os.path.dirname(json_file_path)

    parent_folder = os.path.abspath(os.path.join(json_folder_path, os.pardir))
    fasta_raw_folder = os.path.join(parent_folder, f'fasta_raw')
    fasta_path = os.path.join(fasta_raw_folder, f'{os.path.splitext(json_file)[0]}.fasta')
    if not exists(fasta_raw_folder):
        os.makedirs(fasta_raw_folder)
    else:
        if exists(fasta_path):
            return parent_folder, fasta_path

    if not exists(fasta_path):
        with open(os.path.join(json_folder_path, json_file), 'r') as f:
            sequences = json.load(f)
            records = [SeqRecord(Seq(seq), id=f"seq_{i}", description="") for i, seq in enumerate(sequences)]
            SeqIO.write(records, fasta_path, 'fasta')

    if use_scratch and os.path.exists(os.path.join(json_folder_path, json_file)):
        os.remove(os.path.join(json_folder_path, json_file))

    return parent_folder, fasta_path


def run_mmseqs(fasta_path, fasta_out_dir, parent_folder, use_scratch, min_seq_id, split_memory_limit):
    fasta_filtered_folder = os.path.join(parent_folder, 'fasta_filtered')
    if not exists(fasta_filtered_folder):
        os.makedirs(fasta_filtered_folder)

    fasta_base_name = os.path.splitext(os.path.basename(fasta_path))[0]
    output_prefix = os.path.join(fasta_filtered_folder, f'{fasta_base_name}')
    cmd_create = [
        'mmseqs', 'easy-cluster',
        fasta_path,
        output_prefix,
        fasta_filtered_folder,
        '--cluster-mode', '3',
        '--min-seq-id', f'{min_seq_id}',
        '--split-memory-limit', f"{split_memory_limit}G",
        '--threads', '16',
        '--cov-mode', '1'
    ]
    subprocess.run(cmd_create, check=True)
    rep_seq_path = f'{output_prefix}_rep_seq.fasta'

    if use_scratch:
        print("{} -> {}", os.path.join(fasta_filtered_folder, rep_seq_path), os.path.join(fasta_out_dir, f'{fasta_base_name}.fasta'))
        shutil.copy(os.path.join(fasta_filtered_folder, rep_seq_path), os.path.join(fasta_out_dir, f'{fasta_base_name}.fasta'))


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
        print_args(args, "LOGAN FILTERING ARGUMENTS")
        json_folder_path = args.json_folder_path
        fasta_out_dir = args.fasta_out_dir
        split_memory_limit = args.split_memory_limit
        use_scratch = args.use_scratch
        min_seq_id = args.min_seq_id
        n_smallest_files = args.n_smallest_files

        if not exists(fasta_out_dir):
            os.makedirs(fasta_out_dir)

        parent_folder, fasta_path = json_to_fasta(json_folder_path, use_scratch, min_seq_id)
        run_mmseqs(fasta_path, fasta_out_dir, parent_folder, use_scratch, min_seq_id, split_memory_limit)

