import argparse
import csv
import os
import subprocess
import tempfile
from pprint import pprint
import json

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from datasets import Dataset, DatasetDict, Features, Value
from config import logan_datasets_dir, generated_datasets_dir, generator_cache_dir, logs_dir
import fasta_walker

def json_to_fasta_inmemory(json_path):
    with open(json_path) as f:
        sequences = json.load(f)
    return [SeqRecord(Seq(seq), id=f"seq_{i}", description="") for i, seq in enumerate(sequences)]


def run_mmseqs(records, fasta_out_dir):
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_input = os.path.join(fasta_out_dir, 'input.fasta')

        SeqIO.write(records, fasta_input, 'fasta')

        # Create and execute mmseqs commands
        cmd_create = ['mmseqs', 'easy-cluster', fasta_input, os.path.join(fasta_out_dir, 'mmseqs_out'), fasta_out_dir, '--min-seq-id',
                      '0.95']
        subprocess.run(cmd_create, check=True)

        # MMseqs outputs a representative set at: <prefix>_rep_seq.fasta
        rep_seq_fasta = os.path.join(fasta_out_dir, 'mmseqs_out_rep_seq.fasta')

        # Read the reduced set back into memory
        filtered_records = list(SeqIO.parse(rep_seq_fasta, 'fasta'))

    return filtered_records

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train model either from scratch or from pretrained weights with specified tokenization."
    )

    parser.add_argument(
        "fasta_file_path",
        type=str,
        help="Folder of FASTA files to be processed",
    )

    parser.add_argument(
        "fasta_out_dir",
        type=str,
        help="Folder of FASTA files to be processed",
    )

    parser.add_argument(
        "--metadata_file_path",
        type=str,
        help="Path to metadata file",
    )

    return parser.parse_args()

if __name__ == "__main__":
        args = parse_args()
        fasta_file_path = args.fasta_file_path
        fasta_out_dir = args.fasta_out_dir
        metadata_path = args.metadata_file_path

        fasta_records = json_to_fasta_inmemory(fasta_file_path)
        filtered_records = run_mmseqs(fasta_records, fasta_out_dir)

        print(f"Filtered from {len(fasta_records)} → {len(filtered_records)} sequences.")
