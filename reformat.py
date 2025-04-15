import argparse
from os.path import isfile, join, exists
from pprint import pprint

from tqdm import tqdm
from pre_train.util import print_args
import os
import json
from Bio import SeqIO

def split_fasta_by_length(fasta_path, output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(fasta_path))[0]

    data = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq_str = str(record.seq)
        data.append({"sequence": seq_str})

    out_file = os.path.join(output_dir, f"{base_name}.json")

    with open(out_file, "w") as f:
        json.dump(data, f)


def get_remaining_files(output_dir, fasta_files):
    os.makedirs(output_dir, exist_ok=True)
    full_dir = os.path.join(output_dir, '1k')
    part_dir = os.path.join(output_dir, 'part')
    known_files = []
    if os.path.isdir(full_dir):
        known_files += [f.replace('json', 'fasta') for f in os.listdir(full_dir) if f.endswith('.json') and isfile(join(full_dir, f))]
    if os.path.isdir(part_dir):
        known_files += [f.replace('json', 'fasta') for f in os.listdir(part_dir) if f.endswith('.json') and isfile(join(part_dir, f))]
    known_files = list(set(known_files))
    return [item for item in fasta_files if item not in known_files]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train model either from scratch or from pretrained weights with specified tokenization."
    )

    parser.add_argument(
        "fasta_folder_path",
        type=str,
        help="Folder of JSON files to be processed",
    )

    parser.add_argument(
        "json_out_dir",
        type=str,
        help="Folder of FASTA files to be processed",
    )

    return parser.parse_args()

if __name__ == "__main__":
        args = parse_args()
        print_args(args, "LOGAN SEPARATING ARGUMENTS")
        fasta_folder_path = args.fasta_folder_path
        json_out_dir = args.json_out_dir

        fasta_files = [f for f in os.listdir(fasta_folder_path) if f.endswith('.json') and isfile(join(fasta_folder_path, f))]
        fasta_files = get_remaining_files(json_out_dir, fasta_files)
        if not exists(json_out_dir):
            os.makedirs(json_out_dir)

        for fasta_file in tqdm(fasta_files):
            split_fasta_by_length(os.path.join(fasta_folder_path, fasta_file), output_dir=json_out_dir)