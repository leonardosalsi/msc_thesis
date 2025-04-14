import argparse
import csv
import glob
import os
import re
from pprint import pprint
import json
from datasets import Dataset, DatasetDict, Features, Value


from config import logan_datasets_dir, generated_datasets_dir, generator_cache_dir, logs_dir


def json_files_generator(folder_path):
    pattern = os.path.join(folder_path, '*.json')
    for file_path in glob.glob(pattern):
        file_basename = os.path.basename(file_path)
        match = re.search(r'random_walk_(\d+)\.json$', file_basename)
        if not match:
            print(f"Skipping file {file_path}: filename does not match expected pattern.")
            continue
        organism = int(match.group(1))

        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                strings_array = json.load(f)
                if isinstance(strings_array, list):
                    for string_item in strings_array:
                        if isinstance(string_item, str):
                            yield {'sequence': string_item, 'organism': organism}
                        else:
                            print(f"Skipping non-string item in {file_path}: {string_item}")
                else:
                    print(f"Skipping file {file_path}: Expected a JSON array but got {type(strings_array)}")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train model either from scratch or from pretrained weights with specified tokenization."
    )

    parser.add_argument(
        "json_files_dir",
        type=str,
        help="Folder of FASTA files to be processed",
    )

    return parser.parse_args()

if __name__ == "__main__":
        args = parse_args()
        json_files_dir = args.json_files_dir

        gen = json_files_generator(json_files_dir)
        for g in gen:
            print(g)

        dataset = Dataset.from_generator(generator=lambda: json_files_generator(json_files_dir))

