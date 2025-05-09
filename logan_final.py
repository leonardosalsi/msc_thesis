import argparse
import glob
import os
import re
import shutil
import json
from datasets import Dataset, DatasetDict
from config import generated_datasets_dir, generator_cache_dir

def json_files_generator(folder_path):
    pattern = os.path.join(folder_path, '*.json')
    for file_path in glob.glob(pattern):
        file_basename = os.path.basename(file_path)
        match = re.search(r'random_walk_(\d+)\.json$', file_basename)
        if not match:
            print(f"Skipping file {file_path}: filename does not match expected pattern.")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                array = json.load(f)
                if isinstance(array, list):
                    for it in array:
                        yield it
                else:
                    print(f"Skipping file {file_path}: Expected a JSON array but got {type(array)}")
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

    parser.add_argument(
        "--use_scratch",
        action="store_true",
        dest="use_scratch",
        help="Pre-load everything into local scratch and load from there."
    )

    return parser.parse_args()

if __name__ == "__main__":
        args = parse_args()
        json_files_dir = args.json_files_dir
        use_scratch = args.use_scratch

        full_dataset = Dataset.from_generator(
            generator=lambda: json_files_generator(json_files_dir),
            cache_dir=os.path.join(generator_cache_dir, 'logan'),
            num_proc=4
        )

        #full_dataset = load_dataset("json", data_dir=json_files_dir, num_proc=8)

        split_dataset = full_dataset.train_test_split(test_size=0.2)
        train_dataset = split_dataset['train']
        test_dataset = split_dataset['test']
        dataset = DatasetDict({
            "train": train_dataset,
            "validation": test_dataset
        })

        save_path = None
        final_save_path = os.path.join(generated_datasets_dir, "logan")
        if use_scratch:
            tmpdir = os.environ.get("TMPDIR")
            if tmpdir is None:
                raise ValueError("TMPDIR environment variable is not set, but use_scratch is True.")
            save_path = os.path.join(tmpdir, "logan")
        else:
            save_path = final_save_path

        dataset.save_to_disk(save_path, max_shard_size="50GB")

        if use_scratch:
            if os.path.exists(final_save_path):
                print(f"Removing existing directory at {final_save_path}")
                shutil.rmtree(final_save_path)
            print(f"Moving dataset from {save_path} to {final_save_path}")
            shutil.move(save_path, final_save_path)
            print(f"Dataset moved to {final_save_path}")


