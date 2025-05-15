import os
import json
from collections import Counter

from datasets import Dataset, Features, DatasetDict, Value, load_from_disk, concatenate_datasets
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import get_from_gnomAD
import get_from_clinvar
from config import datasets_cache_dir, generated_datasets_dir

base_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(base_dir, 'data')

def load_or_generate(get_fn, filename, length=None, return_af=False):
    filepath = os.path.join(DATA_PATH, filename)
    if os.path.exists(filepath):
        print(f"Loading cached dataset from: {filepath}")
        with open(filepath, "r") as f:
            return json.load(f)
    else:
        print(f"Generating dataset: {filepath}")
        return get_from_gnomAD.get_generator(length, return_af)


"""
Create UTR'5 Classification Dataset
From ClinVar only get benign sequences and from gnomAD both benign and pathogenic
"""
if __name__ == '__main__':
    length = 6000
    return_af = True
    full = False
    check_data = False

    if not check_data:
        af_dataset_filename = f"5_utr_af_full{f'_{length}' if length is not None else ''}"

        if not os.path.exists(os.path.join(datasets_cache_dir, af_dataset_filename)):
            def generator_fn():
                return get_from_gnomAD.get_generator(length, return_af)

            dataset = Dataset.from_generator(lambda: generator_fn(), features=Features({
                "sequence": Value("string"),
                "label": Value("int64"),
                "chrom": Value("string"),
                "pos": Value("int64"),
                "ref": Value("string"),
                "alt": Value("string"),
                "af": Value("float32"),
                "start": Value("int64"),
            }))

            dataset.save_to_disk(os.path.join(datasets_cache_dir, af_dataset_filename))
        else:
            dataset = load_from_disk(
                os.path.join(datasets_cache_dir, f"5_utr_af_full{f'_{length}' if length is not None else ''}"))

        rare = dataset.filter(lambda x: x["label"] == 1).shuffle().select(range(7045))
        common = dataset.filter(lambda x: x["label"] == 0)

        dataset = concatenate_datasets([rare, common]).shuffle(seed=42).remove_columns(["chrom"])
        dataset.info.dataset_name =  f"5_utr_{length}"
        af_dataset_filename = f"5_utr_af{f'_{length}' if length is not None else ''}"
        dataset.save_to_disk(os.path.join(generated_datasets_dir, af_dataset_filename))
    else:
        dataset = load_from_disk(os.path.join(generated_datasets_dir, f"5_utr_af{f'_{length}' if length is not None else ''}"))

        label_counts = Counter(example["label"] for example in dataset)
        print(label_counts)



