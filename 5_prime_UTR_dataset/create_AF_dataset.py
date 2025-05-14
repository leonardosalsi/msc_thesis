import os
import json

from datasets import Dataset, Features, DatasetDict, Value, load_from_disk
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

        def label_af(example):
            af = example["af"]
            if af <= 0.0001:
                label = 1  # rare
            elif af >= 0.05:
                label = 0  # common
            else:
                label = None  # intermediate
            return {
                "sequence": example["sequence"],
                "af": af,
                "label": label
            }

        dataset = dataset.shuffle().select(range(50000)).remove_columns(["chrom", "label"])
        print(dataset)
        dataset = dataset.map(label_af)

        # Filter to only labeled (i.e., remove AF in intermediate range)
        dataset = dataset.filter(lambda x: x["label"] is not None)
        print(dataset)

        plt.hist([x["af"] for x in dataset], bins=100, log=True)
        plt.title("AF distribution in sampled dataset")
        plt.xlabel("Allele Frequency (AF)")
        plt.ylabel("Count")
        plt.show()
        af_dataset_filename = f"5_utr_af{f'_{length}' if length is not None else ''}"
        dataset.save_to_disk(os.path.join(generated_datasets_dir, af_dataset_filename))
    else:
        dataset = load_from_disk(os.path.join(generated_datasets_dir, f"5_utr_af{f'_{length}' if length is not None else ''}"))

        for e in dataset:
            print(e['af'], e['sequence'])

        print(dataset)



