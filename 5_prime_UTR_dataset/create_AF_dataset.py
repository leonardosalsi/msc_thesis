import os
import json
from collections import Counter
from datasets import Dataset, Features, Value, load_from_disk, concatenate_datasets
import get_from_gnomAD
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
From gnomAD get allele frequencies of SNV
"""
if __name__ == '__main__':
    length = 6000
    return_af = True
    full = False
    check_data = False

    full_af_dataset_filename = f"5_utr_af_prediction_full"
    af_dataset_filename = f"5_utr_af_prediction"

    if not check_data:
        if not os.path.exists(os.path.join(datasets_cache_dir, full_af_dataset_filename)):
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

            dataset.save_to_disk(os.path.join(datasets_cache_dir, full_af_dataset_filename))
        else:
            dataset = load_from_disk(
                os.path.join(datasets_cache_dir, full_af_dataset_filename))

        dataset = dataset.map(lambda x: {"label": get_from_gnomAD._get_label(x["af"])})
        dataset = dataset.filter(lambda x: x["label"] != None)

        rare = dataset.filter(lambda x: x["label"] == 1).shuffle().select(range(7045))
        common = dataset.filter(lambda x: x["label"] == 0)

        rare_split = rare.train_test_split(test_size=0.40)
        common_split = common.train_test_split(test_size=0.40)

        rare_train = rare_split["train"]
        rare_test = rare_split["test"]
        common_train = common_split["train"]
        common_test = common_split["test"]

        rare_train = rare_train.add_column("set", ["train"] * len(rare_train))
        rare_test = rare_test.add_column("set", ["test"] * len(rare_test))
        common_train = common_train.add_column("set", ["train"] * len(common_train))
        common_test = common_test.add_column("set", ["test"] * len(common_test))

        dataset = concatenate_datasets([rare_train, rare_test, common_train, common_test]).shuffle(seed=42).remove_columns(["chrom"])
        dataset.info.dataset_name =  f"5_utr_af_prediction"

        dataset.save_to_disk(os.path.join(generated_datasets_dir, af_dataset_filename))
    else:
        dataset = load_from_disk(os.path.join(generated_datasets_dir, af_dataset_filename))
        label_counts = Counter(example["label"] for example in dataset)
        print(label_counts)



