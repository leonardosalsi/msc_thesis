import os
import json
from collections import Counter

from datasets import Dataset, Features, DatasetDict, Value, load_from_disk, concatenate_datasets
import random
from sklearn.model_selection import train_test_split
import get_from_clinvar
from config import datasets_cache_dir, generated_datasets_dir

base_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(base_dir, 'data')

"""
Create UTR'5 Classification Dataset
"""
if __name__ == '__main__':
    length = 6000
    full = False
    check_data = False
    include_likely = False

    full_class_dataset_filename = f"5_utr_dataset_class_full{f'_{length}' if length is not None else ''}"
    class_dataset_filename = f"5_utr_dataset_class{f'_{length}' if length is not None else ''}"
    if not check_data:

        if not os.path.exists(os.path.join(datasets_cache_dir, full_class_dataset_filename)):
            def generator_fn():
                return get_from_clinvar.get_generator(length, include_likely)

            dataset = Dataset.from_generator(lambda: generator_fn(), features=Features({
                "sequence": Value("string"),
                "label": Value("int64"),
                "chrom": Value("string"),
                "pos": Value("int64"),
                "ref": Value("string"),
                "alt": Value("string"),
            }))

            dataset.save_to_disk(os.path.join(datasets_cache_dir, full_class_dataset_filename))
        else:
            dataset = load_from_disk(
                os.path.join(datasets_cache_dir, full_class_dataset_filename))

        pathogenic = dataset.filter(lambda x: x["label"] == 1)
        benign = dataset.filter(lambda x: x["label"] == 0)
        exit()
        dataset = concatenate_datasets([pathogenic, benign]).shuffle(seed=42).remove_columns(["chrom"])
        dataset.info.dataset_name = f"5_utr_class_{length}"

        dataset.save_to_disk(os.path.join(generated_datasets_dir, class_dataset_filename))
    else:
        dataset = load_from_disk(os.path.join(datasets_cache_dir, class_dataset_filename))
        label_counts = Counter(example["label"] for example in dataset)
        print(label_counts)



