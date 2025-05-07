import multiprocessing
import os
import json

from datasets import Dataset, Features, DatasetDict, Value
import random
from sklearn.model_selection import train_test_split

import get_from_gnomAD
import get_from_clinvar

def load_or_generate(get_fn, dataset_location, filename, length=None):
    filepath = os.path.join(dataset_location, filename)
    if os.path.exists(filepath):
        print(f"Loading cached dataset from: {filepath}")
        with open(filepath, "r") as f:
            return json.load(f)
    else:
        print(f"Generating dataset: {filepath}")
        return get_fn(dataset_location, filename, length)

def check_structure_consistency(dataset1, dataset2):
    if not dataset1 or not dataset2:
        return False
    keys1 = set(dataset1[0].keys())
    keys2 = set(dataset2[0].keys())
    return keys1 == keys2


"""
Create UTR'5 Classification Dataset
From ClinVar only get benign sequences and from gnomAD both benign and pathogenic
"""
if __name__ == '__main__':
    dataset_location = "/shared/5_utr/"
    length = 2200
    full = True

    gnomAD_filename = f"utr5_dataset_gnomAD{f'_{length}' if length is not None else ''}.json"
    dataset_gnomAD = load_or_generate(get_from_gnomAD.get, dataset_location, gnomAD_filename, length)

    clinvar_filename = f"utr5_dataset_clinvar{f'_{length}' if length is not None else ''}.json"
    dataset_clinvar = load_or_generate(get_from_clinvar.get, dataset_location, clinvar_filename, length)

    if check_structure_consistency(dataset_gnomAD, dataset_clinvar):
        combined_dataset = dataset_gnomAD + dataset_clinvar

        benign = [d for d in combined_dataset if d["label"] == 0]
        pathogenic = [d for d in combined_dataset if d["label"] == 1]

        """
        Get length distribution
        """
        if length is None:
            len_benign = []
            len_pathogenic = []

            for p in pathogenic:
                len_pathogenic.append(len(p['sequence']))
            for b in benign:
                len_benign.append(len(b['sequence']))

            with open(os.path.join(dataset_location, f"sequence_lengths{f'_{length}' if length is not None else ''}.json"),
                      "w") as f:
                json.dump({"benign": len_benign, "pathogenic": len_pathogenic}, f)


        """
        Downstream Tasks have comparatively small datasets, downsize results
        """
        random.shuffle(benign)
        random.shuffle(pathogenic)

        if not full:
            perc = len(benign) /len(pathogenic)
            full_size = 200000

            size_benign = int(full_size * perc)
            size_pathogenic = full_size - size_benign

            benign = benign[:size_benign]
            pathogenic = pathogenic[:size_pathogenic]

        print(f"Total: {len(combined_dataset)}, Benign: {len(benign)}, Pathogenic: {len(pathogenic)}")

        benign_train, benign_test = train_test_split(benign, test_size=0.25)
        pathogenic_train, pathogenic_test = train_test_split(pathogenic, test_size=0.25)

        train_data = benign_train + pathogenic_train
        test_data = benign_test + pathogenic_test

        random.shuffle(train_data)
        random.shuffle(test_data)

        features = Features({
            "sequence": Value("string"),
            "label": Value("int64"),
            "chrom": Value("string"),
            "pos": Value("int64"),
            "ref": Value("string"),
            "alt": Value("string"),
        })

        def gen(data):
            for entry in data:
                yield entry


        train_dataset = Dataset.from_generator(lambda: gen(train_data), features=features)
        test_dataset = Dataset.from_generator(lambda: gen(test_data), features=features)

        dataset = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })

        dataset.save_to_disk(os.path.join(dataset_location, f"5_utr_classification{f'_fixed_{length}' if length is not None else ''}{'_full' if full else ''}"))



