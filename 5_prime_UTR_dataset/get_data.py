import os
import json
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


"""
Create UTR'5 Classification Dataset
From ClinVar only get benign sequences and from gnomAD both benign and pathogenic
"""
if __name__ == '__main__':
    dataset_location = "/shared/5_utr/"
    length = 1200

    gnomAD_filename = f"utr5_dataset_gnomAD{f'_{length}' if length is not None else ''}.json"
    dataset_gnomAD = load_or_generate(get_from_gnomAD.get, dataset_location, gnomAD_filename, length)

    clinvar_filename = f"utr5_dataset_clinvar{f'_{length}' if length is not None else ''}.json"
    dataset_clinvar = load_or_generate(get_from_clinvar.get, dataset_location, clinvar_filename, length)
