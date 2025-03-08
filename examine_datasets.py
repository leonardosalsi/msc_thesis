import os

from datasets import load_from_disk

from config import generated_datasets_dir

if __name__ == "__main__":
    dataset_path = os.path.join(generated_datasets_dir, 'logan/kmer_31_reverse/')
    dataset = load_from_disk(dataset_path)['train']
    for d in dataset:
        print(len(d['sequence']))