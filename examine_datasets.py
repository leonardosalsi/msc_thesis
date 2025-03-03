import os

from datasets import load_from_disk

from config import generated_datasets_dir

if __name__ == "__main__":
    kmer = 31
    reverse_complement = None
    min_size = 2200

    if not kmer:
        print("Kmer size must be specified when using logan.")
        exit(1)
    dataset_name = f"kmer_{kmer}"
    if reverse_complement:
        dataset_name += "_reverse"
    dataset_path = os.path.join(generated_datasets_dir, 'logan/kmer_31_reverse_2k/')
    dataset = load_from_disk(dataset_path)
    print(dataset)
    dataset_path = os.path.join(generated_datasets_dir, 'multi_genome_dataset/2_2kbp/train')
    dataset = load_from_disk(dataset_path)
    print(dataset)