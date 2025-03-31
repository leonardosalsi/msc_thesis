import os

from datasets import load_from_disk

from config import generated_datasets_dir

if __name__ == "__main__":
    dataset_path = os.path.join(generated_datasets_dir, 'multi_genome_dataset/1_2kbp/train')
    dataset = load_from_disk(dataset_path)
    print(dataset)