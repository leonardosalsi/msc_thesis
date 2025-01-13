import math
import os.path

from datasets import load_dataset, Dataset, load_from_disk
from config import datasets_cache_dir, models_cache_dir
from tqdm import tqdm

import argparse

from tokenizer.OverlappingEsmTokenizer import OverlappingEsmTokenizer

TOKENIZER_BATCH_SIZE=2048

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to size down splits of the multi_genome dataset"
    )
    parser.add_argument(
        "split",
        type=str,
        help="Dataset split [train|test|validation]",
        choices=["train", "test", "validation"],
    )

    split = parser.parse_args().split

    dataset_path = os.path.join(datasets_cache_dir, "InstaDeepAI___multi_species_genomes/1kbp")
    split_path = os.path.join(dataset_path, split)
    multi_species_genomes = load_from_disk(split_path)

    tokenizer = OverlappingEsmTokenizer(
        vocab_file=os.path.join(models_cache_dir, "nt50-vocab", "vocab.txt"),
        model_max_length=2048,
    )

    tokenizer_name = type(tokenizer).__name__
    def tokenize_function(examples):
        outputs = tokenizer(examples["sequence"])
        return outputs

    tokenizer_path = os.path.join(dataset_path, "tokenized", tokenizer_name, split)

    multi_species_genomes.map(
        tokenize_function,
        batched=True,
        batch_size=TOKENIZER_BATCH_SIZE,
        num_proc=os.cpu_count(),
        cache_file_name=os.path.join(tokenizer_path, f"{tokenizer_name}.arrow")
    )



