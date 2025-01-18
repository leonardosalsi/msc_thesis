import os.path
from datasets import load_from_disk
from config import datasets_cache_dir, models_cache_dir, tokenizer_cache_dir, \
    tokenized_datasets_dir

import argparse
from tokenizer.OverlappingEsmTokenizer import OverlappingEsmTokenizer

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

    fingerprint = ""
    if split == "train":
        fingerprint = "a4b7c9d2e5f60718"
    elif split == "test":
        fingerprint = "9f1c3b4a5d6e7f80"
    elif split == "validation":
        fingerprint = "2e8d4c6b1a3f5d9c"

    dataset_path = os.path.join(datasets_cache_dir, "InstaDeepAI___multi_species_genomes/1kbp-noN")
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

    tf = lambda examples: tokenize_function(examples)

    tokenizer_model_cache_path = os.path.join(tokenizer_cache_dir, tokenizer_name)
    tokenizer_model_datasets_dir = os.path.join(tokenized_datasets_dir, tokenizer_name)

    print("Beginning tokenization")
    tokenized_dataset = multi_species_genomes.map(
        tf,
        batched=False,
        num_proc=40,
        remove_columns=multi_species_genomes.column_names,
        cache_file_name=os.path.join(tokenizer_model_cache_path, f"{split}-noN", f"{split}.arrow"),
        new_fingerprint=fingerprint
    )
    tokenized_dataset.save_to_disk(os.path.join(tokenizer_model_datasets_dir, split))
    print("Tokenization completed")



