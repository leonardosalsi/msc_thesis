import os.path
from datasets import load_from_disk
from transformers import AutoTokenizer

from config import datasets_cache_dir, models_cache_dir, tokenizer_cache_dir, \
    tokenized_datasets_dir

import argparse
from overrides.tokenizer.OverlappingEsmTokenizer import OverlappingEsmTokenizer
from overrides.tokenizer.OverlappingEsmTokenizerWithNSkipping import OverlappingEsmTokenizerWithNSkipping
from util import get_chunk_size_folder_name

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

    parser.add_argument(
        "tokenizer",
        type=str,
        help="Tokenizer",
        choices=["Default", "OverlappingEsmTokenizer", "OverlappingEsmTokenizerWithNSkipping"],
    )

    parser.add_argument(
        "chunk_size",
        type=int,
        help="Chunk size (defined when further splitting data)",
    )

    split = str(parser.parse_args().split)
    selected_tokenizer = parser.parse_args().tokenizer
    chunk_size_folder_name = get_chunk_size_folder_name(parser.parse_args().chunk_size)

    dataset_path = os.path.join(datasets_cache_dir, "InstaDeepAI___multi_species_genomes/1kbp")
    split_path = os.path.join(dataset_path, split)
    multi_species_genomes = load_from_disk(split_path)

    if selected_tokenizer == "Default":
        tokenizer = AutoTokenizer.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
            cache_dir=models_cache_dir,
            trust_remote_code=True,
            local_files_only=True
        )
    elif selected_tokenizer == "OverlappingEsmTokenizer":
        tokenizer = OverlappingEsmTokenizer(
            vocab_file=os.path.join(models_cache_dir, "nt50-vocab", "vocab.txt"),
            model_max_length=2048,
        )
    elif selected_tokenizer == "OverlappingEsmTokenizerWithNSkipping":
        tokenizer = OverlappingEsmTokenizerWithNSkipping(
            vocab_file=os.path.join(models_cache_dir, "nt50-vocab", "vocab.txt"),
            model_max_length=2048,
        )
    else:
        raise ValueError("The specified tokenizer does not exist.")

    def tokenize_function(examples):
        outputs = tokenizer(examples["sequence"])
        return outputs

    tf = lambda examples: tokenize_function(examples)

    tokenizer_model_cache_path = os.path.join(tokenizer_cache_dir, selected_tokenizer, chunk_size_folder_name)
    tokenizer_model_datasets_dir = os.path.join(tokenized_datasets_dir, selected_tokenizer, chunk_size_folder_name)

    print("Beginning tokenization")
    tokenized_dataset = multi_species_genomes.map(
        tf,
        batched=False,
        num_proc=40,
        remove_columns=multi_species_genomes.column_names,
        cache_file_name=os.path.join(tokenizer_model_cache_path, split)
    )
    tokenized_dataset.save_to_disk(os.path.join(tokenizer_model_datasets_dir, split))
    print("Tokenization completed")



