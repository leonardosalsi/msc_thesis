import argparse
import os.path
from datasets import load_from_disk
from config import datasets_cache_dir, models_cache_dir, tokenizer_cache_dir, generated_datasets_dir
from overrides.tokenizer.OverlappingEsmTokenizerWithNSkipping import OverlappingEsmTokenizerWithNSkipping
from util import get_chunk_size_folder_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train model either from scratch or from pretrained weights with specified tokenization."
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Name of the dataset",
        choices=["multi_genome_dataset"]
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
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    selected_tokenizer = args.tokenizer
    selected_dataset = args.dataset
    chunk_size_folder_name = get_chunk_size_folder_name(args.chunk_size)


    dataset_train = load_from_disk(os.path.join(generated_datasets_dir, selected_dataset, chunk_size_folder_name, 'train'))

    tokenizer = OverlappingEsmTokenizerWithNSkipping(
        vocab_file="model_configs/vocab.txt",
        model_max_length=2048,
        num_tokens=1000
    )

    tokenizer_name = type(tokenizer).__name__
    def tokenize_function(examples):
        outputs = tokenizer(examples["sequence"])
        return outputs

    tf = lambda examples: tokenize_function(examples)

    tokenizer_model_cache_path = os.path.join(tokenizer_cache_dir, tokenizer_name)

    print("Beginning tokenization")
    tokenized_dataset = dataset_train.map(
        tf,
        batched=False,
        num_proc=4,
        remove_columns=dataset_train.column_names,
        cache_file_name=os.path.join(tokenizer_model_cache_path, f"TEST.arrow"),
    )
    tokenized_dataset.save_to_disk(os.path.join(generated_datasets_dir, selected_dataset, chunk_size_folder_name, 'train_tokenized'))
