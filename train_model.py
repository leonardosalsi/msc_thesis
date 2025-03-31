#!/usr/bin/env python
import argparse
import gc
import math
import os

import json
import random
import time

import torch
from datasets import load_from_disk, Dataset, load_dataset

from overrides.tokenizer.OverlappingTokenizer import OverlappingTokenizer

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False
from transformers import (
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling, AutoTokenizer, EsmConfig, TrainerCallback,
)

from config import models_cache_dir, pretrained_models_cache_dir, tokenizer_cache_dir, \
    datasets_cache_dir, logs_dir, generated_datasets_dir
from downstream_tasks import PRETRAINED_MODELS
from util import init_logger, LOGLEVEL, get_chunk_size_file_name, get_filtered_dataset_name, get_pretrained_model_by_id
import torch
import torch.profiler

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train model either from scratch or from pretrained weights with specified tokenization."
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Name of the dataset",
        choices=["multi_genome_species", "logan"]
    )
    parser.add_argument(
        "tokenizer",
        type=str,
        help="Tokenizer",
        choices=["Default", "Overlapping"],
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        help="Chunk size (defined when further splitting data)",
    )
    parser.add_argument(
        "--kmer",
        type=int,
        help="Kmer size (only when using logan)",
    )
    parser.add_argument(
        "--reverse_complement",
        action="store_true",
        dest="reverse_complement",
        help="Use dataset generated with reverse complement (only when using logan)."
    )
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        dest="from_scratch",
        help="Train model from scratch. Default is false."
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        nargs=1,
        metavar=("modelId in PRETRAINED_MODELS"),
        help="Use checkpoint instead of pretrained weights."
    )
    parser.add_argument(
        "--freeze",
        type=float,
        help="Freeze a percentual number of layers (between 0.1 and 0.9)",
    )
    return parser.parse_args()

def print_gpu_memory_usage(device):
    """Print the currently allocated and reserved GPU memory (in MB)."""
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        print(f"GPU Memory allocated: {allocated:.2f} MB")
        print(f"GPU Memory reserved: {reserved:.2f} MB")
    else:
        print("Running on CPU; GPU memory usage is not applicable.")


if __name__ == "__main__":
    args = parse_args()
    selected_tokenizer = args.tokenizer
    selected_dataset = args.dataset

    checkpoint = args.checkpoint
    train_from_scratch = args.from_scratch
    kmer = args.kmer
    reverse_complement = args.reverse_complement
    freeze = args.freeze
    logger = init_logger()

    chunk_size = args.chunk_size
    if not chunk_size:
        chunk_size = 2200
    chunk_size_folder_name = get_filtered_dataset_name(chunk_size, args.shannon, args.gc)

    num = math.floor(chunk_size / 1000)
    num_tokens = num * 1000

    kb = ""
    if num != 1:
        # 2kbp -> 2000 tokens per sequence
        kb = f"_{num}kb"
        train_batch_size = 2
        if selected_tokenizer == "Default":
            eval_batch_size = 32
        else:
            eval_batch_size = 16
        gradient_accumulation_steps = 125
    else:
        # 1kbp -> 1000 tokens per sequence
        train_batch_size = 10
        gradient_accumulation_steps = 50
        eval_batch_size = 64

    """
    Define setup name
    """
    from_scratch_txt = ""
    freeze_txt = ""

    if train_from_scratch:
        from_scratch_txt = "_from_scratch"

    if selected_tokenizer == "Default":
        named_tokenizer = 'default'
    elif selected_tokenizer == "Overlapping":
        named_tokenizer = 'overlap'

    if freeze is not None:
        freeze_txt = "_freeze"

    created_model_name = f"{named_tokenizer}_{selected_dataset.lower()}{kb}{from_scratch_txt}{freeze_txt}"

    """
    Get device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.log(LOGLEVEL, f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.log(LOGLEVEL, "GPU not available. Using CPU instead.")
    torch.cuda.empty_cache()
    """
    Load model
    """
    if train_from_scratch:
        config = EsmConfig.from_pretrained(f"model_configs/config-nucleotide-transformer-v2-50m-multi-species.json",
                                           local_files_only=True, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_config(config)
    else:
        model = AutoModelForMaskedLM.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
            cache_dir=models_cache_dir,
            trust_remote_code=True,
            local_files_only=True,
        )

    if freeze is not None:
        n_layers_to_freeze = int(len(model.esm.encoder.layer) * freeze)
        for idx, layer in enumerate(model.esm.encoder.layer):
            if idx < n_layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

    model.to(device)
    torch.compiler.cudagraph_mark_step_begin()
    model = torch.compile(model)
    logger.log(LOGLEVEL, "Model loaded")

    """
    Load tokenizer
    """
    if selected_tokenizer == "Default":
        tokenizer = AutoTokenizer.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
            model_max_length=2048,
            cache_dir=models_cache_dir,
            remove_columns=['sequence'],
            trust_remote_code=True,
            local_files_only=True
        )
    elif selected_tokenizer == "OverlappingEsmTokenizer":
        tokenizer = OverlappingTokenizer(
            vocab_file="model_configs/vocab.txt",
            model_max_length=2048,
            num_tokens=num_tokens
        )
    else:
        raise ValueError("The specified tokenizer does not exist.")

    def tokenize_function(examples):
        outputs = tokenizer(examples['sequence'], max_length=num_tokens, truncation=True)
        return outputs

    tf = lambda examples: tokenize_function(examples)

    logger.log(LOGLEVEL, "Tokenizer loaded")

    """
    Load dataset
    """
    if selected_tokenizer == "Default":
        dataset_train = load_dataset(
            "InstaDeepAI/multi_species_genomes",
            cache_dir=datasets_cache_dir,
            split='train',
            trust_remote_code=True
        )
        dataset_validation = load_dataset(
            "InstaDeepAI/multi_species_genomes",
            cache_dir=datasets_cache_dir,
            split='validation',
            trust_remote_code=True
        )
    elif selected_dataset == "logan":
        if not kmer:
            print("Kmer size must be specified when using logan.")
            exit(1)
        dataset_name = f"kmer_{kmer}"
        if reverse_complement:
            dataset_name += "_reverse"
        dataset_name += f"_{num}k"
        dataset_path = os.path.join(generated_datasets_dir, selected_dataset, dataset_name)
        dataset_train = load_from_disk(dataset_path)['train']
        validation_path = os.path.join(generated_datasets_dir, "multi_genome_dataset", f"{num}_2kbp", "validation")
        dataset_validation = load_from_disk(validation_path)
    elif selected_dataset == "logan_full":
        if not kmer:
            print("Kmer size must be specified when using logan.")
            exit(1)
        dataset_name = f"kmer_{kmer}"
        if reverse_complement:
            dataset_name += "_reverse"
        dataset_path = os.path.join(generated_datasets_dir, 'logan', dataset_name)
        dataset_train = load_from_disk(dataset_path)['train']
        validation_path = os.path.join(generated_datasets_dir, "multi_genome_dataset", f"2_2kbp", "validation")
        dataset_validation = load_from_disk(validation_path)
    else:
        train_folder = "train" if selected_dataset == "multi_genome_dataset" else ""
        validation_folder = "validation" if selected_dataset == "multi_genome_dataset" else ""
        dataset_path = os.path.join(generated_datasets_dir, selected_dataset, chunk_size_folder_name, train_folder)
        validation_path = os.path.join(generated_datasets_dir, selected_dataset, chunk_size_folder_name, "validation")
        logger.log(LOGLEVEL, f"Train data: {dataset_path}")
        logger.log(LOGLEVEL, f"Validation data: {validation_path}")
        dataset_train = load_from_disk(dataset_path)
        dataset_validation = load_from_disk(validation_path)

    columns_to_remove = [col for col in dataset_train.column_names if col != "sequence"]
    dataset_train = dataset_train.remove_columns(columns_to_remove)
    columns_to_remove = [col for col in dataset_validation.column_names if col != "sequence"]
    dataset_validation = dataset_validation.remove_columns(columns_to_remove)

    logger.log(LOGLEVEL, "Dataset loaded")
    logger.log(LOGLEVEL, f"Total training tokens: {len(dataset_train) * 1000}")
    logger.log(LOGLEVEL, f"Total validation tokens: {len(dataset_validation) * 1000}")

    """
    Enable retokenization per epoch
    """

    tokenized_train_sequences = dataset_train.shuffle()
    tokenized_train_sequences.set_transform(tokenize_function)

    tokenized_validation_sequences = dataset_validation.shuffle()
    tokenized_validation_sequences = tokenized_validation_sequences.select(range(5000))
    tokenized_validation_sequences.set_transform(tokenize_function)

    """
    Instantiate collator
    """
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    """
    Check if resume possible
    """
    model_path = os.path.join(pretrained_models_cache_dir, created_model_name)
    dir_exists = os.path.isdir(model_path)
    if dir_exists:
        if len(os.listdir(model_path)) == 0:
            resume_from_checkpoint = False
        else:
            resume_from_checkpoint = True
    else:
        resume_from_checkpoint  = False

    """
    Train model
    """
    training_args = TrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=True,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=eval_batch_size,
        save_steps=6000,
        logging_steps=500,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        dataloader_num_workers=4,
        gradient_checkpointing=False,
        logging_dir='/dev/null',
        remove_unused_columns=False,
        bf16=True,
        max_steps=12000,
        include_num_input_tokens_seen=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_sequences,
        eval_dataset=tokenized_validation_sequences,
        data_collator=data_collator,
    )

    _ = trainer.train()

    logger.log(LOGLEVEL, "Training complete!")
    log_history_path = os.path.join(logs_dir, f"log_history_{created_model_name}.json")
    with open(log_history_path, "w") as log_file:
        json.dump(trainer.state.log_history, log_file, indent=4)