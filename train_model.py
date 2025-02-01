#!/usr/bin/env python
import argparse
import gc
import math
import os

import json
import time

import torch
from datasets import load_from_disk, Dataset, load_dataset

from transformers import (
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling, AutoTokenizer, EsmConfig, TrainerCallback,
)
from config import models_cache_dir, pretrained_models_cache_dir, tokenizer_cache_dir, \
    datasets_cache_dir, logs_dir, generated_datasets_dir
from overrides.tokenizer.OverlappingEsmTokenizer import OverlappingEsmTokenizer
from overrides.tokenizer.OverlappingEsmTokenizerWithNSkipping import OverlappingEsmTokenizerWithNSkipping
from util import init_logger, LOGLEVEL, get_chunk_size_file_name

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
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        dest="from_scratch",
        help="Train model from scratch. Default is false."
    )
    return parser.parse_args()

def memory_safe_train_test_split(data, test_proportion=99.95):
    ratio = int(len(data)/test_proportion) #should be int
    data_train = data[ratio:,:]
    data_test =  data[:ratio,:]
    return data_train, data_test

if __name__ == "__main__":
    args = parse_args()
    selected_tokenizer = args.tokenizer
    selected_dataset = args.dataset
    chunk_size_folder_name = get_chunk_size_file_name(args.chunk_size)
    train_from_scratch = args.from_scratch
    logger = init_logger()

    num = math.floor(args.chunk_size / 1000)
    num_tokens = num * 1000
    gradient_accumulation_steps = 2 / num

    """
    Define setup name
    """
    if train_from_scratch:
        created_model_name = f"{selected_tokenizer.lower()}_{selected_dataset.lower()}_{chunk_size_folder_name}_from_scratch"
    else:
        created_model_name = f"{selected_tokenizer.lower()}_{selected_dataset.lower()}_{chunk_size_folder_name}"

    """
    Get device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.log(LOGLEVEL, f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.log(LOGLEVEL, "GPU not available. Using CPU instead.")

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
            low_cpu_mem_usage=True
        )
    model = model.to(device)
    torch.cuda.empty_cache()


    baseline_memory = torch.cuda.memory_allocated(device)


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
        tokenizer = OverlappingEsmTokenizer(
            vocab_file="model_configs/vocab.txt",
            model_max_length=2048,
            num_tokens=num_tokens
        )
    elif selected_tokenizer == "OverlappingEsmTokenizerWithNSkipping":
        tokenizer = OverlappingEsmTokenizerWithNSkipping(
            vocab_file="model_configs/vocab.txt",
            model_max_length=2048,
            num_tokens=num_tokens
        )
    else:
        raise ValueError("The specified tokenizer does not exist.")


    def tokenize_function(examples):
        outputs = tokenizer(examples['sequence'])
        return outputs

    tf = lambda examples: tokenize_function(examples)

    logger.log(LOGLEVEL, "Tokenizer loaded")

    """
    Load dataset
    """
    dataset_train = load_from_disk(os.path.join(generated_datasets_dir, selected_dataset, chunk_size_folder_name, 'train'))
    columns_to_remove = [col for col in dataset_train.column_names if col != "sequence"]
    dataset_train = dataset_train.remove_columns(columns_to_remove)
    dataset_train = dataset_train.train_test_split(test_size=0.02)
    logger.log(LOGLEVEL, "Splits created")
    train_sequences = dataset_train['train']
    validation_sequences = dataset_train['test']

    logger.log(LOGLEVEL, "Dataset loaded")
    logger.log(LOGLEVEL, f"Total training tokens: {len(train_sequences) * 1000}")
    """
    Enable retokenization per epoch
    """
    tokenized_train_sequences = train_sequences.shuffle()
    tokenized_train_sequences.set_transform(tokenize_function)

    tokenized_validation_sequences = validation_sequences.shuffle()
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
    Train model
    """
    training_args = TrainingArguments(
        output_dir=os.path.join(pretrained_models_cache_dir, created_model_name),
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=64,
        save_steps=1000,
        logging_steps=1000,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        dataloader_num_workers=2,
        gradient_checkpointing=False,
        auto_find_batch_size=True,
        logging_dir='/dev/null',
        remove_unused_columns=False,
        fp16=True,
        max_steps=600000,
        include_num_input_tokens_seen=True,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_sequences,
        eval_dataset=tokenized_validation_sequences,
        data_collator=data_collator,
    )

    logger.log(LOGLEVEL, f"Used batch size: {trainer.args.per_device_train_batch_size}")
    """
        Check VRAM
        
    sample = tokenized_train_sequences[0]
    batch_size_device = trainer.args.per_device_train_batch_size
    logger.log(LOGLEVEL,f"Batch size (Device): {batch_size_device}")
    total_vram = torch.cuda.get_device_properties(device).total_memory
    logger.log(LOGLEVEL, f"Total VRAM: {total_vram / (1024 ** 3):.2f} GB")
    logger.log(LOGLEVEL, f"Baseline VRAM usage: {baseline_memory / (1024 ** 3):.2f}/{total_vram / (1024 ** 3):.2f} GB")

    torch.cuda.empty_cache()
    batch = data_collator([sample] * batch_size_device)
    for key in batch:
        batch[key] = batch[key].to(device)
    before_fwd = torch.cuda.memory_allocated(device)
    logger.log(LOGLEVEL,
               f"VRAM usage before forward pass: {before_fwd / (1024 ** 3):.2f}/{total_vram / (1024 ** 3):.2f} GB")
    output = model(**batch)
    after_fwd = torch.cuda.memory_allocated(device)
    used_for_fwd = after_fwd - before_fwd

    # Print VRAM usage statistics

    logger.log(LOGLEVEL,
               f"VRAM usage after forward pass: {after_fwd / (1024 ** 3):.2f}/{total_vram / (1024 ** 3):.2f} GB")
    logger.log(LOGLEVEL,
               f"VRAM used for forward pass: {used_for_fwd / (1024 ** 3):.2f}/{total_vram / (1024 ** 3):.2f} GB")
    exit(0)
    """
    trainer.train()
    logger.log(LOGLEVEL, "Training complete!")
    log_history_path = os.path.join(logs_dir, f"log_history_{created_model_name}.json")
    with open(log_history_path, "w") as log_file:
        json.dump(trainer.state.log_history, log_file, indent=4)