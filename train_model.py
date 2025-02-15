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

from transformers import (
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling, AutoTokenizer, EsmConfig, TrainerCallback,
)
from config import models_cache_dir, pretrained_models_cache_dir, tokenizer_cache_dir, \
    datasets_cache_dir, logs_dir, generated_datasets_dir
from downstream_tasks import PRETRAINED_MODELS
from overrides.tokenizer.OverlappingEsmTokenizer import OverlappingEsmTokenizer
from overrides.tokenizer.OverlappingEsmTokenizerWithNSkipping import OverlappingEsmTokenizerWithNSkipping
from util import init_logger, LOGLEVEL, get_chunk_size_file_name, get_filtered_dataset_name, get_pretrained_model_by_id
import torch
torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train model either from scratch or from pretrained weights with specified tokenization."
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Name of the dataset",
        choices=["multi_genome_dataset", "logan"]
    )
    parser.add_argument(
        "tokenizer",
        type=str,
        help="Tokenizer",
        choices=["Default", "OverlappingEsmTokenizer", "OverlappingEsmTokenizerWithNSkipping"],
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        help="Chunk size (defined when further splitting data)",
    )
    parser.add_argument(
        "--shannon",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Lower and upper margin of allowed Shannon entropy (e.g., --shannon 1.4 1.8)"
    )
    parser.add_argument(
        "--gc",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Lower and upper margin of allowed GC content (e.g., --gc 0.4 0.6)"
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
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    selected_tokenizer = args.tokenizer
    selected_dataset = args.dataset
    chunk_size_folder_name = get_filtered_dataset_name(args.chunk_size, args.shannon, args.gc)
    shannon = args.shannon
    gc = args.gc
    checkpoint = args.checkpoint
    train_from_scratch = args.from_scratch
    logger = init_logger()

    num = math.floor(args.chunk_size / 1000)
    num_tokens = num * 1000
    gradient_accumulation_steps = 2 / num

    """
    Define setup name
    """
    shannon_txt = ""
    gc_txt = ""
    from_scratch_txt = ""
    if shannon is not None:
        shannon_txt = f"_sh"
    if gc is not None:
        gc_txt = f"_gc"
    if train_from_scratch:
        from_scratch_txt = "_from_scratch"

    if selected_tokenizer == "Default":
        named_tokenizer = 'default'
    elif selected_tokenizer == "OverlappingEsmTokenizer":
        named_tokenizer = 'overlap'
    elif selected_tokenizer == "OverlappingEsmTokenizerWithNSkipping":
        named_tokenizer = 'overlap'

    created_model_name = f"{named_tokenizer}_{selected_dataset.lower()}{shannon_txt}{gc_txt}{from_scratch_txt}"

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
            model_max_length=1000,
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
        outputs = tokenizer(examples['sequence'], max_length=1000, truncation=True)
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
    else:
        train_folder = "train" if selected_dataset == "multi_genome_dataset" else ""
        validation_folder = "validation" if selected_dataset == "multi_genome_dataset" else ""
        dataset_path = os.path.join(generated_datasets_dir, selected_dataset, chunk_size_folder_name, train_folder)
        validation_path = os.path.join(generated_datasets_dir, selected_dataset, chunk_size_folder_name, validation_folder)
        logger.log(LOGLEVEL, f"Train data: {dataset_path}")
        logger.log(LOGLEVEL, f"Validation data: {validation_path}")
        dataset_train = load_from_disk(dataset_path)
        dataset_validation = load_from_disk(validation_path)

    columns_to_remove = [col for col in dataset_train.column_names if col != "sequence"]
    dataset_train = dataset_train.remove_columns(columns_to_remove)
    dataset_validation = dataset_validation.remove_columns(columns_to_remove)

    logger.log(LOGLEVEL, "Dataset loaded")
    logger.log(LOGLEVEL, f"Total training tokens: {len(dataset_train) * 1000}")
    logger.log(LOGLEVEL, f"Total validation tokens: {len(dataset_validation) * 1000}")

    """
    Enable retokenization per epoch
    """

    dataset_train = dataset_train.shuffle()
    dataset_validation = dataset_validation.shuffle().select(range(6000))

    tokenized_train_sequences = dataset_train.map(tokenize_function, remove_columns='sequence', batched=True)
    tokenized_validation_sequences = dataset_validation.map(tokenize_function, remove_columns='sequence', batched=True)

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
        per_device_train_batch_size=10,
        gradient_accumulation_steps=50,
        per_device_eval_batch_size=64,
        save_steps=6000,
        logging_steps=500,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        dataloader_num_workers=4,
        gradient_checkpointing=False,
        logging_dir='/dev/null',
        remove_unused_columns=False,
        fp16=True,
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