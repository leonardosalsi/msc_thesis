#!/usr/bin/env python
import argparse
import os

import json

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling, AutoTokenizer, EsmConfig,
)
from config import models_cache_dir, pretrained_models_cache_dir, tokenizer_cache_dir, tokenized_datasets_dir, \
    datasets_cache_dir
from overrides.tokenizer.OverlappingEsmTokenizer import OverlappingEsmTokenizer
from overrides.tokenizer.OverlappingEsmTokenizerWithNSkipping import OverlappingEsmTokenizerWithNSkipping
from util import init_logger, LOGLEVEL, get_chunk_size_folder_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train model either from scratch or from pretrained weights with specified tokenization."
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

if __name__ == "__main__":
    args = parse_args()
    selected_tokenizer = args.tokenizer
    chunk_size_folder_name = get_chunk_size_folder_name(args.chunk_size)
    train_from_scratch = args.from_scratch
    logger = init_logger()
    """
    Get device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.log(LOGLEVEL, f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.log(LOGLEVEL, "GPU not available. Using CPU instead.")

    """
    Get tokenizer
    """
    if selected_tokenizer == "Default":
        tokenizer = AutoTokenizer.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
            cache_dir=models_cache_dir,
            trust_remote_code=True,
            local_files_only=True
        )
    elif selected_tokenizer == "OverlappingEsmTokenizer":
        tokenizer = OverlappingEsmTokenizer(
            vocab_file="model_configs/vocab.txt",
            model_max_length=2048,
        )
    elif selected_tokenizer == "OverlappingEsmTokenizerWithNSkipping":
        tokenizer = OverlappingEsmTokenizerWithNSkipping(
            vocab_file="model_configs/vocab.txt",
            model_max_length=2048,
        )
    else:
        raise ValueError("The specified tokenizer does not exist.")

    def tokenize_function(examples):
        outputs = tokenizer(examples["sequence"])
        return outputs

    tf = lambda examples: tokenize_function(examples)

    logger.log(LOGLEVEL, "Tokenizer loaded")

    """
    Load dataset
    """
    dataset_train = load_from_disk(os.path.join(datasets_cache_dir, 'InstaDeepAI___multi_species_genomes', chunk_size_folder_name, 'train'))

    logger.log(LOGLEVEL, "Dataset loaded")
    if train_from_scratch:
        config = EsmConfig.from_pretrained(f"model_configs/config-nucleotide-transformer-v2-50m-multi-species.json", local_files_only=True, trust_remote_code=True)
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
    logger.log(LOGLEVEL, "Model loaded")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(pretrained_models_cache_dir, "enhanced_model"),
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=59,
        auto_find_batch_size=True,
        gradient_accumulation_steps=10,
        save_steps=1000,
        logging_steps=1000,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        dataloader_num_workers=2,
        gradient_checkpointing=False,
        logging_dir='/dev/null',
        fp16=False,
        max_steps=3000
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_validation,
        data_collator=data_collator,
    )

    trainer.train()
    logger.log(LOGLEVEL, "Training complete!")
    log_history_path = os.path.join("./log", "log_history.json")
    with open(log_history_path, "w") as log_file:
        json.dump(trainer.state.log_history, log_file, indent=4)

    test_results = trainer.evaluate(eval_dataset=dataset_test)
    test_results_path = os.path.join("./log", "test_results.json")
    with open(test_results_path, "w") as test_file:
        json.dump(test_results, test_file, indent=4)
