#!/usr/bin/env python

import os
import random
from pprint import pprint
from typing import List, Dict

import torch
from datasets import load_dataset, Dataset, load_from_disk
from transformers import (
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    EsmTokenizer,
)

from config import datasets_cache_dir, models_cache_dir, pretrained_models_cache_dir, TOKENIZER_BATCH_SIZE
from tokenizer.OverlappingEsmTokenizer import OverlappingEsmTokenizer


if __name__ == "__main__":
    dataset_path = os.path.join(datasets_cache_dir, "InstaDeepAI___multi_species_genomes/1kbp")

    train__path = os.path.join(dataset_path, "train")
    test__path = os.path.join(dataset_path, "test")
    validation_path = os.path.join(dataset_path, "validation")

    multi_species_genomes_train = load_from_disk(train__path)
    multi_species_genomes_test = load_from_disk(test__path)
    multi_species_genomes_validation = load_from_disk(validation_path)

    model = AutoModelForMaskedLM.from_pretrained(
        "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        cache_dir=models_cache_dir,
        trust_remote_code=True,
        local_files_only=True
    )

    tokenizer = OverlappingEsmTokenizer.from_pretrained(
        vocab_file=os.path.join(models_cache_dir, "nt50-vocab", "vocab.txt"),
        model_max_length=2048,
    )

    def tokenize_function(examples):
        outputs = tokenizer(examples["sequence"])
        return outputs

    tokenizer_name = type(tokenizer).__name__
    tokenizer_path = os.path.join(dataset_path, "tokenized", tokenizer_name)
    tf = lambda examples: tokenize_function(examples)

    dataset_train = multi_species_genomes_train.map(
        tf,
        batched=True,
        batch_size=TOKENIZER_BATCH_SIZE,
        num_proc=124,
        remove_columns=multi_species_genomes_train.column_names,
        cache_file_name=os.path.join(tokenizer_path, f"{tokenizer_name}.arrow"),
        new_fingerprint="a4b7c9d2e5f60718"
    )

    dataset_validation = multi_species_genomes_validation.map(
        tf,
        batched=True,
        batch_size=TOKENIZER_BATCH_SIZE,
        num_proc=124,
        remove_columns=multi_species_genomes_train.column_names,
        cache_file_name=os.path.join(tokenizer_path, "validation", f"{tokenizer_name}.arrow"),
        new_fingerprint="2e8d4c6b1a3f5d9c"
    )

    dataset_test = multi_species_genomes_validation.map(
        tf,
        batched=True,
        batch_size=TOKENIZER_BATCH_SIZE,
        num_proc=124,
        remove_columns=multi_species_genomes_train.column_names,
        cache_file_name=os.path.join(tokenizer_path, "test", f"{tokenizer_name}.arrow"),
        new_fingerprint="9f1c3b4a5d6e7f80"
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(pretrained_models_cache_dir, "enhanced_model"),
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        save_steps=1000,
        logging_steps=200,
        eval_strategy="steps",
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        max_steps=100000
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_validation,
        data_collator=data_collator,
    )

    trainer.train()
    print("Training complete!")
