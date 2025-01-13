#!/usr/bin/env python

import os
import random
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

from config import datasets_cache_dir, models_cache_dir
from tokenizer.OverlappingEsmTokenizer import OverlappingEsmTokenizer

##############################################################################
# 5) Main Script (Training Pipeline)
##############################################################################
if __name__ == "__main__":
    dataset_path = os.path.join(datasets_cache_dir, "InstaDeepAI___multi_species_genomes/1kbp")

    train__path = os.path.join(dataset_path, "train")
    test__path = os.path.join(dataset_path, "test")
    validation_path = os.path.join(dataset_path, "validation")

    multi_species_genomes_train = load_from_disk(train__path).select(range(1000))
    multi_species_genomes_test = load_from_disk(test__path)
    multi_species_genomes_validation = load_from_disk(validation_path)

    model = AutoModelForMaskedLM.from_pretrained(
        "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        cache_dir=models_cache_dir,
        trust_remote_code=True,
        local_files_only=True
    )

    print("HERE")
    tokenizer_batch_size = 2048
    tokenizer = OverlappingEsmTokenizer.from_pretrained(
        "/shared/models",
        model_max_length=2048,
    )

    print(type(tokenizer).__name__)
    def tokenize_function(examples):
        outputs = tokenizer(examples["sequence"])
        return outputs


    tokenizer_name = type(tokenizer).__name__
    print("HERE")
    dataset_train = multi_species_genomes_train.map(
        tokenize_function,
        batched=True,
        batch_size=tokenizer_batch_size,
        num_proc=os.cpu_count(),
        cache_file_name=os.path.join("/shared/datasets/InstaDeepAI___multi_species_genomes/1kbp/tokenized/OverlappingEsmTokenizer/train/", f"{tokenizer_name}.arrow")
    )
    print("HERE")
    print(dataset_train)
    exit(0)
    dataset_validation = multi_species_genomes_validation.map(
        tokenize_function,
        batched=True,
        batch_size=tokenizer_batch_size,
        num_proc=os.cpu_count(),
        cache_file_name="test_overlap_tokenized"
    )

    dataset_test = multi_species_genomes_test.map(
        tokenize_function,
        batched=True,
        batch_size=tokenizer_batch_size,
        num_proc=os.cpu_count()
    )

    print(dataset_train)

    exit(0)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )


    ###############
    # F) Training Arguments
    ###############
    training_args = TrainingArguments(
        output_dir="./overlap_pretrain_output",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        save_steps=1000,
        logging_steps=200,
        evaluation_strategy="no",  # or "steps"/"epoch" if you have a validation set
        dataloader_num_workers=4,
    )

    ###############
    # G) Trainer
    ###############
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=val_dataset,  # if you split
        data_collator=data_collator,
    )

    ###############
    # H) Train
    ###############
    trainer.train()
    print("Training complete!")
