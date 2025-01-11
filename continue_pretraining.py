#!/usr/bin/env python

import os
import random
from typing import List, Dict

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    EsmTokenizer,
)

from config import datasets_cache_dir, models_cache_dir


class OverlappingEsmTokenizer(EsmTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # anything else you need to init?

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Fully override the parent method, ignoring the parent's logic.
        We do our own logic to produce overlapping 6-mers from the entire string.
        """
        text = text.upper().replace(" ", "").replace("\n", "")
        k = 6
        tokens = []
        for i in range(len(text) - k + 1):
            tokens.append(text[i: i + k])
        return tokens



##############################################################################
# 2) Chunk the Raw DNA Sequence
##############################################################################
def chunk_dna_sequence(dna_seq: str, num_chunks: int = 6) -> List[str]:
    """
    Splits a DNA sequence into `num_chunks` parts as evenly as possible.
    If your sequence is ~6200 bases, you'll get chunks of ~1033-1034 bases each.
    """
    seq_len = len(dna_seq)
    chunk_size = seq_len // num_chunks
    remainder = seq_len % num_chunks

    chunks = []
    start = 0
    for i in range(num_chunks):
        extra = 1 if i < remainder else 0
        this_chunk_size = chunk_size + extra
        chunk = dna_seq[start : start + this_chunk_size]
        chunks.append(chunk)
        start += this_chunk_size

    return chunks


##############################################################################
# 3) Tokenize + Pad Each Chunk to a Fixed Max Length
##############################################################################
def tokenize_and_pad_chunk(chunk: str, tokenizer: PreTrainedTokenizer):
    """
    Uses the tokenizer.__call__ to get input_ids & attention_mask.
    We'll add special tokens, truncate, and pad to `max_length`.
    """
    encoding = tokenizer(
        chunk
    )
    return encoding


##############################################################################
# 4) Map function for the dataset
##############################################################################
def process_example_batch(examples, tokenizer, num_chunks=6):
    """
    For each example in the batch, we:
      1. Split the raw sequence into `num_chunks`.
      2. Tokenize each chunk (overlapping 6-mer).
      3. Return multiple rows (one per chunk).
    """
    all_input_ids = []
    all_attention_masks = []

    for seq in examples["sequence"]:
        # 1) chunk
        raw_chunks = chunk_dna_sequence(seq, num_chunks=num_chunks)
        # 2) tokenize
        for c in raw_chunks:
            encoding = tokenizer(c)
            print(encoding)
            exit(0)
            all_input_ids.append(encoding["input_ids"])
            all_attention_masks.append(encoding["attention_mask"])

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
    }


##############################################################################
# 5) Main Script (Training Pipeline)
##############################################################################
if __name__ == "__main__":
    multi_species_genomes = load_dataset(
        "InstaDeepAI/multi_species_genomes",
        cache_dir=datasets_cache_dir,
        trust_remote_code=True
    )

    multi_species_genomes_train = multi_species_genomes["train"]
    multi_species_genomes_test = multi_species_genomes["test"]
    multi_species_genomes_val = multi_species_genomes["validation"]

    model = AutoModelForMaskedLM.from_pretrained(
        "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        cache_dir=models_cache_dir,
        trust_remote_code=True,
        local_files_only=True
    )

    tokenizer = OverlappingEsmTokenizer.from_pretrained(
        "/shared/models",
        model_max_length=2048
    )


    def map_func(examples):
        return process_example_batch(examples, tokenizer, num_chunks=6)

    dataset_train = multi_species_genomes_train.map(
        map_func,
        batched=True
    )

    exit(0)
    dataset_test = multi_species_genomes_test.map(
        map_func,
        batched=True
    )

    dataset_val = multi_species_genomes_val.map(
        map_func,
        batched=True
    )

    print(dataset_train)

    ###############
    # D) Prepare Data Collator for MLM
    ###############
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
