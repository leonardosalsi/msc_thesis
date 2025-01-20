#!/usr/bin/env python

import os
import random

import json
from pprint import pprint
from typing import List, Dict

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from datasets import load_dataset, Dataset, load_from_disk
from transformers import (
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    EsmTokenizer, AutoModel,
)

import logging as pyLogging
from config import datasets_cache_dir, models_cache_dir, pretrained_models_cache_dir, tokenizer_cache_dir, LOGLEVEL, tokenized_datasets_dir
from overrides.OverlappingEsmTokenizer import OverlappingEsmTokenizer

from transformers import TrainerCallback

def init_logger():
    pyLogging.basicConfig(
        filename=f"/dev/null",
        filemode="a",
        level=LOGLEVEL,  # Log level
        format="%(message)s"
    )
    logger = pyLogging.getLogger()
    console_handler = pyLogging.StreamHandler()
    console_handler.setLevel(LOGLEVEL)
    console_handler.setFormatter(pyLogging.Formatter("%(message)s"))
    logger.addHandler(console_handler)
    return logger

if __name__ == "__main__":
    logger = init_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.log(LOGLEVEL, f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.log(LOGLEVEL, "GPU not available. Using CPU instead.")

    tokenizer = OverlappingEsmTokenizer(
        vocab_file=os.path.join(models_cache_dir, "nt50-vocab", "vocab.txt"),
        model_max_length=2048,
    )


    def tokenize_function(examples):
        outputs = tokenizer(examples["sequence"])
        return outputs


    tf = lambda examples: tokenize_function(examples)

    logger.log(LOGLEVEL, "Tokenizer loaded")

    tokenizer_name = type(tokenizer).__name__
    tokenizer_model_cache_path = os.path.join(tokenizer_cache_dir, tokenizer_name)
    tokenizer_model_datasets_dir = os.path.join(tokenized_datasets_dir, tokenizer_name)

    train__path = os.path.join(tokenizer_model_datasets_dir, "train-noN")
    test__path = os.path.join(tokenizer_model_datasets_dir, "test-noN")
    validation_path = os.path.join(tokenizer_model_datasets_dir, "validation-noN")


    def custom_collator(data):
        text_batch = [
            f'informal statement {example["generated informal statement"]} formal statement {example["formal statement"]}'
            for example in data]
        tokenized = tokenizer(text_batch, padding='longest', max_length=128, truncation=True, return_tensors='pt')
        return tokenized

    dataset_train = load_from_disk(train__path)
    dataset_test = load_from_disk(test__path)
    dataset_validation = load_from_disk(validation_path).select(range(1000))

    logger.log(LOGLEVEL, "Dataset loaded")

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
