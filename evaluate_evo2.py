#!/usr/bin/env python
import os
import random
import time
import json
import logging
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import TrainingArguments, Trainer
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from peft import LoraConfig, TaskType, get_peft_model
import psutil

from config import datasets_cache_dir, pretrained_models_cache_dir, results_dir, models_cache_dir, temp_dir
from util import get_task_by_id
import os


LOGLEVEL = logging.INFO


def init_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=LOGLEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
    return logger


def check_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory Usage: {memory_info.rss / (1024 ** 2):.2f} MB")

MODEL = {
    "name": "Evo2",
    "checkpoint": "",  # If using a local checkpoint, specify its name here; otherwise, leave empty to use repo.
    "repo": "ArcInstitute/evo2_1b_base",  # Hugging Face repository name for Evo2
    "tokenizer": "ArcInstitute/evo2_1b_base",  # Assume tokenizer is provided by the Evo2 repo
    "modelId": 1
}

# --------------------------
# Load Evo2 from its Hugging Face repository
# --------------------------
from evo2 import Evo2  # Ensure you have installed the Evo2 package


# --------------------------
# Custom wrapper for Evo2 with a classification head.
# It uses Evo2's ability to return embeddings from a designated layer (e.g., "blocks.24.output")
# and applies mean pooling over the sequence before classification.
# --------------------------
class Evo2ForSequenceClassification(nn.Module):
    def __init__(self, evo2_model, num_labels):
        super().__init__()
        self.evo2 = evo2_model
        self.num_labels = num_labels
        # Assume evo2_model has an attribute hidden_dim; if not, set a fallback value.
        hidden_dim = evo2_model.hidden_dim if hasattr(evo2_model, 'hidden_dim') else 1024
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Request embeddings from the final transformer block.
        # (Adjust "blocks.24.output" if your Evo2 variant has a different number of layers.)
        outputs, embeddings = self.evo2(input_ids, return_embeddings=True, layer_names=["blocks.24.output"])
        hidden_states = embeddings["blocks.24.output"]  # Shape: (batch, seq_len, hidden_dim)
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1, keepdim=True)
            pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / lengths
        else:
            pooled = hidden_states.mean(dim=1)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fn = nn.BCEWithLogitsLoss()
                loss = loss_fn(logits.view(-1), labels.float())
            else:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        if loss is not None:
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


# --------------------------
# Metrics computation (Matthews Correlation Coefficient)
# --------------------------
def compute_metrics_mcc(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    references = eval_pred.label_ids
    mcc = matthews_corrcoef(references, predictions)
    return {"mcc_score": mcc}


# --------------------------
# Main function: Fine-tune Evo2 on a given task using LoRA and evaluate it.
# --------------------------
def finetune_model_by_task_mcc(logger, device, model_dict, mode, task):
    # Load dataset splits (using Hugging Face datasets)
    dataset_train = load_dataset(
        task["repo"],
        name=task["name"],
        cache_dir=datasets_cache_dir,
        split='train'
    )
    dataset_test = load_dataset(
        task["repo"],
        name=task["name"],
        cache_dir=datasets_cache_dir,
        split='train'
    )

    # Load Evo2 weights from Hugging Face.
    # If a checkpoint is provided, load from local pretrained_models_cache_dir; otherwise, use the repo.
    path = os.path.join(pretrained_models_cache_dir, model_dict['checkpoint']) if model_dict['checkpoint'] != '' else \
    model_dict['repo']

    evo2_model = Evo2("evo2_1b_base")
    for param in evo2_model.parameters():
        param.requires_grad = False

    num_labels = task["num_labels"]
    model = Evo2ForSequenceClassification(evo2_model, num_labels)
    model = model.to(device)

    # Apply LoRA on top of Evo2's attention modules.
    # Adjust the target module names ("query", "value") based on Evo2's internal architecture.
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=1,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"]
    )
    model = get_peft_model(model, lora_config)
    model.to(device)

    tokenizer = evo2_model.tokenizer
    # Ensure the tokenizer has a pad token.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset splits.
    sequence_feature = task["sequence_feature"]
    label_feature = task["label_feature"]
    train_sequences = dataset_train[sequence_feature]
    train_labels = dataset_train[label_feature]
    test_sequences = dataset_test[sequence_feature]
    test_labels = dataset_test[label_feature]

    # Create a validation split from the training data.
    random_seed = random.randint(0, 10000)
    train_sequences, validation_sequences, train_labels, validation_labels = train_test_split(
        train_sequences, train_labels, test_size=0.05, random_state=random_seed
    )

    _ds_train = Dataset.from_dict({"data": train_sequences, "labels": train_labels})
    _ds_validation = Dataset.from_dict({"data": validation_sequences, "labels": validation_labels})
    _ds_test = Dataset.from_dict({"data": test_sequences, "labels": test_labels})

    def tokenize_function(examples):
        return tokenizer(examples["data"], padding="max_length", truncation=True)

    tokenized_train = _ds_train.map(tokenize_function, batched=True, remove_columns=["data"])
    tokenized_validation = _ds_validation.map(tokenize_function, batched=True, remove_columns=["data"])
    tokenized_test = _ds_test.map(tokenize_function, batched=True, remove_columns=["data"])

    # Configure training arguments.
    batch_size = 8
    eval_batch_size = 64 if task["taskId"] != 23 else 32
    training_args = TrainingArguments(
        output_dir=os.path.join(temp_dir,
                                f"{model_dict['name']}{mode}-{task['alias']}{str(time.time()).replace('.', '')}"),
        remove_unused_columns=False,
        evaluation_strategy="steps",
        save_strategy="no",
        learning_rate=5e-4,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=3,  # Adjust epochs as needed
        logging_steps=100,
        load_best_model_at_end=False,
        metric_for_best_model="mcc_score",
        label_names=["labels"],
        dataloader_drop_last=True,
        max_steps=10000,
        logging_dir='./log',
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_mcc,
    )

    # Fine-tune the model.
    trainer.train()
    train_history = trainer.state.log_history

    # Evaluate on the test set.
    prediction_results = trainer.predict(tokenized_test)
    predictions = np.argmax(prediction_results.predictions, axis=-1)
    labels_out = prediction_results.label_ids

    return {'labels': labels_out.tolist(), 'predictions': predictions.tolist(), 'training': train_history}


# --------------------------
# Argument parsing and main entry point.
# --------------------------
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Benchmark Evo2 model on downstream classification tasks with LoRA."
    )
    # Here we only need taskId since Evo2 is our fixed model.
    parser.add_argument("taskId", type=int, help="Task ID (integer).")
    parser.add_argument("--samples", type=int, default=1, help="Number of training iterations (samples).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    task = get_task_by_id(args.taskId)
    if task is None:
        print(f"Task with ID {args.taskId} not found in TASKS dictionary.")
        exit(1)

    logger = init_logger()
    logger.info(f"Benchmarking Evo2 on task: {task['alias']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("GPU not available. Using CPU instead.")

    eval_trained_dir = os.path.join(results_dir, "eval_pretrained_model_Evo2")
    os.makedirs(eval_trained_dir, exist_ok=True)
    output_file = os.path.join(eval_trained_dir, f"{task['alias']}.json")
    if os.path.exists(output_file):
        logger.info(f"Results already exist at {output_file}. Exiting.")
        exit(0)

    all_results = []
    for i in tqdm(range(args.samples)):
        results = finetune_model_by_task_mcc(logger, device, MODEL, f"-{task['alias']}", task)
        all_results.append(results)

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    logger.info(f"Results saved to {output_file}")
