import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification, logging, \
    AutoConfig, EsmConfig
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from config import models_cache_dir, datasets_cache_dir, pretrained_models_cache_dir, results_dir
from datasets.utils.logging import disable_progress_bar, set_verbosity
from util import LOGLEVEL, init_logger, get_model_by_id, get_task_by_id, get_pretrained_model_by_id
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model
import traceback
import psutil

def check_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory Usage: {memory_info.rss / (1024 ** 2):.2f} MB")

def compute_metrics_mcc(eval_pred):
    """Computes Matthews correlation coefficient (MCC score) for binary classification"""
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    references = eval_pred.label_ids
    r={'mcc_score': matthews_corrcoef(references, predictions)}
    return r

def finetune_model_by_task_mcc(logger, device, model_dict, mode, task):
    disable_progress_bar()
    set_verbosity(logging.ERROR)
    logging.set_verbosity_error()

    """Load dataset splits"""
    dataset_train = load_dataset(
        task["repo"],
        name=task["name"],
        cache_dir=datasets_cache_dir,
        trust_remote_code=True,
        split='train'
    )

    dataset_test = load_dataset(
        task["repo"],
        name=task["name"],
        cache_dir=datasets_cache_dir,
        trust_remote_code=True,
        split='train'
    )

    logger.log(LOGLEVEL, f"Dataset {task['name']} loaded and splits created")

    """Load model and move to device"""
    logger.log(LOGLEVEL, f"Loading model with pretrained weights.")
    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(pretrained_models_cache_dir, model_dict['checkpoint']),
        cache_dir=models_cache_dir,
        num_labels=task["num_labels"],
        trust_remote_code=True,
        local_files_only=True
    )
    model = model.to(device)


    logger.log(LOGLEVEL, f"LoRA model loaded")
    """Employ LoRA """
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, inference_mode=False, r=1, lora_alpha=32, lora_dropout=0.1,
        target_modules=["query", "value"],
        # modules_to_save=["intermediate"] # modules that are not frozen and updated during the training
    )
    lora_classifier = get_peft_model(model, peft_config)
    lora_classifier.to(device)

    logger.log(LOGLEVEL, f"Model {model_dict['name']} loaded on device {device}")

    """Get corresponding feature name and load"""
    sequence_feature = task["sequence_feature"]
    label_feature = task["label_feature"]

    train_sequences = dataset_train[sequence_feature]
    train_labels = dataset_train[label_feature]

    test_sequences = dataset_test[sequence_feature]
    test_labels = dataset_test[label_feature]

    """Generate validation splits"""
    train_sequences, validation_sequences, train_labels, validation_labels = train_test_split(train_sequences, train_labels, test_size=0.05, random_state=42)

    """Load model overrides"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_dict['tokenizer'],
        cache_dir=models_cache_dir,
        trust_remote_code=True,
        local_files_only = True
    )
    logger.log(LOGLEVEL, f"Tokenizer {model_dict['name']} loaded")
    """Repack splits"""
    _ds_train = Dataset.from_dict({"data": train_sequences,'labels':train_labels})
    _ds_validation = Dataset.from_dict({"data": validation_sequences,'labels':validation_labels})
    _ds_test = Dataset.from_dict({"data": test_sequences,'labels':test_labels})

    """Tokenizer function"""
    def tokenize_function(examples):
        outputs = tokenizer(examples["data"])
        return outputs

    """Tokenize splits"""
    tokenized_train_sequences = _ds_train.map(
        tokenize_function,
        batched=True,
        remove_columns=["data"]
    )
    tokenized_validation_sequences = _ds_validation.map(
        tokenize_function,
        batched=True,
        remove_columns=["data"]
    )
    tokenized_test_sequences = _ds_test.map(
        tokenize_function,
        batched=True,
        remove_columns=["data"],
    )

    with open("len.txt", "w") as file:
        # Write tokenized_train_sequences
        file.write("Train Sequences:\n")
        for t in tokenized_train_sequences:
            file.write(f"{t}\n")

        # Write tokenized_validation_sequences
        file.write("\nValidation Sequences:\n")
        for t in tokenized_validation_sequences:
            file.write(f"{t}\n")

        # Write tokenized_test_sequences
        file.write("\nTest Sequences:\n")
        for t in tokenized_test_sequences:
            file.write(f"{t}\n")

import sys
import torch
from downstream_tasks import TASKS, MODELS
import json
from util import LOGLEVEL
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to logan model and task by MCC with optional random weights and LoRA configurations."
    )
    parser.add_argument(
        "modelId",
        type=int,
        help="The model ID (integer)."
    )
    parser.add_argument(
        "taskId",
        type=int,
        help="The task ID (integer)."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of times training and prediction process is repeated. Default is 1."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = get_pretrained_model_by_id(args.modelId)
    task = get_task_by_id(args.taskId)

    iterations = args.samples
    mode = f"-{task['alias']}"

    filename = f"{model['name'] + mode}-{task['alias']}"
    logger = init_logger()
    logger.log(LOGLEVEL, f"{model['name']}{mode} on {task['alias']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.log(LOGLEVEL, f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.log(LOGLEVEL, "GPU not available. Using CPU instead.")

    output_file = os.path.join(results_dir, 'eval_trained',f"{filename}.json")
    if os.path.exists(output_file):
        with open(output_file, "r") as file:
            results = json.load(file)

    finetune_model_by_task_mcc(logger, device, model, mode, task)