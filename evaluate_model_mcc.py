import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification, logging, \
    AutoConfig, EsmConfig
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from config import models_cache_dir, datasets_cache_dir
from datasets.utils.logging import disable_progress_bar, set_verbosity
from config import LOGLEVEL
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

def finetune_model_by_task_mcc(logger, device, model_name, task, random_weights, lora):
    disable_progress_bar()
    set_verbosity(logging.ERROR)
    logging.set_verbosity_error()

    if random_weights:
        mode = "-with-random-weights"
    else:
        mode = ""

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

    """Load model and move to device"""
    if random_weights:
        _model_name = model_name.split('/')[-1]
        config = EsmConfig.from_pretrained(f"{models_cache_dir}/config-{_model_name}.json", num_labels=task["num_labels"], local_files_only=True, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_config(config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=models_cache_dir,
            num_labels=task["num_labels"],
            trust_remote_code=True,
            local_files_only=True
        )
    model = model.to(device)
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4  # 4 bytes per float32
    print(f"Model size in memory: {model_size / (1024 ** 2):.2f} MB")
    if lora:
        """Employ LoRA """
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, inference_mode=False, r=1, lora_alpha=32, lora_dropout=0.1,
            target_modules=["query", "value"],
            # modules_to_save=["intermediate"] # modules that are not frozen and updated during the training
        )
        lora_classifier = get_peft_model(model, peft_config)
        lora_classifier.to(device)

    """Get corresponding feature name and load"""
    sequence_feature = task["sequence_feature"]
    label_feature = task["label_feature"]

    train_sequences = dataset_train[sequence_feature]
    train_labels = dataset_train[label_feature]

    test_sequences = dataset_test[sequence_feature]
    test_labels = dataset_test[label_feature]

    """Generate validation splits"""
    train_sequences, validation_sequences, train_labels, validation_labels = train_test_split(train_sequences, train_labels, test_size=0.05, random_state=42)

    check_memory_usage()
    """Load model tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=models_cache_dir,
        trust_remote_code=True,
        local_files_only = True
    )
    check_memory_usage()

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

    """Configure trainer"""
    batch_size = 8
    training_args = TrainingArguments(
        f"{model_name}{mode}-{task['alias']}",
        remove_unused_columns=False,
        eval_strategy="steps",
        save_strategy="no",
        learning_rate=3e-3,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps= 1,
        per_device_eval_batch_size= 64,
        num_train_epochs= 2,
        logging_steps= 100,
        load_best_model_at_end=False,
        metric_for_best_model="mcc_score",
        label_names=["labels"],
        dataloader_drop_last=True,
        max_steps= 10000,
        logging_dir='/dev/null',
        disable_tqdm=True
    )

    if lora:
        trainer = Trainer(
            lora_classifier,
            training_args,
            train_dataset= tokenized_train_sequences,
            eval_dataset= tokenized_validation_sequences,
            processing_class=tokenizer,
            compute_metrics=compute_metrics_mcc,
        )
    else:
        trainer = Trainer(
            model,
            training_args,
            train_dataset=tokenized_train_sequences,
            eval_dataset=tokenized_validation_sequences,
            processing_class=tokenizer,
            compute_metrics=compute_metrics_mcc,
        )

    """Finetune pre-trained model"""
    _ = trainer.train()
    logger.log(LOGLEVEL, trainer.state.log_history)
    """Get MCC score"""
    preduction_results = trainer.predict(tokenized_test_sequences)
    predictions = np.argmax(preduction_results.predictions, axis=-1)
    labels = preduction_results.label_ids
    """Apply Bootstrapping"""
    scores = []
    for _ in range(10000):
        idx = np.random.choice(len(predictions), size=len(predictions), replace=True)
        score = matthews_corrcoef(labels[idx], predictions[idx])
        scores.append(score)

    return {'mean': np.mean(scores), 'std': np.std(scores), 'scores': scores}

import sys
import torch
import logging as pyLogging
from downstream_tasks import TASKS, MODELS
import json
from config import LOGLEVEL
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to evaluate model and task by MCC with optional random weights and LoRA configurations."
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
        "--random-weights",
        action="store_true",
        help="Use random weights. Default is False."
    )
    parser.add_argument(
        "--no-lora",
        action="store_false",
        dest="lora",
        help="Disable LoRA. Default is True."
    )
    return parser.parse_args()

def init_logger(logfile):
    logfile = logfile.split('/')[-1]
    pyLogging.basicConfig(
        filename=f"log/{logfile}.log",
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

def get_model_by_id(modelId):
    for model in MODELS:
        if model['modelId'] == modelId:
            return model['name']
    return None

def get_task_by_id(taskId):
    for task in TASKS:
        if task['taskId'] == taskId:
            return task
    return None

if __name__ == "__main__":
    args = parse_args()
    model = get_model_by_id(args.modelId)
    task = get_task_by_id(args.taskId)

    mode = ""
    if args.random_weights:
        mode += "-with-random-weights"
    if not args.lora:
        mode += "-no-lora"

    logger = init_logger(model+mode)
    logger.log(LOGLEVEL, "Getting device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.log(LOGLEVEL, f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.log(LOGLEVEL, "GPU not available. Using CPU instead.")

    results = {}
    _model_name = model.split('/')[-1]
    output_file = f'data/{_model_name + mode}.json'

    if os.path.exists(output_file):
        with open(output_file, "r") as file:
            results = json.load(file)

    try:
        if task['alias'] in results:
            exit(0)
        logger.log(LOGLEVEL, f"{model}{mode} on {task['alias']}")
        results = finetune_model_by_task_mcc(logger, device, model, task, args.random_weights, args.lora)
        logger.log(LOGLEVEL, f"MCC of {model}{mode} on {task['alias']} => mean: {results['mean']}, std: {results['std']}")
    except Exception:
        print(traceback.format_exc())
        pass

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {output_file}")