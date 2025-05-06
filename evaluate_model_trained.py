import os
import random
from dataclasses import dataclass
from typing import Optional
import sys
import torch
import json

from torch.utils.data import DataLoader

from util import LOGLEVEL
from utils.PCAModel import EsmForMaskedLMPCA, EsmForSequenceClassificationPCA
from argparse_dataclass import ArgumentParser
from datasets import load_dataset, Dataset, load_from_disk
from tqdm import tqdm
from safetensors.torch import load_file
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification, logging, \
    AutoModelForMaskedLM, default_data_collator
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from config import models_cache_dir, datasets_cache_dir, pretrained_models_cache_dir, results_dir, logs_dir
from datasets.utils.logging import disable_progress_bar, set_verbosity

from utils.model import get_eval_model
from utils.tokenizer import get_eval_tokenizer
from utils.util import print_args, get_device
from util import init_logger, get_task_by_id
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model
import psutil
  
def check_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory Usage: {memory_info.rss / (1024 ** 2):.2f} MB")

def compute_metrics_mcc(eval_pred):
    """Computes Matthews correlation coefficient (MCC score) for binary classification."""
    preds = eval_pred.predictions

    if isinstance(preds, (tuple, list)):
        preds = preds[0]

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

    predictions = np.argmax(preds, axis=-1)
    references = eval_pred.label_ids
    return {'mcc_score': matthews_corrcoef(references, predictions)}

def finetune_model_by_task_mcc(args, device, task, timestamp):
    disable_progress_bar()
    set_verbosity(logging.ERROR)
    logging.set_verbosity_error()

    """Load dataset splits"""
    """5'UTR-Tasks are generated locally and must be loaded from disk"""
    if task['taskId'] in [28, 29]:
        _dataset = load_from_disk(task["repo"])
        dataset_train = _dataset['train']
        dataset_test = _dataset['test']
    else:
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
            split='test'
        )

    """Load model and move to device"""

    model, repo = get_eval_model(args, task['num_labels'], device)

    """Employ LoRA """
    modules_to_save = None
    if args.pca:
        modules_to_save = ["pca_proj", "layernorm"]
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        modules_to_save=modules_to_save
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
    random_seed = random.randint(0, 10000)
    train_sequences, validation_sequences, train_labels, validation_labels = train_test_split(train_sequences, train_labels, test_size=0.05, random_state=random_seed)

    """Load model overrides"""
    tokenizer = get_eval_tokenizer(args, repo)

    #logger.log(LOGLEVEL, f"Tokenizer {model_dict['name']} loaded")
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
    batch_size = 4
    eval_batch_size = 16
    gradient_accumulation_steps = 2
    ignore_keys = None

    if task["taskId"] in [23]:
        eval_batch_size = 8

    if args.pca:
        batch_size = int(batch_size / 2)
        eval_batch_size = int(eval_batch_size / 4)
        gradient_accumulation_steps = int(gradient_accumulation_steps * 2)
        ignore_keys = ["hidden_states", "attentions", "auxiliary_loss", "model_loss", "pca_embedding"]

    training_args = TrainingArguments(
        run_name=timestamp,
        remove_unused_columns=False,
        report_to="none",
        eval_strategy="steps",
        save_strategy="no",
        learning_rate=5e-4,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps= gradient_accumulation_steps,
        per_device_eval_batch_size= eval_batch_size,
        num_train_epochs= 2,
        logging_steps= 100,
        load_best_model_at_end=False,
        metric_for_best_model="mcc_score",
        label_names=["labels"],
        dataloader_drop_last=True,
        max_steps= 1000,
        logging_dir=logs_dir,
        disable_tqdm=True
    )

    trainer = Trainer(
        lora_classifier,
        training_args,
        train_dataset=tokenized_train_sequences,
        eval_dataset=tokenized_validation_sequences,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_mcc,
    )

    """Finetune pre-trained model"""
    _ = trainer.train(ignore_keys_for_eval=ignore_keys)

    train_history = trainer.state.log_history
    """Get MCC score"""

    prediction_results = trainer.predict(tokenized_test_sequences, ignore_keys=ignore_keys)
    predictions = np.argmax(prediction_results.predictions, axis=-1)
    labels = prediction_results.label_ids
    labels = labels.tolist()
    predictions = predictions.tolist()

    return {'labels': labels, 'predictions': predictions, 'training': train_history}

@dataclass
class EvalConfig:
    model_name: str
    checkpoint: str
    task_id: int
    samples: int = 1
    pca: bool = False
    pca_embeddings: Optional[str] = None
    pca_dims: Optional[int] = None

def parse_args():
    parser = ArgumentParser(EvalConfig)
    return parser.parse_args()

def get_output_dir(args):
    if args.task_id in [28, 29]:
        benchmark_dir = os.path.join(results_dir, f"benchmark")
    else:
        benchmark_dir = os.path.join(results_dir, f"utr5")
    os.makedirs(benchmark_dir, exist_ok=True)
    benchmark_dir = os.path.join(benchmark_dir, args.model_name)
    os.makedirs(benchmark_dir, exist_ok=True)
    eval_trained_dir = os.path.join(benchmark_dir, f"checkpoint-{int(int(args.checkpoint) / 2000)}B")
    os.makedirs(eval_trained_dir, exist_ok=True)
    return eval_trained_dir

if __name__ == "__main__":
    args = parse_args()
    timestamp = print_args(args, "BENCHMARK ARGUMENTS")

    task = get_task_by_id(args.task_id)
    logger = init_logger()
    logger.log(LOGLEVEL, f"{args.model_name} on {task['alias']}")

    device = get_device()
    eval_trained_dir = get_output_dir(args)

    logger.log(LOGLEVEL, f"Output directory: {eval_trained_dir}")

    output_file = os.path.join(eval_trained_dir ,f"{task['alias']}.json")
    logger.log(LOGLEVEL, f"Output file: {output_file}")
    if os.path.exists(output_file):
        exit(0)

    all_results = []
    for i in tqdm(range(args.samples)):
        results = finetune_model_by_task_mcc(args, device, task, timestamp)
        all_results.append(results)

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    logger.info(f"Results saved to {output_file}")