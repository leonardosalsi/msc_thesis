import os
os.environ["TRITON_ALLOW_TF32"] = "1"  # Allow using TF32 if available (safe fallback)
os.environ["USE_BF16"] = "0"
import random
import datetime
import time
from pprint import pprint

from datasets import load_dataset, Dataset
from evo2 import Evo2
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification, logging, \
    AutoConfig, EsmConfig
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from config import models_cache_dir, datasets_cache_dir, pretrained_models_cache_dir, results_dir, temp_dir
from datasets.utils.logging import disable_progress_bar, set_verbosity
from utils.util import LOGLEVEL, init_logger, get_model_by_id, get_task_by_id, get_pretrained_model_by_id
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model
import psutil
import torch.nn as nn
import torch
from evo2 import Evo2
import contextlib
torch.inference_mode = contextlib.nullcontext

import transformer_engine.pytorch as te
te.fp8_autocast(enabled=False)

def custom_data_collator(features):
    input_ids = [f["input_ids"] for f in features]
    labels = torch.tensor([f["labels"] for f in features])
    max_len = max(len(ids) for ids in input_ids)

    pad_token_id = 1

    padded_input_ids = torch.tensor([
        ids + [pad_token_id] * (max_len - len(ids))
        for ids in input_ids
    ])

    attention_mask = torch.tensor([
        [1] * len(ids) + [0] * (max_len - len(ids))
        for ids in input_ids
    ])



    return {
        "input_ids": padded_input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

class Evo2WithClassificationHead(nn.Module):
    def __init__(self, model_name, num_classes):
        super(Evo2WithClassificationHead, self).__init__()
        _model = Evo2(model_name)
        _model.model.train()
        self.evo2 = _model.model
        self.num_labels = num_classes
        self.evo2.config["use_fp8_input_projections"] = False
        self.evo2.config["use_return_dict"] = True
        self.evo2.config["inference_mode"] = False
        self.tokenizer = _model.tokenizer
        self.config = OmegaConf.load(f".archive/data/{model_name}.yml")
        if "use_return_dict" not in self.config:
            self.config.use_return_dict = True
        self.add_module("evo2", self.evo2)

        self.evo2.unembed = nn.Identity()

        self.classifier = nn.Linear(self.evo2.config["hidden_size"], num_classes)

        # Monkey-patch: Register a forward hook to clone outputs for modules with a 'scale' attribute.
        def clone_hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                return output.clone()
            elif isinstance(output, (list, tuple)):
                return type(output)(o.clone() if isinstance(o, torch.Tensor) else o for o in output)
            return output

        for name, module in self.evo2.named_modules():
            if hasattr(module, 'scale'):
                module.register_forward_hook(clone_hook)

    def forward(self, input_ids, attention_mask=None, labels=None, inputs_embeds=None, **kwargs):
        # Pass the input tensor positionally since StripedHyena.forward() doesn't accept "input_ids" as a keyword.
        if inputs_embeds is not None:
            outputs = self.evo2(inputs_embeds)
        else:
            outputs = self.evo2(input_ids)

        pooled_output = outputs[0]  # Assuming the second output is the pooled output
        pooled_output = pooled_output.mean(dim=1)
        pooled_output = pooled_output.to(torch.float32)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return {'loss': loss, 'logits': logits}

def check_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory Usage: {memory_info.rss / (1024 ** 2):.2f} MB")

def compute_metrics_mcc(eval_pred):
    """Computes Matthews correlation coefficient (MCC score) for binary classification"""
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    references = eval_pred.label_ids
    r = {'mcc_score': matthews_corrcoef(references, predictions)}
    return r

def finetune_model_by_task_mcc(logger, device, model_name, task):
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

    #logger.log(LOGLEVEL, f"Dataset {task['name']} loaded and splits created")

    """Load model and move to device"""
    #logger.log(LOGLEVEL, f"Loading model with pretrained weights.")
    model = Evo2WithClassificationHead(model_name, task["num_labels"])
    model.to(device)

    #logger.log(LOGLEVEL, f"LoRA model loaded")
    """Employ LoRA """
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, inference_mode=False, r=1, lora_alpha=32, lora_dropout=0.1,
        target_modules=["Wqkv", "out_proj"],
    )
    lora_classifier = get_peft_model(model, peft_config)
    lora_classifier.to(device)

    #logger.log(LOGLEVEL, f"Model {model_dict['name']} loaded on device {device}")

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
    tokenizer = model.tokenizer

    """Repack splits"""
    _ds_train = Dataset.from_dict({"data": train_sequences, "labels": train_labels})
    _ds_validation = Dataset.from_dict({"data": validation_sequences, "labels": validation_labels})
    _ds_test = Dataset.from_dict({"data": test_sequences, "labels": test_labels})

    """Tokenizer function"""
    def tokenize_function(examples):
        outputs = tokenizer.tokenize_batch(examples["data"])  # Tokenize the batch of sequences
        return {"input_ids": outputs}

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
    if task["taskId"] == 23:
        eval_batch_size = 32
    else:
        eval_batch_size = 64
    training_args = TrainingArguments(
        os.path.join(temp_dir, f"{model_name}-{task['alias']}{str(time.time()).replace('.', '')}"),
        remove_unused_columns=False,
        eval_strategy="steps",
        save_strategy="no",
        learning_rate=5e-4,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=100,
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
        lora_classifier,
        training_args,
        train_dataset=tokenized_train_sequences,
        eval_dataset=tokenized_validation_sequences,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_mcc,
        data_collator=custom_data_collator,
    )

    """Finetune pre-trained model"""
    _ = trainer.train()

    train_history = trainer.state.log_history
    """Get MCC score"""
    preduction_results = trainer.predict(tokenized_test_sequences)
    predictions = np.argmax(preduction_results.predictions, axis=-1)
    labels = preduction_results.label_ids

    """Apply Bootstrapping
    scores = []
    for _ in range(10000):
        idx = np.random.choice(len(predictions), size=len(predictions), replace=True)
        score = matthews_corrcoef(labels[idx], predictions[idx])
        scores.append(score)
    """

    return {'labels': labels.tolist(), 'predictions': predictions.tolist(), 'training': train_history}

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
        "model",
        type=str,
        choices=['evo2_1b_base', 'evo2_7b_base'],
        help="The model name."
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
    model_name = args.model
    task = get_task_by_id(args.taskId)

    iterations = args.samples

    filename = f"{model_name}-{task['alias']}"
    logger = init_logger()
    logger.log(LOGLEVEL, f"{model_name} on {task['alias']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.log(LOGLEVEL, f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.log(LOGLEVEL, "GPU not available. Using CPU instead.")

    eval_trained_dir = os.path.join(results_dir, f"eval_pretrained_model_{model_name}")
    os.makedirs(eval_trained_dir, exist_ok=True)

    output_file = os.path.join(eval_trained_dir, f"{task['alias']}.json")
    if os.path.exists(output_file):
        exit(0)

    all_results = []
    for i in tqdm(range(3)):
        results = finetune_model_by_task_mcc(logger, device, model_name, task)
        all_results.append(results)

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    logger.info(f"Results saved to {output_file}")
