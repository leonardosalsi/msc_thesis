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

def compute_metrics_mcc(eval_pred):
    """Computes Matthews correlation coefficient (MCC score) for binary classification"""
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    references = eval_pred.label_ids
    r={'mcc_score': matthews_corrcoef(references, predictions)}
    return r

def finetune_model_by_task_mcc(logger, device, model_name, task, random_weights):
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
        config = EsmConfig.from_pretrained(f"{models_cache_dir}config-{_model_name}.json", local_files_only=True, trust_remote_code=True)
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

    """Get corresponding feature name and load"""
    sequence_feature = task["sequence_feature"]
    label_feature = task["label_feature"]

    train_sequences = dataset_train[sequence_feature]
    train_labels = dataset_train[label_feature]

    test_sequences = dataset_test[sequence_feature]
    test_labels = dataset_test[label_feature]

    """Generate validation splits"""
    train_sequences, validation_sequences, train_labels, validation_labels = train_test_split(train_sequences, train_labels, test_size=0.05, random_state=42)

    """Load model tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=models_cache_dir,
        trust_remote_code=True,
        local_files_only = True
    )

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
    batch_size = 32
    training_args = TrainingArguments(
        f"{model_name}{mode}_finetuned_{task['alias']}",
        remove_unused_columns=False,
        eval_strategy="steps",
        save_strategy="steps",
        learning_rate=1e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps= 1,
        per_device_eval_batch_size= 64,
        num_train_epochs= 2,
        logging_steps= 100,
        load_best_model_at_end=True,  # Keep the best model according to the evaluation
        metric_for_best_model="mcc_score",
        label_names=["labels"],
        dataloader_drop_last=True,
        max_steps= 1000,
        logging_dir='/dev/null',
        disable_tqdm=True
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset= tokenized_train_sequences,
        eval_dataset= tokenized_validation_sequences,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_mcc,
    )

    """Finetune pre-trained model"""
    train_results = trainer.train()
    logger.log(LOGLEVEL, trainer.state.log_history)
    """Get MCC score"""
    preduction_results = trainer.predict(tokenized_test_sequences)
    predictions = np.argmax(preduction_results.predictions, axis=-1)
    labels = preduction_results.label_ids
    mcc = preduction_results.metrics['test_mcc_score']

    scores = []

    for _ in range(10000):
        idx = np.random.choice(len(predictions), size=len(predictions), replace=True)
        score = matthews_corrcoef(predictions[idx], labels[idx])
        scores.append(score)

    return {'mean': np.mean(scores), 'std': np.std(scores), 'scores': scores}

import sys
import torch
import logging as pyLogging
from downstream_tasks import TASKS as tasks, MODELS as model_names
import json
from config import LOGLEVEL

def init_logger(logfile):
    logfile = logfile.split('/')[-1]
    pyLogging.basicConfig(
        filename=f"{logfile}.log",
        filemode="w",  # Overwrite log file on each run
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
    if len(sys.argv) < 2:
        print("Specify model name (optionally if with random weights)")
        sys.exit(1)

    values = sys.argv[1:]
    model_name = values[0]
    random_weight = 0
    if len(values) > 1:
        random_weight = values[1]

    if random_weight == "1":
        random_weight = True
        mode = "-with-random-weights"
    else:
        random_weight = False
        mode = ""


    logger = init_logger(model_name+mode)
    logger.log(LOGLEVEL, "Getting device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.log(LOGLEVEL, f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.log(LOGLEVEL, "GPU not available. Using CPU instead.")

    results = {}
    _model_name = model_name.split('/')[-1]
    output_file = f'data/{_model_name + mode}.json'

    if os.path.exists(output_file):
        with open(output_file, "r") as file:
            results = json.load(file)

    try:
        for task in tasks:
            if task['alias'] in results:
                continue
            logger.log(LOGLEVEL, f"{model_name}{mode} on {task['alias']}")
            mcc = finetune_model_by_task_mcc(logger, device, model_name, task, random_weight)
            results[task['alias']] = mcc
            logger.log(LOGLEVEL, f"MCC of {model_name}{mode} on {task['alias']} => mean: {mcc['mean']}, std: {mcc['std']}")
    except:
        pass

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {output_file}")


