import os
from dataclasses import dataclass
import pickle
from typing import Optional

from argparse_dataclass import ArgumentParser
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model, IA3Config
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from config import results_dir, generated_datasets_dir, logs_dir, cache_dir
from utils.model import get_classification_model
from utils.tokenizer import get_eval_tokenizer
from utils.util import get_device, print_args

@dataclass
class MRLConfig:
    model_name: str
    checkpoint: str
    keep_in_memory: bool = True
    pca: bool = False
    pca_dims: int = None
    pca_embeddings: str = None
    map_cache_dir: str = None
    peft: Optional[str] = 'IA3'
    random_init: bool = False

def parse_args():
    parser = ArgumentParser(MRLConfig)
    return parser.parse_args()

def compute_pearsonr(eval_pred):
    preds = eval_pred.predictions
    references = eval_pred.label_ids
    return pearsonr(references, preds)[0]

if __name__ == "__main__":
    args = parse_args()
    timestamp = print_args(args, "LoRA FINETUNE & TEST MRL ARGUMENTS")
    device = get_device()

    pred_folder = os.path.join(results_dir, 'mrl_predictions')
    os.makedirs(pred_folder, exist_ok=True)
    file_name = os.path.join(pred_folder, f"{args.model_name}.pkl")
    if os.path.exists(file_name):
        exit()

    if args.map_cache_dir:
        os.environ["HF_DATASETS_CACHE"] = args.map_cache_dir

    dataset = load_from_disk(os.path.join(generated_datasets_dir, 'mrl_prediction'), keep_in_memory=args.keep_in_memory)

    train_set = dataset['train'].shuffle()
    test_random_fixed = dataset['test_random_fixed'].shuffle()
    test_random_var = dataset['test_random_var'].shuffle()
    test_human_fixed = dataset['test_human_fixed'].shuffle()
    test_human_var = dataset['test_human_var'].shuffle()

    model, repo = get_classification_model(args, device, regression=True)
    tokenizer = get_eval_tokenizer(args, repo)

    def tokenize_function(example):
        tokens = tokenizer(
            example["sequence"],
            truncation=True,
            padding="max_length",
            max_length=100,

        )
        tokens["labels"] = example["label"]
        return tokens


    train_splits = train_set.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_splits['train']
    eval_dataset = train_splits['test']

    keep_cols = ["input_ids", "attention_mask", "labels"]
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=[c for c in train_dataset.column_names if c not in keep_cols])
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=[c for c in train_dataset.column_names if c not in keep_cols])

    tokenized_test_random_fixed = test_random_fixed.map(tokenize_function, batched=True, remove_columns=[c for c in train_dataset.column_names if c not in keep_cols])
    tokenized_test_random_var = test_random_var.map(tokenize_function, batched=True, remove_columns=[c for c in train_dataset.column_names if c not in keep_cols])
    tokenized_test_human_fixed = test_human_fixed.map(tokenize_function, batched=True, remove_columns=[c for c in train_dataset.column_names if c not in keep_cols])
    tokenized_test_human_var = test_human_var.map(tokenize_function, batched=True, remove_columns=[c for c in train_dataset.column_names if c not in keep_cols])

    training_args = TrainingArguments(
        run_name=f"mrl_{args.model_name}",
        output_dir=os.path.join(cache_dir, 'eval_models', f"mrl_{args.model_name}_{timestamp}"),
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="eval_loss",
        per_device_train_batch_size=128,
        per_device_eval_batch_size=16,
        eval_accumulation_steps=200,
        greater_is_better=False,
        num_train_epochs=5,
        learning_rate=2e-4,
        logging_dir=logs_dir,
        report_to="none",
    )

    """Employ PEFT """
    modules_to_save = None
    if args.pca:
        modules_to_save = ["pca_proj", "layernorm"]

    if args.peft == "LoRA":
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "value"],
            modules_to_save=modules_to_save
        )
    elif args.peft == "IA3":
        peft_config = IA3Config(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            target_modules=["query", "value", "intermediate.dense", "output.dense"],
            feedforward_modules=["intermediate.dense", "output.dense"],
            modules_to_save=modules_to_save
        )
    else:
        raise f"PEFT {args.peft} is not supported, only use 'IA3' or 'LoRA'."

    model = get_peft_model(model, peft_config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    _ = trainer.train()

    preds_random_fixed = trainer.predict(tokenized_test_random_fixed, ignore_keys=["hidden_states","attentions"])
    y_pred_random_fixed = preds_random_fixed.predictions.reshape(-1)
    y_true_random_fixed = preds_random_fixed.label_ids

    preds_random_var = trainer.predict(tokenized_test_random_var, ignore_keys=["hidden_states","attentions"])
    y_pred_random_var = preds_random_var.predictions.reshape(-1)
    y_true_random_var = preds_random_var.label_ids

    preds_human_fixed = trainer.predict(tokenized_test_human_fixed, ignore_keys=["hidden_states","attentions"])
    y_pred_human_fixed = preds_human_fixed.predictions.reshape(-1)
    y_true_human_fixed = preds_human_fixed.label_ids

    preds_human_var = trainer.predict(tokenized_test_human_var, ignore_keys=["hidden_states","attentions"])
    y_pred_human_var = preds_human_var.predictions.reshape(-1)
    y_true_human_var = preds_human_var.label_ids

    print("[RANDOM] Fixed length test set:")
    print("Pearson R:", pearsonr(y_pred_random_fixed, y_true_random_fixed)[0])
    print()
    print("[RANDOM] Variable length test set:")
    print("Pearson R:", pearsonr(y_pred_random_var, y_true_random_var)[0])
    print()
    print("[HUMAN] Fixed length test set:")
    print("Pearson R:", pearsonr(y_pred_human_fixed, y_true_human_fixed)[0])
    print()
    print("[HUMAN] Variable length test set:")
    print("Pearson R:", pearsonr(y_pred_human_var, y_true_human_var)[0])


    with open(file_name, "wb") as f:
        pickle.dump({
            "y_pred_random_fixed": y_pred_random_fixed,
            "y_true_random_fixed": y_true_random_fixed,
            "y_pred_random_var": y_pred_random_var,
            "y_true_random_var": y_true_random_var,
            "y_pred_human_fixed": y_pred_human_fixed,
            "y_true_human_fixed": y_true_human_fixed,
            "y_pred_human_var": y_pred_human_var,
            "y_true_human_var": y_true_human_var,
        }, f)