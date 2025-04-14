
import argparse
import datetime
import os
import json
import torch

from pre_train.trainer import get_trainer

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False

from config import pretrained_models_cache_dir, logs_dir
from pre_train.dataset import get_dataset
from pre_train.model import get_model
from pre_train.tokenizer import get_tokenizer
from pre_train.util import get_device, print_args, compute_metrics
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EsmForMaskedLM,
)

def parse_args():
    """
    Parse command line arguments for model training.
    """
    parser = argparse.ArgumentParser(
        description="Train a model from scratch or from pretrained weights with specified tokenization."
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to the dataset containing training and validation data."
    )
    parser.add_argument(
        "tokenizer",
        type=str,
        help="Tokenizer type to use. Options: 'default' or 'overlapping'."
    )
    parser.add_argument(
        "--chunk_size",
        default=1200,
        type=int,
        help="Chunk size for splitting data (in base pairs). Default is 1200."
    )
    parser.add_argument(
        "--max_workers",
        default=4,
        type=int,
        help="Chunk size for splitting data (in base pairs). Default is 1200."
    )
    parser.add_argument(
        "--reverse_complement",
        action="store_true",
        dest="reverse_complement",
        help="Use dataset generated with reverse complement sequences (if applicable)."
    )
    parser.add_argument(
        "--compile_model",
        action="store_true",
        dest="compile_model",
        help="Compile the model with torch.compile for potential performance improvements."
    )
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        dest="from_scratch",
        help="Train the model from scratch instead of using pretrained weights."
    )
    parser.add_argument(
        "--pca_embeddings",
        action="store_true",
        dest="pca_embeddings",
        help="Apply PCA-based post-embedding processing to reduce embedding dimensionality."
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        nargs=1,
        metavar=("modelId in PRETRAINED_MODELS"),
        help="Use a checkpoint from PRETRAINED_MODELS instead of pretrained weights."
    )
    parser.add_argument(
        "--freeze",
        type=float,
        help="Fraction of encoder layers to freeze (between 0.1 and 0.9)."
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=10,
        help="Training batch size per device. Default is 10."
    )
    parser.add_argument(
        "--eval_size",
        type=int,
        default=64,
        help="Evaluation batch size per device. Default is 64."
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=50,
        help="Number of gradient accumulation steps. Default is 50."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=6000,
        help="Frequency of saving model checkpoints (in steps). Default is 6000."
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Frequency of logging training metrics (in steps). Default is 500."
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=12000,
        help="Maximum number of training steps. Default is 12000."
    )
    parser.add_argument(
        "--use_scratch",
        action="store_true",
        dest="use_scratch",
            help="Pre-load everything into local scratch and load from there."
    )
    parser.add_argument(
        "--keep_in_memory",
        action="store_true",
        dest="keep_in_memory",
        help="Keep dataset in memory."
    )
    parser.add_argument(
        "--ewc_lambda",
        type=int,
        default=0,
        help="Maximum number of training steps. Default is 12000."
    )
    parser.add_argument(
        "--original_dataset",
        type=str,
        default=None,
        help="Maximum number of training steps. Default is 12000."
    )
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()

    timestamp = print_args(args, "TRAINING ARGUMENTS")

    device = get_device()
    model = get_model(args, device)
    dataset_train, dataset_validation = get_dataset(args)
    tokenizer, num_tokens = get_tokenizer(args)

    def tokenize_function(examples):
        """
        Tokenizes the 'sequence' field in the examples.
        """
        outputs = tokenizer(examples['sequence'], max_length=num_tokens, truncation=True)
        return outputs

    tokenized_train_sequences = dataset_train.shuffle()
    tokenized_train_sequences.set_transform(tokenize_function)
    tokenized_validation_sequences = dataset_validation.shuffle()
    tokenized_validation_sequences = tokenized_validation_sequences.select(range(20))
    tokenized_validation_sequences.set_transform(tokenize_function)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    model_path = os.path.join(pretrained_models_cache_dir, timestamp)
    if os.path.isdir(model_path) and os.listdir(model_path):
        resume_from_checkpoint = True
    else:
        resume_from_checkpoint = False
    
    training_args = TrainingArguments(
        run_name=timestamp,
        report_to="none",
        output_dir=model_path,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.train_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        per_device_eval_batch_size=args.eval_size,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        dataloader_num_workers=args.max_workers,
        gradient_checkpointing=False,
        logging_dir=None,
        remove_unused_columns=False,
        bf16=True,
        max_steps=args.max_steps,
        include_num_input_tokens_seen=True,
        prediction_loss_only=True,
        torch_compile=args.compile_model
    )

    trainer = get_trainer(
        args,
        training_args,
        model,
        device,
        tokenizer,
        tokenized_train_sequences,
        tokenized_validation_sequences,
        data_collator,
        num_tokens
    )

    _ = trainer.train()

    log_history_path = os.path.join(logs_dir, f"log_history_{timestamp}.json")
    with open(log_history_path, "w") as log_file:
        json.dump(trainer.state.log_history, log_file, indent=4)
