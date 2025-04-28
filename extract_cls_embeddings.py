
import argparse
import datetime
import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.trainer import get_trainer

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False

from config import pretrained_models_cache_dir, logs_dir
from utils.dataset import get_dataset
from utils.model import get_model
from utils.tokenizer import get_tokenizer
from utils.util import get_device, print_args, compute_metrics
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
        "--pca_dims",
        type=int,
        default=128,
        help="Number of gradient accumulation steps. Default is 50."
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
        "--load_from_json",
        action="store_true",
        dest="load_from_json",
        help="Keep dataset in memory."
    )
    parser.add_argument(
        "--ewc_lambda",
        type=float,
        default=0.0,
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
    dataset_train, _ = get_dataset(args)
    tokenizer, num_tokens = get_tokenizer(args)

    def tokenize_function(examples):
        """
        Tokenizes the 'sequence' field in the examples.
        """
        outputs = tokenizer(examples['sequence'], max_length=num_tokens, truncation=True)
        return outputs

    tokenized_train_sequences = dataset_train.shuffle()
    tokenized_train_sequences.set_transform(tokenize_function)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    dataloader = DataLoader(
        tokenized_train_sequences,
        batch_size=10,
        shuffle=False,
        collate_fn=data_collator
    )

    model.eval()
    cls_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting CLS embeddings"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, output_hidden_states=True, return_dict=True)
            cls = outputs.hidden_states[-1][:, 0]  # CLS token
            cls_embeddings.append(cls.cpu().numpy())

    cls_embeddings = np.concatenate(cls_embeddings, axis=0)
    np.save(f"cls_embeddings_{timestamp}.npy", cls_embeddings)

