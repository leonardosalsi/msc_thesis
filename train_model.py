import json
import os
from dataclasses import dataclass
from typing import Optional
from config import pretrained_models_cache_dir, logs_dir
from utils.dataset import get_dataset
from utils.model import get_model
from utils.tokenizer import get_tokenizer
from utils.trainer import get_trainer
from utils.PCACollator import PCACollator
from utils.util import get_device, print_args
from argparse_dataclass import ArgumentParser
from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling
)

@dataclass
class TrainConfig:
    dataset: str
    tokenizer: str
    chunk_size: int = 1200
    max_workers: int = 4
    reverse_complement: bool = False
    compile_model: bool = False
    from_scratch: bool = False
    pca_dim: int = 0
    pca_embeddings: str = "cls"
    freeze_pca: bool = False
    checkpoint: Optional[int] = None
    freeze: Optional[float] = None
    train_size: int = 10
    eval_size: int = 64
    gradient_accumulation: int = 50
    save_steps: int = 6000
    logging_steps: int = 500
    max_steps: int = 12000
    use_scratch: bool = False
    keep_in_memory: bool = False
    load_from_json: bool = False
    ewc_lambda: float = 0.0
    original_dataset: Optional[str] = None
    gradient_checkpointing: bool = False
    mapping_cache: str = None
    deepspeed_config: str = None
    model_name: str = None
    resume: bool = True
    num_tokens: int = 1000

def parse_args():
    parser = ArgumentParser(TrainConfig)
    args, unknown = parser.parse_known_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    timestamp = print_args(args, "TRAINING ARGUMENTS")

    device = get_device()
    dataset_train, dataset_validation = get_dataset(args)
    tokenizer, num_tokens = get_tokenizer(args)


    def tokenize_function(examples):
        """
        Tokenizes the 'sequence' field in the examples.
        """
        outputs = tokenizer(examples['sequence'], max_length=num_tokens, truncation=True)
        return outputs


    if args.mapping_cache:
        name = 'logan' if 'logan' in args.dataset else 'multi_species'
        tokenized_train_sequences = dataset_train.map(
            tokenize_function,
            batched=True,
            num_proc=args.max_workers,
            cache_file_name=os.path.join(args.mapping_cache, f"train_{name}.arrow"),
            keep_in_memory=args.keep_in_memory
        )
        tokenized_validation_sequences = dataset_validation.select(range(500000)).map(
            tokenize_function,
            batched=True,
            num_proc=args.max_workers,
            cache_file_name=os.path.join(args.mapping_cache, f"validation_{name}.arrow"),
            keep_in_memory=args.keep_in_memory
        )
    else:
        tokenized_train_sequences = dataset_train.shuffle()
        tokenized_train_sequences.set_transform(tokenize_function)
        tokenized_validation_sequences = dataset_validation.shuffle()
        tokenized_validation_sequences = tokenized_validation_sequences.select(range(500000))
        tokenized_validation_sequences.set_transform(tokenize_function)

    if args.pca_dim > 0:
        data_collator = PCACollator(
            tokenizer=tokenizer,
            mlm_probability=0.15
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

    model = get_model(args, device)
    model_path = os.path.join(pretrained_models_cache_dir, timestamp)
    if os.path.isdir(model_path) and os.listdir(model_path) and args.resume:
        resume_from_checkpoint = model_path
    else:
        resume_from_checkpoint = None

    model_name = args.model_name if args.model_name else timestamp

    training_args = TrainingArguments(
        run_name=model_name,
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
        torch_compile=args.compile_model,
        label_names=['labels'],
        deepspeed=args.deepspeed_config,
        resume_from_checkpoint=resume_from_checkpoint
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
