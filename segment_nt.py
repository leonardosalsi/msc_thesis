import argparse
import json
import os
import sys
from pprint import pprint
from typing import Optional
from argparse_dataclass import ArgumentParser
import numpy as np
from dataclasses import dataclass
from datasets import load_from_disk, load_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from config import models_cache_dir, results_dir
from pre_train.dataset import get_dataset
from pre_train.util import print_args, get_device
torch.set_printoptions(threshold=sys.maxsize)

def collate_fn(batch):
    sequences = [example["sequence"] for example in batch]

    tokens = tokenizer.batch_encode_plus(
        sequences,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    return tokens

def parse_args():
    parser = ArgumentParser(TrainConfig)
    return parser.parse_args()

@dataclass
class TrainConfig:
    dataset: str
    tokenizer: str = 'SPEC'
    use_scratch: bool = False
    keep_in_memory: bool = False
    load_from_json: bool = False

if __name__ == "__main__":
    args = parse_args()
    timestamp = print_args(args, "TRAINING ARGUMENTS")

    device = get_device()
    dataset_train, _ = get_dataset(args)

    dataset = dataset_train.shuffle()
    print(dataset)

    max_length = 8 + 1
    print("cache dir: {}".format(models_cache_dir))
    model = AutoModel.from_pretrained(
        "InstaDeepAI/segment_nt",
        cache_dir=models_cache_dir,
        trust_remote_code=True,
        local_files_only=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "InstaDeepAI/segment_nt",
        cache_dir=models_cache_dir,
        trust_remote_code=True,
        local_files_only=True,
        model_max_length=2000
    )

    model.eval()

    dataloader = DataLoader(dataset, batch_size=2000, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    features = model.config.features

    all_counts = torch.zeros(14, dtype=torch.int32).to(device)
    num_sequences = len(dataset)

    with torch.inference_mode():
        for batch in tqdm(dataloader):
            tokens = batch["input_ids"]
            tokens = tokens.to(device)
            attention_mask = tokens != tokenizer.pad_token_id

            outs = model(tokens, attention_mask=attention_mask, output_hidden_states=True)
            logits = outs.logits.detach()
            probs = torch.nn.functional.softmax(logits, dim=-1)
            certainty = torch.abs(probs[..., 0] - probs[..., 1])
            certain_mask = certainty >= 0.4
            certain_mask = certain_mask.float().mean(dim=1)
            threshold = 0.5
            sequence_uncertain = certain_mask > threshold
            all_counts += certain_mask.int().sum(dim=0)

    result = {
        'num_data': num_sequences,
        'results': {}
    }

    for f in features:
        feat_idx = features.index(f)
        result['results'][f] = int(all_counts[feat_idx])

    result_file = os.path.join(results_dir, f"{timestamp}_segment_nt.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=4)