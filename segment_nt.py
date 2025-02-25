import argparse
import json
import os
import sys
from pprint import pprint

import numpy as np
from datasets import load_from_disk, load_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from config import generated_datasets_dir, models_cache_dir, datasets_cache_dir, results_dir
from util import get_filtered_dataset_name

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
    parser = argparse.ArgumentParser(
        description="Script to examine a dataset with SegmentNT."
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Name of the dataset",
        choices=["multi_genome_dataset", "logan"]
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        help="Chunk size (defined when further splitting data)",
    )
    parser.add_argument(
        "--shannon",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Lower and upper margin of allowed Shannon entropy (e.g., --shannon 1.4 1.8)"
    )
    parser.add_argument(
        "--gc",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Lower and upper margin of allowed GC content (e.g., --gc 0.4 0.6)"
    )
    parser.add_argument(
        "--kmer",
        type=int,
        help="Kmer size (only when using logan)",
    )
    parser.add_argument(
        "--reverse_complement",
        action="store_true",
        dest="reverse_complement",
        help="Use dataset generated with reverse complement (only when using logan)."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    selected_dataset = args.dataset
    chunk_size_folder_name = get_filtered_dataset_name(args.chunk_size, args.shannon, args.gc)
    shannon = args.shannon
    gc = args.gc
    kmer = args.kmer
    reverse_complement = args.reverse_complement

    if selected_dataset == "multi_genome_dataset" and shannon is None and gc is None:
        dataset = load_dataset(
            "InstaDeepAI/multi_species_genomes",
            cache_dir=datasets_cache_dir,
            split='train',
            trust_remote_code=True
        )
    elif selected_dataset == "logan":
        if not kmer:
            print("Kmer size must be specified when using logan.")
            exit(1)
        dataset_name = f"kmer_{kmer}"
        if reverse_complement:
            dataset_name += "_reverse"
        dataset_path = os.path.join(generated_datasets_dir, selected_dataset, dataset_name)
        dataset = load_from_disk(dataset_path)['train']
    else:
        print("Unknown dataset selected")
        exit(1)

    dataset = dataset.shuffle()
    print(dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    model.to(device)
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
    print(num_sequences)
    pprint(all_counts)
    result = {
        'num_data': num_sequences,
        'results': {}
    }

    for f in features:
        feat_idx = features.index(f)
        result['results'][f] = int(all_counts[feat_idx])
    pprint(result)
    result_file = os.path.join(results_dir, f"{selected_dataset}_segment_nt.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=4)