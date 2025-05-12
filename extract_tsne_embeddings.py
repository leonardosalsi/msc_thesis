import os
from dataclasses import dataclass
import pickle
import numpy as np
import torch
from argparse_dataclass import ArgumentParser
from datasets import load_from_disk
from tqdm import tqdm
from config import generated_datasets_dir, results_dir
from utils.model import get_emb_model
from utils.tokenizer import get_eval_tokenizer
from utils.util import get_device, print_args

@dataclass
class EmbConfig:
    model_name: str
    checkpoint: str
    pca: bool = False
    var: bool = False

def parse_args():
    parser = ArgumentParser(EmbConfig)
    return parser.parse_args()

def extract_region_embeddings(args, device):

    model, repo, num_params = get_emb_model(args, device)
    model.eval()
    print(model.config.num_hidden_layers)

    tokenizer = get_eval_tokenizer(args, repo)

    if args.var:
        dataset = load_from_disk(os.path.join(generated_datasets_dir, 'tSNE_6000_var'))
    else:
        dataset = load_from_disk(os.path.join(generated_datasets_dir, 'tSNE_6000'))
    dataset = dataset.shuffle()

    L = model.config.num_hidden_layers
    layers = sorted(set([0, int(L * 0.25), int(L * 0.70), int(L * 0.90)]))
    if num_params <= 100:
        batch_size = 32
    else:
        batch_size = 16
    all_embeddings = {layer: [] for layer in layers}
    meta = {layer: [] for layer in layers}

    def get_kmer_offsets(sequence: str, kmer: int = 6):
        return [(i, i + kmer) for i in range(0, len(sequence) - kmer + 1, kmer)]

    for i in tqdm(range(0, len(dataset), batch_size), desc="Extracting mean-pooled embeddings"):
        batch = dataset[i:i + batch_size]
        sequences = batch["sequence"]
        feat_starts = batch["region_start"]
        feat_ends = batch["region_end"]

        tokens = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True,
                           add_special_tokens=False)

        with torch.no_grad():
            tokens = {k: v.to(device) for k, v in tokens.items() if isinstance(v, torch.Tensor)}
            outputs = model(**tokens, output_hidden_states=True)

        all_layer_embeddings = outputs.hidden_states
        for layer in layers:
            embeddings = all_layer_embeddings[layer]
            for j in range(len(sequences)):
                seq = sequences[j]
                rel_feat_start = feat_starts[j] - batch["start"][j]
                rel_feat_end = feat_ends[j] - batch["start"][j]

                kmer_offsets = get_kmer_offsets(seq, kmer=6)

                token_indices = [
                    idx for idx, (s, e) in enumerate(kmer_offsets)
                    if not (e <= rel_feat_start or s >= rel_feat_end) and 0 <= idx < embeddings.shape[1]
                ]

                if not token_indices:
                    print(f"[WARN] No valid overlapping tokens for sample {i + j}, skipping. Len: {len(all_embeddings)}")
                    continue

                selected = embeddings[j, token_indices, :]
                pooled = selected.mean(dim=0).cpu().numpy()
                all_embeddings[layer].append(pooled)
                meta[layer].append({
                    "sequence": seq,
                    "label": batch["region"][j],
                    "start": feat_starts,
                    "end": feat_ends,
                    "GC": batch["region_gc"][j],
                    "full_start":  batch["start"][j],
                    "full_end": batch["end"][j],
                    "GC_full": batch["seq_gc"][j]
                })

    ouput_dir = os.path.join(results_dir, f"tSNE_embeddings")
    os.makedirs(ouput_dir, exist_ok=True)
    model_dir = os.path.join(ouput_dir, args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    for layer, embeddings in all_embeddings.items():
        output_path = os.path.join(model_dir, f"layer_{layer}.pkl")
        with open(output_path, "wb") as f:
            pickle.dump({
                "embeddings": np.vstack(embeddings),
                "meta": meta[layer]
            }, f)

if __name__ == "__main__":
    args = parse_args()
    timestamp = print_args(args, "EMBEDDING EXTRACTION ARGUMENTS")
    device = get_device()
    extract_region_embeddings(args, device)
