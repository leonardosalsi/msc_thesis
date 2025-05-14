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
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F


@dataclass
class EmbConfig:
    model_name: str
    checkpoint: str
    pca: bool = False

def parse_args():
    parser = ArgumentParser(EmbConfig)
    return parser.parse_args()

def mutate(seq, pos, ref, alt):
    if seq[pos] != ref:
        raise ValueError(f"Expected ref base {ref}, but got {seq[pos]}")
    return seq[:pos] + alt + seq[pos + 1:]

def extract_region_embeddings(args, device):

    model, repo, num_params = get_emb_model(args, device)
    model.eval()
    print(model.config.num_hidden_layers)

    tokenizer = get_eval_tokenizer(args, repo)


    dataset = load_from_disk(os.path.join(generated_datasets_dir, '5_utr_af_6000'))
    dataset = dataset.shuffle()

    L = model.config.num_hidden_layers
    layers = sorted(set([0, int(L * 0.5), int(L -1)]))
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
        mutated_sequences = [mutate(b['sequence'], b['pos'], b['start'], b['ref'], b['alt']) for b in batch]
        tokens = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True,
                           add_special_tokens=False)
        mut_tokens = tokenizer(mutated_sequences, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True,
                           add_special_tokens=False)
        with torch.no_grad():
            tokens = {k: v.to(device) for k, v in tokens.items() if isinstance(v, torch.Tensor)}
            outputs = model(**tokens, output_hidden_states=True)
            mut_tokens = {k: v.to(device) for k, v in mut_tokens.items() if isinstance(v, torch.Tensor)}
            mut_outputs = model(**mut_tokens, output_hidden_states=True)

        all_layer_embeddings = outputs.hidden_states
        mut_all_layer_embeddings = mut_outputs.hidden_states
        for layer in layers:
            embeddings = all_layer_embeddings[layer]
            mut_embeddings = mut_all_layer_embeddings[layer]
            for j in range(len(sequences)):
                seq = sequences[j]
                rel_pos = batch["pos"][j] - batch["start"][j]
                window = 100
                rel_feat_start = max(0, rel_pos - window)
                rel_feat_end = min(len(seq), rel_pos + window + 1)
                kmer_offsets = get_kmer_offsets(seq, kmer=6)

                token_indices = [
                    idx for idx, (s, e) in enumerate(kmer_offsets)
                    if not (e <= rel_feat_start or s >= rel_feat_end) and 0 <= idx < embeddings.shape[1]
                ]

                if not token_indices:
                    print(f"[WARN] No valid overlapping tokens for sample {i + j}, skipping. Len: {len(all_embeddings)}")
                    continue

                ref_embedding = embeddings[j].mean(dim=0).cpu()
                mut_embedding = mut_embeddings[j].mean(dim=0).cpu()

                cos_similarity = F.cosine_similarity(ref_embedding.unsqueeze(0), mut_embedding.unsqueeze(0)).item()
                dot_product = torch.dot(ref_embedding, mut_embedding).item()
                ref_norm = F.normalize(ref_embedding.unsqueeze(0), dim=1)
                mut_norm = F.normalize(mut_embedding.unsqueeze(0), dim=1)
                dot_product_norm   = torch.dot(ref_norm.squeeze(), mut_norm.squeeze()).item()

                selected = mut_embeddings[j, token_indices, :]
                pooled = selected.mean(dim=0).cpu().numpy()
                all_embeddings[layer].append(pooled)
                meta[layer].append({
                    "label": batch["label"][j],
                    "af": batch["af"][j],
                    "cos_similarity": cos_similarity,
                    "dot_product": dot_product,
                    "dot_product_norm": dot_product_norm,
                })

    ouput_dir = os.path.join(results_dir, f"5_utr_embeddings")
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
