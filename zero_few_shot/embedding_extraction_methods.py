import torch
from tqdm import tqdm
from utils.model import get_emb_model
from utils.tokenizer import get_eval_tokenizer
import torch.nn.functional as F

def _get_kmer_offsets(sequence: str, kmer: int = 6):
    return [(i, i + kmer) for i in range(0, len(sequence) - kmer + 1, kmer)]

def _inner_extraction_function(model, dataset, tokenizer, device, layers, batch_size):
    all_embeddings = {layer: [] for layer in layers}
    meta = {layer: [] for layer in layers}

    for i in tqdm(range(0, len(dataset), batch_size), desc="Extracting mean-pooled embeddings"):
        batch = dataset[i:i + batch_size]
        sequences = batch["sequence"]
        feat_starts = batch["loc_start"]
        feat_ends = batch["loc_end"]

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
                kmer_offsets = _get_kmer_offsets(seq, kmer=6)

                token_indices = [
                    idx for idx, (s, e) in enumerate(kmer_offsets)
                    if not (e <= feat_starts[j] or s >= feat_ends[j]) and 0 <= idx < embeddings.shape[1]
                ]

                if not token_indices:
                    print(
                        f"[WARN] No valid overlapping tokens for sample {i + j}, skipping. Len: {len(all_embeddings)}")
                    continue

                selected = embeddings[j, token_indices, :]
                pooled = selected.mean(dim=0).cpu().numpy()

                all_embeddings[layer].append(pooled)
                meta[layer].append({
                    "label": batch["region"][j],
                    "reg_len": feat_ends[j] - feat_starts[j],
                    "GC": batch["region_gc"][j],
                    "GC_full": batch["seq_gc"][j]
                })

    return all_embeddings, meta


def extract_region_embeddings_genomic_elements(args, device, dataset):
    model, repo, num_params = get_emb_model(args, device)
    model.eval()
    tokenizer = get_eval_tokenizer(args, repo)
    L = model.config.num_hidden_layers

    dataset_train = dataset['train']
    dataset_test = dataset['test']

    layers = sorted(set([0, int(L * 0.5), int(L - 1)]))
    if num_params <= 100:
        batch_size = 32
    else:
        batch_size = 16

    train_embeddings, train_meta = _inner_extraction_function(model, dataset_train, tokenizer, device, layers, batch_size)
    test_embeddings, test_meta = _inner_extraction_function(model, dataset_test, tokenizer, device, layers, batch_size)

    return train_embeddings, train_meta, test_embeddings, test_meta

def extract_region_embeddings_5_prime_UTR(args, device, dataset):

    def mutate(seq, pos, ref, alt):
        if seq[pos] != ref:
            raise ValueError(f"Expected ref base {ref}, but got {seq[pos]}")
        return seq[:pos] + alt + seq[pos + 1:]

    model, repo, num_params = get_emb_model(args, device)
    model.eval()

    tokenizer = get_eval_tokenizer(args, repo)

    L = model.config.num_hidden_layers
    layers = sorted(set([0, int(L * 0.5), int(L - 1)]))
    if num_params <= 100:
        batch_size = 16
    else:
        batch_size = 8

    train_embeddings = {layer: [] for layer in layers}
    test_embeddings = {layer: [] for layer in layers}

    train_meta = {layer: [] for layer in layers}
    test_meta = {layer: [] for layer in layers}

    for i in tqdm(range(0, len(dataset), batch_size), desc="Extracting mean-pooled embeddings"):
        batch = dataset[i:i + batch_size]
        sequences = batch["sequence"]
        split_set = batch["set"]
        mutated_sequences = [mutate(batch['sequence'][i], batch['pos'][i], batch['ref'][i], batch['alt'][i]) for i in range(len(sequences))]
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
                rel_pos = batch["pos"][j]
                window = 100
                rel_feat_start = max(0, rel_pos - window)
                rel_feat_end = min(len(seq), rel_pos + window + 1)
                kmer_offsets = _get_kmer_offsets(seq, kmer=6)

                token_indices = [
                    idx for idx, (s, e) in enumerate(kmer_offsets)
                    if not (e <= rel_feat_start or s >= rel_feat_end) and 0 <= idx < embeddings.shape[1]
                ]

                if not token_indices:
                    print(f"[WARN] No valid overlapping tokens for sample {i + j}, skipping.")
                    continue

                ref_selected = embeddings[j, token_indices, :]
                mut_selected = mut_embeddings[j, token_indices, :]
                ref_embedding = ref_selected.mean(dim=0).cpu()
                mut_embedding = mut_selected.mean(dim=0).cpu()

                cos_similarity = F.cosine_similarity(ref_embedding.unsqueeze(0), mut_embedding.unsqueeze(0)).item()
                pooled = mut_embedding.numpy()

                if split_set[j] == "train":
                    train_embeddings[layer].append(pooled)
                    train_meta[layer].append({
                        "label": batch["label"][j],
                        "af": batch["af"][j],
                        "cos_similarity": cos_similarity,
                    })
                else:
                    test_embeddings[layer].append(pooled)
                    test_meta[layer].append({
                        "label": batch["label"][j],
                        "af": batch["af"][j],
                        "cos_similarity": cos_similarity,
                    })

    return train_embeddings, train_meta, test_embeddings, test_meta