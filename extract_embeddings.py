import os
from dataclasses import dataclass
import pickle
import numpy as np
from argparse_dataclass import ArgumentParser
from datasets import load_from_disk
from config import results_dir
from embedding_inspection.extraction_methods import extract_region_embeddings_5_prime_UTR, extract_region_embeddings_genomic_elements
from utils.util import get_device, print_args

@dataclass
class EmbConfig:
    model_name: str
    checkpoint: str
    dataset_path: str

def parse_args():
    parser = ArgumentParser(EmbConfig)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    timestamp = print_args(args, "EMBEDDING EXTRACTION ARGUMENTS")
    device = get_device()
    dataset = load_from_disk(args.dataset_path)

    if 'genomic_regions_annotated' in args.dataset_path:
        extraction_function = extract_region_embeddings_genomic_elements
    elif '5_utr_af_prediction' in args.dataset_path:
        extraction_function = extract_region_embeddings_5_prime_UTR
    else:
        raise ValueError(f"Dataset {dataset.info.dataset_name} not supported")

    dataset = dataset.shuffle()

    train_embeddings, train_meta, test_embeddings, test_meta = extraction_function(args, device, dataset)
    embeddings_dir = os.path.join(results_dir, 'embeddings')
    os.makedirs(embeddings_dir, exist_ok=True)
    dataset_emb_dir = os.path.join(embeddings_dir, dataset.info.dataset_name)
    os.makedirs(dataset_emb_dir, exist_ok=True)
    model_dir = os.path.join(dataset_emb_dir, args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    for layer, embeddings in train_embeddings.items():
        output_path = os.path.join(model_dir, f"layer_{layer}.pkl")
        with open(output_path, "wb") as f:
            pickle.dump({
                "train_embeddings": np.vstack(train_embeddings[layer]),
                "train_meta": train_meta[layer],
                "test_embeddings": np.vstack(test_embeddings[layer]),
                "test_meta": test_meta[layer],
            }, f)