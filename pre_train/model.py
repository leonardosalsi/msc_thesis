import os

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, EsmConfig
import torch.nn as nn
from config import models_cache_dir
from pre_train.PCAModel import NucleotideModelWithPCA

def save_pca(pca, path):
    joblib.dump(pca, path)

def load_pca(path):
    return joblib.load(path)

def pca_exists(path):
    return os.path.exists(path)

def get_pca_weights(model, args, dataloader, device):
    save_path = os.path.join(args.dataset, f"pca_{args.pca_dim}.joblib")

    if not pca_exists(save_path):
        model.eval()
        model.to(device)
        cls_embeddings = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Collecting CLS embeddings"):
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                outputs = model(**batch, output_hidden_states=True, return_dict=True)
                cls = outputs.hidden_states[-1][:, 0]  # CLS token
                cls_embeddings.append(cls.cpu().numpy())

        cls_embeddings = np.concatenate(cls_embeddings, axis=0)
        pca = PCA(n_components=args.pca_dim)
        pca.fit(cls_embeddings)
        save_pca(pca, save_path)
        return pca
    else:
        return load_pca(save_path)

def get_model(args, dataloader, device):
    """
    Loads the model (either from scratch or pretrained) and applies optional modifications:
      - Freezes a portion of layers if specified.
      - Optionally wraps the input embedding with a PCA projection.
      - Moves the model to the target device.
      - Optionally compiles the model with torch.compile and wraps the forward method to
        handle an unexpected keyword argument ('num_items_in_batch').

    :param args: Arguments containing configuration flags (e.g. freeze, from_scratch, pca_embeddings, compile_model).
    :param dataloader: Dataloader containing the training data in case embeddings need to be extracted.
    :param device: The target device.
    :return: The modified model.
    """
    freeze = args.freeze
    train_from_scratch = args.from_scratch
    pca_embeddings = args.pca_embeddings
    pca_dim = args.pca_dim
    not_freeze_pca = not args.freeze_pca
    compile_model = args.compile_model

    if train_from_scratch:
        config = EsmConfig.from_pretrained(
            f"model_configs/config-nucleotide-transformer-v2-50m-multi-species.json",
            local_files_only=True, trust_remote_code=True
        )
        model = AutoModelForMaskedLM.from_config(config)
    else:
        model = AutoModelForMaskedLM.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
            cache_dir=models_cache_dir,
            trust_remote_code=True,
            local_files_only=True,
        )

    if freeze is not None:
        n_layers_to_freeze = int(len(model.esm.encoder.layer) * freeze)
        for idx, layer in enumerate(model.esm.encoder.layer):
            if idx < n_layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

    if pca_embeddings:
        pca = get_pca_weights(model, args, dataloader, device)
        model = NucleotideModelWithPCA(model.config, model, pca_dim=pca_dim)
        model.pca_proj.weight.data.copy_(torch.tensor(pca.components_, dtype=torch.float32))
        model.pca_proj.requires_grad_(not_freeze_pca)

    model.to(device)

    return model
