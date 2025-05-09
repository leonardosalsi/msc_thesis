import os
from safetensors.torch import load_file
from transformers import AutoModelForMaskedLM, EsmConfig, AutoModelForSequenceClassification
from config import models_cache_dir, pretrained_models_cache_dir
from utils.PCAModel import EsmForMaskedLMPCA, EsmForSequenceClassificationPCA
import re

def get_model(args, device):
    """
    Loads the model (either from scratch or pretrained) and applies optional modifications:
      - Freezes a portion of layers if specified.
      - Optionally wraps the input embedding with a PCA projection.
      - Moves the model to the target device.
      - Optionally compiles the model with torch.compile and wraps the forward method to
        handle an unexpected keyword argument ('num_items_in_batch').

    :param args: Arguments containing configuration flags (e.g. freeze, from_scratch, pca_embeddings, compile_model).
    :param device: The target device.
    :return: The modified model.
    """
    freeze = args.freeze
    train_from_scratch = args.from_scratch
    pca_dim = args.pca_dim
    pca_embeddings = args.pca_embeddings
    gradient_accumulation_steps = args.gradient_accumulation

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

    if pca_dim > 0:
        model = EsmForMaskedLMPCA(model.config, model, pca_dim=pca_dim, pca_embeddings=pca_embeddings, gradient_accumulation_steps=gradient_accumulation_steps)

    model.to(device)

    return model

def get_eval_model(args, num_labels, device):
    repo = None
    if 'untrained' in args.model_name:
        match = re.search(r'\d+', args.model_name)
        if match:
            number = int(match.group())
            if number == 100:
                repo = 'InstaDeepAI/nucleotide-transformer-v2-100m-multi-species'
            elif number == 250:
                repo = 'InstaDeepAI/nucleotide-transformer-v2-250m-multi-species'
            elif number == 500:
                if 'tg' in args.model_name:
                    repo = 'InstaDeepAI/nucleotide-transformer-500m-1000g'
                elif 'human' in args.model_name:
                    repo = 'InstaDeepAI/nucleotide-transformer-500m-human-ref'
                else:
                    repo = 'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species'
        else:
            repo = 'InstaDeepAI/nucleotide-transformer-v2-50m-multi-species'

        if repo is None:
            raise ValueError(f"No model existing with {args.model_name}")

        model = AutoModelForSequenceClassification.from_pretrained(
            repo,
            cache_dir=models_cache_dir,
            num_labels=num_labels,
            trust_remote_code=True,
            local_files_only=True
        )

    else:
        model_dir = os.path.join(pretrained_models_cache_dir, f"{args.model_name}", f"checkpoint-{args.checkpoint}")
        if args.pca:
            base_model = AutoModelForMaskedLM.from_pretrained(
                model_dir,
                cache_dir=models_cache_dir,
                num_labels=num_labels,
                trust_remote_code=True,
                local_files_only=True
            )

            pca_model = EsmForMaskedLMPCA(
                config=base_model.config,
                base_model=base_model,
                pca_dim=args.pca_dims,
                aux_loss_weight=0.1,
                temperature=0.1,
                pca_embeddings=args.pca_embeddings,
                gradient_accumulation_steps=50,
                contrastive=False
            )

            state_dict = load_file(f"{model_dir}/model.safetensors")
            missing_keys, unexpected_keys = pca_model.load_state_dict(state_dict, strict=True)

            if len(missing_keys) > 0:
                print("Missing keys:", missing_keys)
            if len(unexpected_keys) > 0:
                print("Unexpected keys:", unexpected_keys)

            model = EsmForSequenceClassificationPCA(pca_model=pca_model, num_labels=num_labels)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                cache_dir=models_cache_dir,
                num_labels=num_labels,
                trust_remote_code=True,
                local_files_only=True
            )

    model.to(device)

    return model, repo

def get_emb_model(args, device):
    repo = None
    num_params = None
    if 'untrained' in args.model_name:
        match = re.search(r'\d+', args.model_name)
        if match:
            number = int(match.group())
            if number == 100:
                num_params = 100
                repo = 'InstaDeepAI/nucleotide-transformer-v2-100m-multi-species'
            elif number == 250:
                num_params = 250
                repo = 'InstaDeepAI/nucleotide-transformer-v2-250m-multi-species'
            elif number == 500:
                num_params = 500
                if 'tg' in args.model_name:
                    repo = 'InstaDeepAI/nucleotide-transformer-500m-1000g'
                elif 'human' in args.model_name:
                    repo = 'InstaDeepAI/nucleotide-transformer-500m-human-ref'
                else:
                    repo = 'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species'
        else:
            num_params = 50
            repo = 'InstaDeepAI/nucleotide-transformer-v2-50m-multi-species'

        if repo is None:
            raise ValueError(f"No model existing with {args.model_name}")

        model = AutoModelForMaskedLM.from_pretrained(
            repo,
            cache_dir=models_cache_dir,
            trust_remote_code=True,
            local_files_only=True
        )

    else:
        num_params = 50
        model_dir = os.path.join(pretrained_models_cache_dir, f"{args.model_name}", f"checkpoint-{args.checkpoint}")
        model = AutoModelForMaskedLM.from_pretrained(
            model_dir,
            cache_dir=models_cache_dir,
            trust_remote_code=True,
            local_files_only=True
        )

    model.to(device)

    return model, repo, num_params