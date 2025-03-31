import torch
from transformers import AutoModelForMaskedLM, EsmConfig

from config import models_cache_dir


def get_model(args, device):
    freeze = args.freeze
    train_from_scratch = args.from_scratch
    if train_from_scratch:
        config = EsmConfig.from_pretrained(f"model_configs/config-nucleotide-transformer-v2-50m-multi-species.json",
                                           local_files_only=True, trust_remote_code=True)
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
    model.to(device)

    torch.compiler.cudagraph_mark_step_begin()
    model = torch.compile(model)
    return model
