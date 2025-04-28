from transformers import AutoModelForMaskedLM, EsmConfig
from config import models_cache_dir
from utils.PCAModel import EsmForMaskedLMPCA

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
