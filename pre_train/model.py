import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForMaskedLM, EsmConfig
import torch.nn as nn
from config import models_cache_dir

def compute_pca_projection(embedding_layer, target_dim):
    """
    Computes a PCA projection matrix on the embedding layer's weights.
    :param embedding_layer: The original embedding layer.
    :param target_dim: The desired reduced dimension.
    :return: A torch.Tensor with shape (original_dim, target_dim)
    """
    weights = embedding_layer.weight.detach().cpu().numpy()
    pca = PCA(n_components=target_dim)
    pca.fit(weights)
    # pca.components_ shape: (target_dim, emb_dim); we need (emb_dim, target_dim)
    projection_matrix = torch.tensor(pca.components_.T, dtype=torch.float32)
    return projection_matrix

class PCABottleneckEmbedding(nn.Module):
    """
    A custom embedding module that wraps an existing embedding layer with a PCA-based bottleneck.
    It first computes the embedding, then projects it to a lower-dimensional space via PCA,
    and finally reconstructs it back to the original embedding dimension.
    """
    def __init__(self, orig_embedding, reduction_factor=0.5, freeze_pca=True):
        super().__init__()
        self.orig_embedding = orig_embedding
        orig_dim = orig_embedding.embedding_dim
        target_dim = int(orig_dim * reduction_factor)
        projection_matrix = compute_pca_projection(orig_embedding, target_dim)
        self.pca_layer = nn.Linear(orig_dim, target_dim, bias=False)
        self.pca_layer.weight.data.copy_(projection_matrix.T)
        if freeze_pca:
            self.pca_layer.weight.requires_grad = False
        self.reconstruction_layer = nn.Linear(target_dim, orig_dim, bias=False)

    def forward(self, input_ids):
        # Compute the original embeddings.
        emb = self.orig_embedding(input_ids)  # shape: (batch_size, sequence_length, orig_dim)
        # Project down to the PCA space.
        reduced = self.pca_layer(emb)          # shape: (batch_size, sequence_length, target_dim)
        # Reconstruct back to the original dimension.
        reconstructed = self.reconstruction_layer(reduced)  # shape: (batch_size, sequence_length, orig_dim)
        return reconstructed

def apply_post_embedding_pca(model, reduction_factor=0.5, freeze_pca=True):
    """
    Replaces the model's input embedding with a custom PCA-based bottleneck module.
    :param model: The transformer model.
    :param reduction_factor: Fraction by which to reduce the embedding dimension.
    :param freeze_pca: If True, the PCA projection is frozen.
    :return: The model with updated input embeddings.
    """
    orig_embedding = model.get_input_embeddings()  # e.g., nn.Embedding instance.
    new_embedding = PCABottleneckEmbedding(orig_embedding, reduction_factor, freeze_pca)
    model.set_input_embeddings(new_embedding)
    return model


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
    pca_embeddings = args.pca_embeddings
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
        model = apply_post_embedding_pca(model, reduction_factor=0.5, freeze_pca=True)

    model.to(device)

    if compile_model:
        torch.compiler.cudagraph_mark_step_begin()
        model = torch.compile(model)
        original_forward = model.forward
        def new_forward(*args, **kwargs):
            kwargs.pop("num_items_in_batch", None)
            return original_forward(*args, **kwargs)
        model.forward = new_forward

    return model
