import torch.nn.functional as F
import joblib
import torch
from torch import nn


class NucleotideModelWithPCA(nn.Module):
    def __init__(self, base_model, pca_path):
        super().__init__()
        self.base_model = base_model
        self.pca = joblib.load(pca_path)

        self.pca_mean = torch.tensor(self.pca.mean_, dtype=torch.float32)
        self.pca_components = torch.tensor(self.pca.components_, dtype=torch.float32)

        self.pca_proj = nn.Linear(
            self.pca_components.shape[1],
            self.pca_components.shape[0],
            bias=False
        )
        self.pca_proj.weight.data = self.pca_components
        self.pca_proj.weight.requires_grad = False

    def forward(self, *args, **kwargs):
        output = self.base_model(*args, **kwargs)

        last_hidden = output.last_hidden_state

        pooled = last_hidden.mean(dim=1)

        centered = pooled - self.pca_mean.to(pooled.device)
        reduced = self.pca_proj(centered)

        return {"reduced_embedding": reduced, "transformer_output": output}
