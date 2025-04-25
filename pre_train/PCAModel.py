import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import MaskedLMOutput
from transformers import PreTrainedModel

class NucleotideModelWithPCA(PreTrainedModel):
    def __init__(
        self,
        config,
        base_model,
        pca_dim,
        aux_loss_weight=0.1,
        temperature=0.1,
        pooling_method="mean",
    ):
        super().__init__(config)
        self.model = base_model
        hidden_size = self.model.config.hidden_size
        self.pca_proj = nn.Linear(hidden_size, pca_dim, bias=False)
        self.layernorm = nn.LayerNorm(pca_dim)
        self.aux_loss_weight = aux_loss_weight
        self.temperature = temperature
        self.pooling_method = pooling_method

    def forward(self, *args, **kwargs):
        kwargs.pop("num_items_in_batch", None)

        attention_mask = kwargs.get("attention_mask", None)

        output = self.model(
            *args,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        last_hidden = output.hidden_states[-1]

        if self.pooling_method == "mean":
            if attention_mask is None:
                raise ValueError("attention_mask is required for mean pooling.")
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * mask, dim=1)
            lengths = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / lengths # Mean Pooling of embeddings
        else:  # default to CLS
            pooled = last_hidden[:, 0] # Only CLS Embeddings

        pca_emb = self.layernorm(self.pca_proj(pooled))
        pca_emb = F.normalize(pca_emb, dim=-1)

        bsz = pca_emb.shape[0] // 2
        z1, z2 = pca_emb[:bsz], pca_emb[bsz:]

        logits = z1 @ z2.T / self.temperature
        labels = torch.arange(bsz, device=z1.device)
        contrastive_loss = F.cross_entropy(logits, labels)

        total_loss = None
        if output.loss is not None:
            total_loss = output.loss + self.aux_loss_weight * contrastive_loss

        return MaskedLMOutput(
            loss=total_loss,
            logits=output.logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )
