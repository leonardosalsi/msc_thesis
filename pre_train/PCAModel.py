import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import MaskedLMOutput
from transformers import PreTrainedModel

class NucleotideModelWithPCA(PreTrainedModel):
    def __init__(self, config, base_model, pca_dim, aux_loss_weight=0.1, temperature=0.1):
        super().__init__(config)
        self.model = base_model
        hidden_size = self.model.config.hidden_size
        self.pca_proj = nn.Linear(hidden_size, pca_dim, bias=False)
        self.layernorm = nn.LayerNorm(pca_dim)
        self.aux_loss_weight = aux_loss_weight
        self.temperature = temperature

    def forward(self, *args, **kwargs):
        kwargs.pop("num_items_in_batch", None)

        output = self.model(
            *args,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        last_hidden = output.hidden_states[-1]
        pooled = last_hidden[:, 0]
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

