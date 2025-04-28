from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from transformers import PreTrainedModel
from dataclasses import dataclass

@dataclass
class MaskedPCALMOutput(MaskedLMOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    auxiliary_loss: Optional[torch.FloatTensor] = None
    model_loss: Optional[torch.FloatTensor] = None
    pca_embedding: Optional[torch.Tensor] = None

class EsmForMaskedLMPCA(PreTrainedModel):
    def __init__(
        self,
        config,
        base_model,
        pca_dim,
        aux_loss_weight=0.1,
        temperature=0.1,
        pca_embeddings="mean",
        gradient_accumulation_steps=1
    ):
        super().__init__(config)
        self.model = base_model
        hidden_size = self.model.config.hidden_size
        self.pca_proj = nn.Linear(hidden_size, pca_dim, bias=False)
        self.layernorm = nn.LayerNorm(pca_dim )
        self.aux_loss_weight = aux_loss_weight
        self.temperature = temperature
        self.pca_embeddings = pca_embeddings
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def forward(self, *args, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict", None)

        attention_mask = kwargs.get("attention_mask", None)

        output = self.model(
            *args,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        last_hidden = output.hidden_states[-1]

        if self.pca_embeddings == "mean":
            if attention_mask is None:
                raise ValueError("attention_mask is required for mean pooling.")
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * mask, dim=1)
            lengths = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / lengths
        else:
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

        if self.training and self.gradient_accumulation_steps > 1 and total_loss is not None:
            total_loss = total_loss / self.gradient_accumulation_steps

        return MaskedPCALMOutput(
            loss=total_loss,
            logits=output.logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            auxiliary_loss=(self.aux_loss_weight * contrastive_loss.detach()).float(),
            model_loss=output.loss.detach() if output.loss is not None else None,
            pca_embedding=pca_emb
        )

class EsmForSequenceClassificationPCA(PreTrainedModel):
    def __init__(self, pca_model: EsmForMaskedLMPCA, num_labels: int):
        super().__init__(pca_model.config)   # <-- important!
        self.pca_model = pca_model
        hidden_size = pca_model.pca_proj.out_features
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.pca_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        pooled_output = outputs.pca_embedding
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )