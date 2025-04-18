from pprint import pprint

from torch import nn
from transformers.modeling_outputs import MaskedLMOutput
from transformers import PreTrainedModel


class NucleotideModelWithPCA(PreTrainedModel):
    def __init__(self, config, base_model, pca_dim):
        super().__init__(config)
        self.model = base_model
        hidden_size = self.model.config.hidden_size
        self.pca_proj = nn.Linear(hidden_size, pca_dim, bias=False)
        self.layernorm = nn.LayerNorm(pca_dim)



    def forward(self, *args, **kwargs):
        kwargs.pop("num_items_in_batch", None)

        output = self.model(
            *args,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        if not self.training:
            pprint(output.loss)

        last_hidden = output.hidden_states[-1]
        pooled = last_hidden[:, 0]
        pca_emb = self.layernorm(self.pca_proj(pooled))

        return MaskedLMOutput(
            loss=output.loss,
            logits=output.logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )
