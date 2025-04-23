class PCACollator:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(self, examples):
        from transformers.data.data_collator import torch_default_data_collator
        import torch

        batch = self.tokenizer.pad(
            examples,
            padding=True,
            return_tensors="pt"
        )
        input_ids = batch["input_ids"]

        # Generate two masked versions per sequence
        def mask_inputs(ids):
            labels = ids.clone()
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
            probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # Only compute loss on masked tokens
            ids[masked_indices] = self.tokenizer.mask_token_id
            return ids, labels

        input_ids1, labels1 = mask_inputs(input_ids.clone())
        input_ids2, labels2 = mask_inputs(input_ids.clone())

        # Stack views
        input_ids = torch.cat([input_ids1, input_ids2], dim=0)
        labels = torch.cat([labels1, labels2], dim=0)
        attention_mask = torch.cat([batch["attention_mask"]] * 2, dim=0)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }
