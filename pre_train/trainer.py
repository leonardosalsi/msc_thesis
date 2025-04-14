import torch
from torch.utils.data import DataLoader
from transformers import Trainer

from pre_train.dataset import get_original_training_dataset


class EWCTrainer(Trainer):
    def __init__(self, *args, ewc_lambda=0.0, fisher_matrix=None, original_params=None, **kwargs):
        """
        Custom Trainer that adds an EWC loss term during training.

        :param ewc_lambda: Strength of EWC regularization (lambda in formula).
        :param fisher_matrix: Fisher information (diagonal) for each parameter (keyed by param name).
        :param original_params: Original parameter values (keyed by param name).
        """
        super().__init__(*args, **kwargs)
        self.ewc_lambda = ewc_lambda
        self.fisher_matrix = fisher_matrix if fisher_matrix is not None else {}
        self.original_params = original_params if original_params is not None else {}
        for name, param in self.model.named_parameters():
            if name in self.fisher_matrix:
                self.fisher_matrix[name] = self.fisher_matrix[name].to(param.device).to(param.dtype)
            if name in self.original_params:
                self.original_params[name] = self.original_params[name].to(param.device).to(param.dtype)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the loss with an added EWC penalty term.
        """
        outputs = model(**inputs)
        if hasattr(outputs, "loss"):
            base_loss = outputs.loss
        elif isinstance(outputs, tuple):
            base_loss = outputs[0]
        else:
            base_loss = outputs["loss"] if "loss" in outputs else outputs[0]
        ewc_loss = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher_matrix:
                diff = param - self.original_params[name]
                ewc_loss += (self.fisher_matrix[name] * diff ** 2).sum()
        ewc_loss = 0.5 * self.ewc_lambda * ewc_loss  # multiply by lambda/2
        total_loss = base_loss + ewc_loss
        return (total_loss, outputs) if return_outputs else total_loss

def compute_fisher(model, device, orig_data_loader):
    """
    Compute diagonal Fisher Information for all model parameters on the original domain data.
    Returns a dict mapping parameter names to their Fisher values (tensor of same shape).
    """

    model.eval()
    fisher = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters()}
    for batch in orig_data_loader:
        for key, value in batch.items():
            batch[key] = value.to(device)
        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        model.zero_grad()
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += (param.grad ** 2)
    for name in fisher:
        fisher[name] /= len(orig_data_loader)
    return fisher

def get_trainer(args, training_args, model, device, tokenizer, training_dataset, eval_dataset, data_collator, num_tokens):
    ewc_lambda = args.ewc_lambda
    if ewc_lambda and ewc_lambda > 0:
        original_dataset = get_original_training_dataset(args)

        def tokenize_function(examples):
            return tokenizer(examples['sequence'], max_length=num_tokens, truncation=True, padding=True)

        tokenized_dataset = original_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=original_dataset.column_names)

        orig_data_loader = DataLoader(
            tokenized_dataset.with_format("torch"),
            batch_size=args.train_size,
            collate_fn=data_collator,
            shuffle=False,
        )

        fisher_matrix = compute_fisher(model, device, orig_data_loader)
        original_params = {name: param.detach().clone() for name, param in model.named_parameters()}
        trainer = EWCTrainer(
            model=model,
            args=training_args,
            train_dataset=training_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            ewc_lambda=ewc_lambda,
            fisher_matrix=fisher_matrix,
            original_params=original_params
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=training_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

    return trainer
