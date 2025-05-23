import math
from transformers import AutoTokenizer
from config import models_cache_dir
import random
from typing import List
from transformers import EsmTokenizer


class OverlappingTokenizer(EsmTokenizer):
    """
    Fully override the parent tokenization method, ignoring the parent's logic.
    We do our own logic to produce overlapping 6-mers from the entire string.
    Same as described in the paper, we sample a number from 0 to total_generated_tokens - num_tokens and
    start tokenizing from there until we have num_tokens tokens.
    When an N is encountered, it is tokenized alone, i.e., never with other nucleotides.
    It returns num_count tokens including the special tokens.
    """
    def __init__(self, num_tokens, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens - 2

    def tokenize(self, text: str, **kwargs) -> List[str]:
        k = 6
        tokens = []
        i = 0
        while i < (len(text) - k + 1):
            token = text[i:i+k]
            if 'N' not in token:
                tokens.append(text[i: i + k])
            else:
                if i == (len(text) - k):
                    for t in token:
                        tokens.append(t)
                else:
                    tokens.append("N")
                    i += k - 1
            i += 1
        length = len(tokens)
        end_idx = length - self.num_tokens
        if end_idx <= 0:
            idx = 0
        else:
            idx = random.randint(0, end_idx)
        tokens = tokens[idx:idx+self.num_tokens]
        return tokens

def get_tokenizer(args):
    """
    Loads and returns the tokenizer along with the computed number of tokens.

    Depending on the value of args.tokenizer, this function either:
      - Loads the default tokenizer from the pretrained model.
      - Loads a custom OverlappingTokenizer with a specified number of tokens.

    The number of tokens is determined based on the provided chunk_size, where
    num_tokens = (floor(chunk_size / 1000)) * 1000.

    :param args: Arguments containing the tokenizer type and chunk_size.
    :return: A tuple (tokenizer, num_tokens)
    """
    selected_tokenizer = args.tokenizer
    chunk_size = args.chunk_size
    num_tokens = args.num_tokens

    if selected_tokenizer == "default":
        tokenizer = AutoTokenizer.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
            model_max_length=2048,
            cache_dir=models_cache_dir,
            remove_columns=['sequence'],
            trust_remote_code=True,
            local_files_only=True
        )
    elif selected_tokenizer == "overlapping":
        tokenizer = OverlappingTokenizer(
            vocab_file="model_configs/vocab.txt",
            model_max_length=2048,
            num_tokens=num_tokens
        )
    else:
        raise ValueError("The specified tokenizer does not exist.")
    return tokenizer, num_tokens


def get_eval_tokenizer(args, repo=None):
    """
    We use default tokenizer for all tasks for fair comparison.
    tokenizer = None
    if repo is None:
        repo = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    if 'default' in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            repo,
            model_max_length=2048,
            cache_dir=models_cache_dir,
            remove_columns=['sequence'],
            trust_remote_code=True,
            local_files_only=True
        )
    elif 'overlap' in args.model_name:
        if '2kb' in args.model_name:
            num_tokens = 2000
        else:
            num_tokens = 1000

        tokenizer = OverlappingTokenizer(
            vocab_file="model_configs/vocab.txt",
            model_max_length=2048,
            num_tokens=num_tokens
        )

    if tokenizer is None:
        raise ValueError("The specified tokenizer does not exist.")"""
    if repo is None:
        repo = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    tokenizer = AutoTokenizer.from_pretrained(
        repo,
        model_max_length=2048,
        cache_dir=models_cache_dir,
        remove_columns=['sequence'],
        trust_remote_code=True,
        local_files_only=True
    )

    return tokenizer