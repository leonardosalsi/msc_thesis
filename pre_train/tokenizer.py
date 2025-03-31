import math
from transformers import AutoTokenizer
from config import models_cache_dir
from overrides.tokenizer.OverlappingTokenizer import OverlappingTokenizer

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
    num = math.floor(chunk_size / 1000)
    num_tokens = num * 1000

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
