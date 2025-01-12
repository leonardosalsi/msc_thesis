import math
from typing import List

from transformers import EsmTokenizer, AutoTokenizer

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

def load_vocab_file(vocab_file):
    with open(vocab_file, "r") as f:
        lines = f.read().splitlines()
        return [l.strip() for l in lines]

class OverlappingEsmTokenizer(EsmTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # anything else you need to init?

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Fully override the parent method, ignoring the parent's logic.
        We do our own logic to produce overlapping 6-mers from the entire string.
        """
        text = text.upper().replace(" ", "").replace("\n", "")
        k = 6
        tokens = []
        for i in range(len(text) - k + 1):
            tokens.append(text[i: i + k])
        return tokens


from config import models_cache_dir, datasets_cache_dir
from datasets import load_dataset

if __name__ == "__main__":
    multi_species_genomes_small = load_dataset(
        "InstaDeepAI/multi_species_genomes",
        cache_dir=datasets_cache_dir,
        name="1kbp",
        trust_remote_code=True
    )

    print(multi_species_genomes_small['train'][0])
    print(multi_species_genomes_small['train'][1])

    default_tokenizer = EsmTokenizer.from_pretrained(
        "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        cache_dir=models_cache_dir,
        trust_remote_code=True,
        local_files_only=True
    )

    overlapping_tokenizer = OverlappingEsmTokenizer.from_pretrained(
        "/shared/models",
        model_max_length=2048,
    )

    sequence = multi_species_genomes_small['train'][0]['sequence']
    default_encoded = default_tokenizer(sequence)
    overlapping_encoded = overlapping_tokenizer(sequence)

    print(sequence)
    print("======================")
    print("DefaultEncoded:", default_encoded)
    print(len(default_tokenizer.convert_ids_to_tokens(default_encoded["input_ids"])))
    print(len(default_encoded["input_ids"]))
    print("======================")
    print("OverlappingEncoded:", overlapping_encoded)
    print(len(overlapping_tokenizer.convert_ids_to_tokens(overlapping_encoded["input_ids"])))
    print(len(overlapping_encoded["input_ids"]))
    print(overlapping_tokenizer)
