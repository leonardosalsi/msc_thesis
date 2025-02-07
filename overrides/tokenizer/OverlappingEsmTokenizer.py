import random
from typing import List
from transformers import EsmTokenizer

"""
Fully override the parent method, ignoring the parent's logic.
We do our own logic to produce overlapping 6-mers from the entire string.
Same as described in the paper, we sample a number from 0 to total_generated_tokens - num_tokens and
start tokenizing from there until we have num_tokens tokens.
N is not treated differently than other nucleotides.
"""
class OverlappingEsmTokenizer(EsmTokenizer):
    def __init__(self, num_tokens, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens

    def tokenize(self, text: str, **kwargs) -> List[str]:
        k = 6
        tokens = []
        for i in range(len(text) - k + 1):
            tokens.append(text[i: i + k])
        length = len(tokens)
        end_idx = length - self.num_tokens
        idx = random.randint(0, end_idx)
        tokens = tokens[idx:idx + self.num_tokens]
        return tokens