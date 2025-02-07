import random
from typing import List
from transformers import EsmTokenizer

"""
Fully override the parent method, ignoring the parent's logic.
We implement the same logic that was described in the paper, where they only generated 1000 tokens.
Same as described in the paper, we sample a number from 0 to total_generated_tokens - num_tokens and
start tokenizing from there until we have num_tokens tokens.
N is not treated differently than other nucleotides.
"""
class PaperTokenizer(EsmTokenizer):
    def __init__(self, num_tokens, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens - 2

    def tokenize(self, text: str, **kwargs) -> List[str]:
        k = 6
        tokens = []
        i = 0
        while i < (len(text) - k + 1):
            token = text[i:i + k]
            if 'N' not in token:
                tokens.append(text[i: i + k])
                i += 6
            else:
                if i == (len(text) - k):
                    for t in token:
                        tokens.append(t)
                else:
                    for t in token:
                        i += 1
                        if t == 'N':
                            tokens.append('N')

                            break
                        else:
                            tokens.append(t)


        length = len(tokens)
        end_idx = length - self.num_tokens
        idx = random.randint(0, end_idx)
        print(idx, end_idx)
        tokens = tokens[idx:idx + self.num_tokens]
        return tokens