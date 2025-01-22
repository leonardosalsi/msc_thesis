from typing import List
from transformers import EsmTokenizer

class OverlappingEsmTokenizerWithNSkipping(EsmTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Fully override the parent method, ignoring the parent's logic.
        We do our own logic to produce overlapping 6-mers from the entire string.
        """
        k = 6
        tokens = []
        for i in range(len(text) - k + 1):
            tokens.append(text[i: i + k])
        return tokens