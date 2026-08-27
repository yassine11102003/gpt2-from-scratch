"""
Simple regex-based tokenizers built from scratch.
For real use, prefer tiktoken (GPT-2 BPE tokenizer).
"""

import re


class TokenizerV1:
    """Basic tokenizer — raises KeyError for unknown words."""

    def __init__(self, vocab: dict):
        self.str_to_int = vocab
        self.int_to_str = {id: token for token, id in vocab.items()}

    def encode(self, text: str) -> list[int]:
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [t.strip() for t in tokens if t.strip()]
        return [self.str_to_int[t] for t in tokens]

    def decode(self, ids: list[int]) -> str:
        text = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(r'\s+([,.?!"()\'])', r'\1', text)


class TokenizerV2:
    """Tokenizer with <|unk|> and <|endoftext|> special tokens."""

    def __init__(self, vocab: dict):
        self.str_to_int = vocab
        self.int_to_str = {id: token for token, id in vocab.items()}

    def encode(self, text: str) -> list[int]:
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [t.strip() for t in tokens if t.strip()]
        tokens = [t if t in self.str_to_int else "<|unk|>" for t in tokens]
        return [self.str_to_int[t] for t in tokens]

    def decode(self, ids: list[int]) -> str:
        text = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(r'\s+([,.?!"()\'])', r'\1', text)


def build_vocab(text: str) -> dict:
    """Build a vocabulary from raw text."""
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    tokens = sorted(set(t.strip() for t in tokens if t.strip()))
    tokens += ["<|endoftext|>", "<|unk|>"]
    return {token: id for id, token in enumerate(tokens)}
