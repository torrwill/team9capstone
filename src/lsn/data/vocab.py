"""Character vocabulary and encode/decode helpers.

Note on `VOCAB_SIZE = 40` (load-bearing — see spec §4.1):
The trained checkpoints were instantiated with `vocab_size=40`, producing a
classifier of shape `Linear(1024, 41)` (40 + 1 CTC blank). The actual
encodable alphabet (CHARS below) is only 27 characters, so the classifier
has 41 channels of which 28 are populated (blank + space + 26 letters).
Do NOT "fix" VOCAB_SIZE to len(CHARS) — that would change the classifier's
weight shape and break strict=True load of every existing .pt file.
"""
from __future__ import annotations

import string

import torch

# 27-char alphabet: space + a-z (lowercase, no digits/punctuation in GRID).
CHARS: list[str] = [" "] + list(string.ascii_lowercase)

# Reserved for CTC blank — PyTorch CTCLoss(blank=0) convention.
BLANK_INDEX: int = 0

# IDs start at 1 (0 is reserved for blank).
char_to_idx: dict[str, int] = {ch: i + 1 for i, ch in enumerate(CHARS)}
idx_to_char: dict[int, str] = {i + 1: ch for i, ch in enumerate(CHARS)}

# Load-bearing constants — see module docstring.
VOCAB_SIZE: int = 40
NUM_CLASSES: int = VOCAB_SIZE + 1   # 41 — includes blank


def encode_text(text: str) -> torch.Tensor:
    """Encode lowercased text → tensor of int ids."""
    text = text.lower().strip()
    ids = [char_to_idx[ch] for ch in text]
    return torch.tensor(ids, dtype=torch.long)


def decode_ids(ids) -> str:
    """Decode a sequence of int ids → string. Skips BLANK_INDEX (CTC)."""
    chars = []
    for idx in ids:
        idx = int(idx)
        if idx == BLANK_INDEX:
            continue
        chars.append(idx_to_char[idx])
    return "".join(chars)
