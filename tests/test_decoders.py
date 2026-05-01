import pytest
import torch

from lsn.evaluation.decoders import beam_decode, greedy_decode
from lsn.data.vocab import BLANK_INDEX, char_to_idx


def test_greedy_decode_shapes_and_types():
    T, B, C = 75, 2, 41
    log_probs = torch.log_softmax(torch.randn(T, B, C), dim=-1)
    out = greedy_decode(log_probs)
    assert isinstance(out, list)
    assert len(out) == B
    assert all(isinstance(s, str) for s in out)


def test_greedy_decode_deterministic_input():
    """Hand-construct an argmax sequence and verify the decoded string.

    Sequence (T=6): [a, a, blank, b, b, c]
    CTC collapse-then-drop-blanks → "abc"
    """
    a = char_to_idx["a"]
    b = char_to_idx["b"]
    c = char_to_idx["c"]
    seq = [a, a, BLANK_INDEX, b, b, c]
    T, B, C = len(seq), 1, 41
    log_probs = torch.full((T, B, C), -10.0)
    for t, idx in enumerate(seq):
        log_probs[t, 0, idx] = 0.0
    log_probs = torch.log_softmax(log_probs, dim=-1)

    decoded = greedy_decode(log_probs)
    assert decoded == ["abc"], f"expected ['abc'], got {decoded!r}"


def test_beam_decode_requires_input_lengths():
    """beam_decode signature includes input_lengths (spec §4.6, B3 fix)."""
    T, B, C = 75, 2, 41
    log_probs = torch.log_softmax(torch.randn(T, B, C), dim=-1)
    input_lengths = torch.tensor([T, T], dtype=torch.long)
    try:
        out = beam_decode(log_probs, input_lengths=input_lengths, beam_width=10)
    except (ImportError, OSError):
        pytest.skip("torchaudio not available in this environment")
    assert isinstance(out, list)
    assert len(out) == B
    assert all(isinstance(s, str) for s in out)


def test_beam_decode_deterministic_input():
    """A peaked argmax sequence should produce the same string under greedy
    and beam decoding (when the hypothesis is unambiguous)."""
    a = char_to_idx["a"]
    seq = [a, a, BLANK_INDEX, a]
    T, B, C = len(seq), 1, 41
    log_probs = torch.full((T, B, C), -10.0)
    for t, idx in enumerate(seq):
        log_probs[t, 0, idx] = 0.0
    log_probs = torch.log_softmax(log_probs, dim=-1)
    input_lengths = torch.tensor([T], dtype=torch.long)

    g = greedy_decode(log_probs)
    try:
        b = beam_decode(log_probs, input_lengths=input_lengths, beam_width=10)
    except (ImportError, OSError):
        pytest.skip("torchaudio not available in this environment")
    assert g == b == ["aa"], f"greedy={g}, beam={b}"
