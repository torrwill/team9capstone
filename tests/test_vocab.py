import torch
from lsn.data import vocab as v


def test_vocab_constants_load_bearing():
    # vocab_size=40 is preserved verbatim from trained checkpoints (spec §4.1).
    # CHARS is the actual 27-char alphabet; vocab_size>len(CHARS) is intentional.
    assert v.VOCAB_SIZE == 40
    assert v.NUM_CLASSES == 41
    assert v.BLANK_INDEX == 0
    assert v.CHARS[0] == " "
    assert v.CHARS[-1] == "z"
    assert len(v.CHARS) == 27


def test_encode_decode_roundtrip():
    text = "set white at b nine again"
    ids = v.encode_text(text)
    assert isinstance(ids, torch.Tensor)
    assert ids.dtype == torch.long
    decoded = v.decode_ids(ids.tolist())
    assert decoded == text


def test_decode_skips_blank():
    # blank index 0 must be filtered out during decode (CTC convention)
    ids = [0, v.char_to_idx["a"], 0, v.char_to_idx["b"], 0]
    assert v.decode_ids(ids) == "ab"


def test_encode_lowercases():
    assert v.decode_ids(v.encode_text("HELLO").tolist()) == "hello"
