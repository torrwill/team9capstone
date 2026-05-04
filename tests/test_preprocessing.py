import numpy as np
import pytest
from lsn.preprocessing import load_align, normalize


def test_load_align_basic(tmp_path):
    align = tmp_path / "test.align"
    align.write_text(
        "0 5000 sil\n"
        "5000 10000 bin\n"
        "10000 15000 blue\n"
        "15000 20000 sp\n"
        "20000 25000 at\n"
    )
    assert load_align(align) == "bin blue at"


def test_load_align_all_sil(tmp_path):
    align = tmp_path / "sil.align"
    align.write_text("0 5000 sil\n5000 10000 sp\n")
    assert load_align(align) == ""


def test_normalize_shape():
    frames = np.random.randint(0, 255, (75, 46, 140), dtype=np.uint8)
    out = normalize(frames)
    assert out.shape == (75, 46, 140)
    assert out.dtype == np.float32


def test_normalize_zero_mean():
    frames = np.full((75, 46, 140), 128, dtype=np.uint8)
    out = normalize(frames)
    assert abs(float(out.mean())) < 1e-3
