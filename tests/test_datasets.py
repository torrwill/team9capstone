from pathlib import Path

import numpy as np
import pytest
import torch

from lsn.data.datasets import GridLipReadingDataset, grid_collate_fn


def _write_npz(path: Path, label: str = "set white at b nine again",
               with_channel: bool = False) -> None:
    if with_channel:
        frames = np.zeros((75, 46, 140, 1), dtype=np.float32)
    else:
        frames = np.zeros((75, 46, 140), dtype=np.float32)
    np.savez(path, frames=frames, label=label)


def test_grid_dataset_loads_npz_no_channel(tmp_path):
    p = tmp_path / "s1_001.npz"
    _write_npz(p)
    ds = GridLipReadingDataset([p])
    sample = ds[0]
    assert sample["frames"].shape == (75, 46, 140)
    assert sample["frames"].dtype == torch.float32
    assert sample["text"] == "set white at b nine again"
    assert sample["target_length"] == len(sample["text"])
    assert sample["path"] == str(p)


def test_grid_dataset_loads_npz_with_channel(tmp_path):
    p = tmp_path / "s1_001.npz"
    _write_npz(p, with_channel=True)
    ds = GridLipReadingDataset([p])
    sample = ds[0]
    # trailing channel-1 dim is squeezed
    assert sample["frames"].shape == (75, 46, 140)


def test_grid_dataset_validates_shape(tmp_path):
    """Spec §9.6 — boundary validation with clear ValueError."""
    p = tmp_path / "bad.npz"
    np.savez(p, frames=np.zeros((75, 50, 140), dtype=np.float32),
             label="hello")
    ds = GridLipReadingDataset([p])
    with pytest.raises(ValueError):
        ds[0]


def test_grid_collate_fn_shapes(tmp_path):
    paths = []
    for i in range(2):
        p = tmp_path / f"s1_{i}.npz"
        _write_npz(p, label="hello")
        paths.append(p)

    ds = GridLipReadingDataset(paths)
    batch = [ds[0], ds[1]]
    out = grid_collate_fn(batch)

    assert out["frames"].shape == (2, 75, 46, 140)
    assert out["targets"].shape == (10,)             # 5 chars × 2 samples
    assert out["target_lengths"].tolist() == [5, 5]
    assert out["input_lengths"].tolist() == [75, 75]
    assert len(out["texts"]) == 2
    assert len(out["paths"]) == 2
