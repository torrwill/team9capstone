from pathlib import Path

import numpy as np

from lsn.data.splits import create_paper_split


def _make_synthetic_dataset(tmp_path: Path, speakers: list[str], n_per: int = 200):
    """Build a fake speaker_dir/*.npz layout."""
    paths = []
    for sid in speakers:
        d = tmp_path / sid
        d.mkdir(exist_ok=True)
        for i in range(n_per):
            p = d / f"{sid}_{i:04d}.npz"
            np.savez(p, frames=np.zeros((75, 46, 140), dtype=np.float32),
                     label="hi")
            paths.append(p)
    return paths


def test_paper_split_sizes(tmp_path):
    paths = _make_synthetic_dataset(
        tmp_path, ["s1", "s2", "s3", "s4", "s5"], n_per=200,
    )
    train, test = create_paper_split(
        paths, speakers=["s1", "s2", "s3", "s4", "s5"],
        samples_per_speaker=200, train_size=450, seed=42,
    )
    assert len(train) == 450
    assert len(test) == 550


def test_paper_split_deterministic(tmp_path):
    paths = _make_synthetic_dataset(
        tmp_path, ["s1", "s2", "s3", "s4", "s5"], n_per=200,
    )
    train1, test1 = create_paper_split(
        paths, speakers=["s1", "s2", "s3", "s4", "s5"],
        samples_per_speaker=200, train_size=450, seed=42,
    )
    train2, test2 = create_paper_split(
        paths, speakers=["s1", "s2", "s3", "s4", "s5"],
        samples_per_speaker=200, train_size=450, seed=42,
    )
    assert [str(p) for p in train1] == [str(p) for p in train2]
    assert [str(p) for p in test1] == [str(p) for p in test2]


def test_paper_split_disjoint(tmp_path):
    paths = _make_synthetic_dataset(
        tmp_path, ["s1", "s2", "s3", "s4", "s5"], n_per=200,
    )
    train, test = create_paper_split(
        paths, speakers=["s1", "s2", "s3", "s4", "s5"],
        samples_per_speaker=200, train_size=450, seed=42,
    )
    assert set(map(str, train)).isdisjoint(set(map(str, test)))
