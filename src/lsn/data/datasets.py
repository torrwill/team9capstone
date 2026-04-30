"""Lip-reading datasets and CTC collate function.

The .npz file contract (spec §11):
    - "frames": numpy array of shape (75, 46, 140) or (75, 46, 140, 1)
    - "label":  string (the spoken text)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from lsn.data.vocab import encode_text


class GridLipReadingDataset(Dataset):
    """
    Loads preprocessed .npz files.

    Expected .npz keys:
        - "frames": numpy array, shape (75, 46, 140) or (75, 46, 140, 1)
        - "label":  string (the spoken text)

    If your .npz uses different key names, update FRAME_KEY and LABEL_KEY below.
    """
    FRAME_KEY = "frames"   # change if your npz uses a different key
    LABEL_KEY = "label"    # change if your npz uses a different key (e.g., "text")

    def __init__(self, npz_paths):
        self.paths = [str(p) for p in npz_paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        data = np.load(path, allow_pickle=True)

        frames = data[self.FRAME_KEY]          # (75, 46, 140) or (75, 46, 140, 1)
        text = str(data[self.LABEL_KEY]).lower().strip()

        # Remove trailing channel dim if present
        if frames.ndim == 4 and frames.shape[-1] == 1:
            frames = frames[..., 0]            # (75, 46, 140)

        if tuple(frames.shape) != (75, 46, 140):
            raise ValueError(
                f"unexpected frames shape {tuple(frames.shape)} in {path}; "
                f"see docs/data-format.md for the expected .npz contract"
            )

        frames = torch.tensor(frames, dtype=torch.float32)
        target = encode_text(text)
        target_length = len(target)

        return {
            "frames": frames,                  # (75, 46, 140)
            "text": text,
            "target": target,                  # (target_len,)
            "target_length": target_length,
            "path": path,
        }


# collate function - preparing the results for the CTC function
def grid_collate_fn(batch):
    frames = torch.stack([item["frames"] for item in batch], dim=0)
    targets = torch.cat([item["target"] for item in batch], dim=0)

    target_lengths = torch.tensor(
        [item["target_length"] for item in batch],
        dtype=torch.long,
    )

    input_lengths = torch.full(
        size=(len(batch),),
        fill_value=frames.shape[1],   # 75 frames (LSN paper)
        dtype=torch.long,
    )

    texts = [item["text"] for item in batch]
    paths = [item["path"] for item in batch]

    return {
        "frames": frames,                  # (B, 75, 46, 140)
        "targets": targets,                # (sum target lengths,)
        "target_lengths": target_lengths,  # (B,)
        "input_lengths": input_lengths,    # (B,)
        "texts": texts,
        "paths": paths,
    }


# ============================================================
# LRS2-specific dataset: skips target encoding
# ============================================================
# LRS2 labels contain characters outside the GRID vocabulary (apostrophes,
# digits) that crash encode_text. At inference time we never need the
# encoded target — we only use the raw string for the reference in
# evaluation. So this class returns text but not target.

class LRS2Dataset(Dataset):
    FRAME_KEY = "frames"
    LABEL_KEY = "label"

    def __init__(self, npz_paths):
        self.paths = [str(p) for p in npz_paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        data = np.load(path, allow_pickle=True)

        frames = data[self.FRAME_KEY]
        text = str(data[self.LABEL_KEY]).lower().strip()

        if frames.ndim == 4 and frames.shape[-1] == 1:
            frames = frames[..., 0]

        if tuple(frames.shape) != (75, 46, 140):
            raise ValueError(
                f"unexpected frames shape {tuple(frames.shape)} in {path}; "
                f"see docs/data-format.md for the expected .npz contract"
            )

        frames = torch.tensor(frames, dtype=torch.float32)

        # NO target encoding — LRS2 labels contain chars (apostrophes, digits)
        # that aren't in the GRID vocab. We return a placeholder so the
        # collate function doesn't crash on missing keys.
        return {
            "frames":        frames,
            "text":          text,
            "target":        torch.zeros(1, dtype=torch.long),  # placeholder
            "target_length": 1,                                  # placeholder
            "path":          path,
        }
