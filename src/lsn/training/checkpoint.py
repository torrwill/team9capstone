"""Checkpointing — atomic save, full-state self-describing dict, HF gating.

Refactored from notebook cell 22. HF-side ops dispatched via HFStore (spec §9.3);
torch.load uses weights_only=False (spec §14 — required for legacy checkpoints
with optimizer_state_dict).
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from lsn.training.hf_store import HFStore

logger = logging.getLogger(__name__)


LAST_CKPT_NAME = "last_checkpoint.pt"
BEST_CKPT_NAME = "best_model.pt"


def freeze_bn_stats(model: nn.Module) -> int:
    """Put BatchNorm layers in eval mode if their direct parameters are all frozen.

    Prevents the running_mean / running_var from drifting on small batches
    when you don't want them to update (i.e., the frozen EfficientNet stem).
    Call this after model.train() at the start of each epoch.

    Returns the number of BN modules switched to eval mode.
    """
    count = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                          nn.SyncBatchNorm)):
            params = list(m.parameters(recurse=False))
            if params and not any(p.requires_grad for p in params):
                m.eval()
                count += 1
    return count


def _model_state_dict(model: nn.Module) -> dict:
    """Unwrap DataParallel so checkpoints stay portable across GPU configs."""
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module.state_dict()
    return model.state_dict()


def _load_into_model(model: nn.Module, state: dict) -> None:
    """Mirror of _model_state_dict — handles DP wrapper on load."""
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)


def save_checkpoint_safe(
    model, optimizer, scaler, *,
    epoch, train_loss, val_loss, best_val_loss, history,
    save_path: Path,
    remote: "HFStore | None" = None,
) -> None:
    """Save FULL training state — model + optimizer + scaler + best_val_loss + history —
    in a single self-describing checkpoint.

    Atomic-ish via tmp file: write to tmp first, then copy to final path so a crash
    mid-write can't leave a corrupt last_checkpoint.pt. If `remote` is provided,
    upload via HFStore (failures logged, never raised — local copy is safe).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch":                epoch,
        "model_state_dict":     _model_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict":    scaler.state_dict() if scaler is not None else None,
        "train_loss":           train_loss,
        "val_loss":             val_loss,
        "best_val_loss":        best_val_loss,
        "history":              history,
    }

    # Write to a temp file first, then copy to the final destination.
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="checkpoint_tmp_", suffix=".pt")
    os.close(tmp_fd)
    try:
        torch.save(checkpoint, tmp_path)
        shutil.copy2(tmp_path, save_path)

        if remote is not None:
            remote.upload(
                Path(tmp_path),
                save_path.name,
                commit_message=f"epoch {epoch} | val_loss={val_loss:.4f}",
            )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def try_resume(
    model, optimizer, scaler, device,
    local_dir: Path,
    remote: "HFStore | None",
) -> tuple[int, float, list]:
    """Try to resume training in priority order:

        1) remote.try_download(LAST_CKPT_NAME)
        2) remote.try_download(BEST_CKPT_NAME)
        3) local_dir / LAST_CKPT_NAME
        4) local_dir / BEST_CKPT_NAME
        5) fresh — return (0, float('inf'), [])

    Loads via torch.load(..., weights_only=False) — required because legacy
    checkpoints contain optimizer_state_dict (not weight tensors only).

    Returns (start_epoch, best_val_loss, history). start_epoch is the last
    COMPLETED epoch; the loop begins at start_epoch + 1.
    """
    local_dir = Path(local_dir)

    candidates: list[tuple[str, str]] = [
        ("hf",    LAST_CKPT_NAME),
        ("hf",    BEST_CKPT_NAME),
        ("local", LAST_CKPT_NAME),
        ("local", BEST_CKPT_NAME),
    ]

    for source, fname in candidates:
        try:
            if source == "hf":
                if remote is None:
                    continue
                path = remote.try_download(fname, local_dir)
                if path is None:
                    continue
            else:
                path = local_dir / fname
                if not path.exists():
                    continue

            ckpt = torch.load(path, map_location=device, weights_only=False)
            _load_into_model(model, ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if scaler is not None and ckpt.get("scaler_state_dict") is not None:
                scaler.load_state_dict(ckpt["scaler_state_dict"])

            start_epoch = ckpt["epoch"]
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            history = ckpt.get("history", [])
            logger.info(
                "Resumed from %s/%s: epoch=%d, best_val_loss=%.4f, history_len=%d",
                source, fname, start_epoch, best_val_loss, len(history),
            )
            return start_epoch, best_val_loss, history

        except Exception as e:
            logger.info("could not load %s/%s: %s", source, fname, e)
            continue

    logger.info("No checkpoint found — starting fresh.")
    return 0, float("inf"), []
