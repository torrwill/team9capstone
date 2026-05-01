"""High-level training runner — wires Config → model + data + loop + checkpoint.

Refactored from notebook cells 24-25. Called by scripts/train.py.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# torch 2.3+ exposes GradScaler at torch.amp.GradScaler; older builds keep it
# at torch.cuda.amp.GradScaler. Mirror the compat shim used in tests.
if hasattr(torch.amp, "GradScaler"):
    from torch.amp import GradScaler
else:  # pragma: no cover - exercised on older torch only
    from torch.cuda.amp import GradScaler  # type: ignore[no-redef]

from lsn.config import Config
from lsn.data.datasets import GridLipReadingDataset, grid_collate_fn
from lsn.data.splits import create_paper_split
from lsn.data.vocab import BLANK_INDEX
from lsn.env import configure_cudnn, set_seed
from lsn.models import build_from_config, count_parameters
from lsn.training.checkpoint import (
    BEST_CKPT_NAME,
    LAST_CKPT_NAME,
    save_checkpoint_safe,
    try_resume,
)
from lsn.training.hf_store import HFStore
from lsn.training.loop import train_one_epoch, validate_one_epoch

logger = logging.getLogger(__name__)


def run(cfg: Config, *, data_dir: Path, ckpt_dir: Path,
        device: torch.device) -> None:
    """End-to-end training run from a Config. Called by scripts/train.py.

    Builds the per-experiment subdirectory `ckpt_dir / cfg.experiment_name`
    and passes it to checkpoint.save_*/try_resume. Caller (scripts/train.py)
    supplies the parent ckpt_dir; runner.run owns per-experiment naming.
    """
    set_seed(cfg.data.seed)
    configure_cudnn(benchmark=True)

    # Per-experiment ckpt dir
    run_ckpt_dir = ckpt_dir / cfg.experiment_name
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # HF gating
    remote: HFStore | None = None
    if cfg.checkpointing.hf_repo:
        subfolder = cfg.checkpointing.hf_subfolder or cfg.experiment_name
        remote = HFStore(cfg.checkpointing.hf_repo, subfolder)
        logger.info("HF enabled: repo=%s subfolder=%s",
                    cfg.checkpointing.hf_repo, subfolder)
    else:
        logger.info("HF disabled - local-only checkpointing")

    # Dataset + split
    npz_paths = sorted(Path(data_dir).glob("*/*.npz"))
    if not npz_paths:
        raise FileNotFoundError(
            f"No .npz files found in {data_dir} (expected speaker-subdir layout)"
        )
    train_paths, val_paths = create_paper_split(
        npz_paths,
        speakers=cfg.data.speakers,
        samples_per_speaker=cfg.data.samples_per_speaker,
        train_size=cfg.data.train_size,
        seed=cfg.data.seed,
    )

    train_loader = DataLoader(
        GridLipReadingDataset(train_paths),
        batch_size=cfg.training.batch_size, shuffle=True,
        collate_fn=grid_collate_fn,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=cfg.training.num_workers > 0,
        prefetch_factor=cfg.training.prefetch if cfg.training.num_workers > 0 else None,
        drop_last=False,
    )
    val_loader = DataLoader(
        GridLipReadingDataset(val_paths),
        batch_size=cfg.training.batch_size, shuffle=False,
        collate_fn=grid_collate_fn,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=cfg.training.num_workers > 0,
        prefetch_factor=cfg.training.prefetch if cfg.training.num_workers > 0 else None,
        drop_last=False,
    )
    logger.info(
        "train=%d batches=%d  val=%d batches=%d  effective_batch=%d",
        len(train_loader.dataset), len(train_loader),
        len(val_loader.dataset), len(val_loader),
        cfg.training.batch_size * cfg.training.accum_steps,
    )

    # Model
    model = build_from_config(cfg.model, device=device)
    counts = count_parameters(model)
    logger.info(
        "params: total=%(total)d trainable=%(trainable)d frozen=%(frozen)d",
        counts,
    )

    # Loss / optim / scaler
    loss_fn = nn.CTCLoss(blank=BLANK_INDEX, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.learning_rate,
    )
    amp_enabled = cfg.training.use_amp and device.type == "cuda"
    if hasattr(torch.amp, "GradScaler"):
        scaler = GradScaler("cuda", enabled=amp_enabled)
    else:  # pragma: no cover - exercised on older torch only
        scaler = GradScaler(enabled=amp_enabled)

    # Resume
    start_epoch, best_val_loss, history = try_resume(
        model, optimizer, scaler, device, run_ckpt_dir, remote,
    )

    # Training loop
    for epoch in range(start_epoch + 1, cfg.training.num_epochs + 1):
        train_stats = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            loss_fn=loss_fn, device=device, scaler=scaler,
            max_grad_norm=cfg.training.max_grad_norm,
            accum_steps=cfg.training.accum_steps,
            use_amp=cfg.training.use_amp,
        )
        val_stats = validate_one_epoch(
            model=model, loader=val_loader, loss_fn=loss_fn,
            device=device, use_amp=cfg.training.use_amp,
        )

        train_loss = train_stats["loss"]
        val_loss = val_stats["loss"]

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "grad_norm": train_stats["grad_norm"],
            "train_time": train_stats["time_sec"],
            "val_time": val_stats["time_sec"],
        })

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss

        save_checkpoint_safe(
            model, optimizer, scaler,
            epoch=epoch, train_loss=train_loss, val_loss=val_loss,
            best_val_loss=best_val_loss, history=history,
            save_path=run_ckpt_dir / LAST_CKPT_NAME, remote=remote,
        )
        if improved:
            save_checkpoint_safe(
                model, optimizer, scaler,
                epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                best_val_loss=best_val_loss, history=history,
                save_path=run_ckpt_dir / BEST_CKPT_NAME, remote=remote,
            )

        logger.info(
            "epoch %d/%d | train=%.4f val=%.4f | gnorm=%.2f | "
            "train_t=%.1fs val_t=%.1fs | best=%.4f%s",
            epoch, cfg.training.num_epochs, train_loss, val_loss,
            train_stats["grad_norm"], train_stats["time_sec"],
            val_stats["time_sec"], best_val_loss,
            "  improved" if improved else "",
        )
