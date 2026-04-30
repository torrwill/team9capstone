"""Training and validation epoch functions.

Refactored from notebook cell 23. Preserves the AMP-fp16-cast-to-fp32-for-CTC
trick (CTC NaNs in fp16; forward under autocast but log_probs.float() before
CTCLoss). Gradient accumulation supports effective batch sizes larger than
fits in GPU memory.
"""
from __future__ import annotations

import logging
import time

import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm

from lsn.training.checkpoint import freeze_bn_stats

logger = logging.getLogger(__name__)


def train_one_epoch(
    model, loader, optimizer, loss_fn, device,
    scaler, max_grad_norm: float = 5.0, accum_steps: int = 1,
    use_amp: bool = True,
) -> dict:
    """Run one training epoch with AMP + gradient accumulation.

    CTC in fp16 is numerically unstable (NaN very quickly). Strategy: run the
    forward pass under autocast, but cast log_probs back to fp32 before
    feeding CTCLoss. PyTorch docs recommend this explicitly.

    For gradient accumulation: divide the loss by accum_steps so the gradient
    magnitude at optimizer.step() matches what you would get from a batch of
    (batch_size * accum_steps). Steps on accumulation boundary OR on the
    final batch (so a partial accum tail doesn't get dropped).

    Returns dict with keys: loss, grad_norm, time_sec.
    """
    model.train()
    freeze_bn_stats(model)  # keep frozen BN running stats fixed

    running_loss = 0.0
    running_gnorm = 0.0
    num_batches = 0
    start_time = time.time()

    optimizer.zero_grad(set_to_none=True)
    n_loader = len(loader)

    for step, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        frames         = batch["frames"].to(device, non_blocking=True)
        targets        = batch["targets"].to(device, non_blocking=True)
        input_lengths  = batch["input_lengths"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)

        with autocast("cuda", dtype=torch.float16, enabled=use_amp):
            log_probs = model(frames)

        # Cast to fp32 for CTC — it NaNs in fp16.
        loss = loss_fn(log_probs.float(), targets, input_lengths, target_lengths)
        loss_for_backward = loss / accum_steps

        if use_amp:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        # Step on accumulation boundary OR on the final batch (flush partial accum).
        is_step_boundary = ((step + 1) % accum_steps == 0) or (step + 1 == n_loader)
        if is_step_boundary:
            if use_amp:
                scaler.unscale_(optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                max_grad_norm,
            ).item()
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            running_gnorm += gnorm

        running_loss += loss.item()
        num_batches  += 1

    n_steps = max(1, num_batches // accum_steps)
    return {
        "loss":      running_loss / max(num_batches, 1),
        "grad_norm": running_gnorm / n_steps,
        "time_sec":  time.time() - start_time,
    }


def validate_one_epoch(
    model, loader, loss_fn, device, use_amp: bool = True,
) -> dict:
    """Run one validation epoch under torch.no_grad with AMP forward + fp32 CTC.

    Returns dict with keys: loss, time_sec.
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0
    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            frames         = batch["frames"].to(device, non_blocking=True)
            targets        = batch["targets"].to(device, non_blocking=True)
            input_lengths  = batch["input_lengths"].to(device, non_blocking=True)
            target_lengths = batch["target_lengths"].to(device, non_blocking=True)

            with autocast("cuda", dtype=torch.float16, enabled=use_amp):
                log_probs = model(frames)
            loss = loss_fn(log_probs.float(), targets, input_lengths, target_lengths)

            running_loss += loss.item()
            num_batches  += 1

    return {
        "loss":     running_loss / max(num_batches, 1),
        "time_sec": time.time() - start_time,
    }
