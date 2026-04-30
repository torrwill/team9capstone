"""Public API for the lsn.models package.

Mirrors notebook cell 12. Adds `build_from_config` for YAML-driven dispatch.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lsn.config import ModelCfg
from lsn.models.lipsyncnet import LipSyncNetPaper, LipSyncNetVariant


def build_paper_model(vocab_size: int = 40,
                      use_self_attn: bool = False,
                      device: str = "cpu") -> LipSyncNetPaper:
    """Instantiate the paper-faithful model."""
    return LipSyncNetPaper(
        vocab_size=vocab_size, use_self_attn=use_self_attn,
    ).to(device)


def build_variant(backend: str = "bilstm",
                  vocab_size: int = 40,
                  device: str = "cpu",
                  **backend_kwargs) -> LipSyncNetVariant:
    """Instantiate the modular variant with chosen backend."""
    return LipSyncNetVariant(
        backend=backend, vocab_size=vocab_size, **backend_kwargs,
    ).to(device)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total / trainable / frozen parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def build_from_config(cfg: ModelCfg, device: torch.device) -> nn.Module:
    """Dispatch from a ModelCfg to the right builder.

    `backend == "paper"` routes to LipSyncNetPaper; everything else routes
    to LipSyncNetVariant via the `_BACKEND_REGISTRY`.
    """
    if cfg.backend == "paper":
        return build_paper_model(
            vocab_size=cfg.vocab_size,
            use_self_attn=cfg.use_self_attn,
            device=str(device),
        )
    # Validate against the registry without leaking it
    from lsn.models.backends import _BACKEND_REGISTRY
    if cfg.backend not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend {cfg.backend!r}. "
            f"Valid options: paper, {', '.join(sorted(_BACKEND_REGISTRY))}"
        )
    return build_variant(
        backend=cfg.backend, vocab_size=cfg.vocab_size, device=str(device),
        **cfg.backend_kwargs,
    )


__all__ = [
    "build_paper_model", "build_variant", "count_parameters",
    "build_from_config",
]
