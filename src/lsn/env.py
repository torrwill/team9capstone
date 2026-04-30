"""Environment helpers: deterministic seeding, device selection, cudnn config.

Refactored from notebook cell 2. The codebase deliberately has no
environment-detection branches (no `if 'google.colab' in sys.modules`) - all
env-specific glue is the user's responsibility (README has setup blocks).
"""
from __future__ import annotations

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python random, numpy, and torch (CPU + CUDA all visible devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(override: str | None) -> torch.device:
    """Return torch.device. Honors --device CLI override; otherwise auto-detect."""
    if override is not None:
        return torch.device(override)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            logger.info("cuda:%d = %s, %.1f GB", i, p.name, p.total_memory / 1e9)
        return device
    return torch.device("cpu")


def configure_cudnn(benchmark: bool) -> None:
    """Set cudnn.benchmark.

    True for training (matches notebook): faster, non-deterministic conv algos.
    False for inference: deterministic eval.
    """
    torch.backends.cudnn.benchmark = benchmark
