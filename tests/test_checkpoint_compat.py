"""Backward-compat canary — load each existing best_model.pt with strict=True.

This test enforces the spec section 10 state_dict invariant. Any rename of a
model attribute (e.g. cnn3d -> frontend) breaks this test.

Set LSN_CKPT_DIR to a local directory containing:
    {LSN_CKPT_DIR}/paper_best_model.pt
    {LSN_CKPT_DIR}/identity_best_model.pt
    {LSN_CKPT_DIR}/transformer_best_model.pt

If LSN_CKPT_DIR is unset, this test is skipped (fine for CI without secrets).
"""
import os
from pathlib import Path

import pytest
import torch

from lsn.config import ModelCfg
from lsn.models import build_from_config


CKPT_DIR = os.environ.get("LSN_CKPT_DIR")
pytestmark = pytest.mark.skipif(
    CKPT_DIR is None,
    reason="LSN_CKPT_DIR not set — see test docstring",
)


CONFIGS = {
    "paper": ModelCfg(
        backend="paper", vocab_size=40,
        freeze_early_effnet=True, use_self_attn=False,
    ),
    "identity": ModelCfg(
        backend="identity", vocab_size=40, freeze_early_effnet=True,
    ),
    "transformer": ModelCfg(
        backend="transformer_perstream", vocab_size=40,
        freeze_early_effnet=True,
        backend_kwargs={"d_model": 1024, "nhead": 4, "num_layers": 2},
    ),
}


@pytest.mark.parametrize("name,cfg", list(CONFIGS.items()))
def test_existing_checkpoint_loads_strict(name: str, cfg: ModelCfg):
    ckpt_path = Path(CKPT_DIR) / f"{name}_best_model.pt"
    if not ckpt_path.exists():
        pytest.skip(f"checkpoint not found at {ckpt_path}")

    model = build_from_config(cfg, device=torch.device("cpu"))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    # Assert strict load — any drift raises RuntimeError.
    model.load_state_dict(state_dict, strict=True)
