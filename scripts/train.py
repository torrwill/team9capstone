"""Entry point for training. Usage:

    python scripts/train.py --config configs/identity.yaml \
        --data-dir <path> [--ckpt-dir results/checkpoints] \
        [--hf-repo ranro1/lipsyncnet-checkpoints] \
        [--device cuda] [--epochs N]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lsn.config import apply_cli_overrides, load_config
from lsn.env import get_device
from lsn.training.runner import run


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data-dir", required=True, type=Path,
                   help="Directory containing speaker-subdir layout: <dir>/s1/*.npz")
    p.add_argument("--ckpt-dir", type=Path, default=Path("results/checkpoints"),
                   help="Parent directory; per-experiment subdir built from experiment_name")
    p.add_argument("--hf-repo", default=None,
                   help="Enable HF resume/upload (overrides YAML)")
    p.add_argument("--device", default=None, choices=["cuda", "cpu", None])
    p.add_argument("--epochs", type=int, default=None,
                   help="Override training.num_epochs (smoke-test convenience)")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = apply_cli_overrides(load_config(args.config), args)
    device = get_device(args.device)
    run(cfg, data_dir=args.data_dir, ckpt_dir=args.ckpt_dir, device=device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
