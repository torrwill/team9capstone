"""Run inference on a test split. Writes <experiment>_<dataset>_eval.json.

Usage:

    python scripts/infer.py --config configs/identity.yaml \
        --weights <path-to-best_model.pt> \
        --dataset {grid|lrs2} --data-dir <path> \
        [--output-dir results/predictions] \
        [--decoder {beam|greedy}] [--device cuda]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lsn.config import load_config
from lsn.data.datasets import (
    GridLipReadingDataset, LRS2Dataset, grid_collate_fn,
)
from lsn.data.splits import create_paper_split
from lsn.env import configure_cudnn, get_device, set_seed
from lsn.evaluation.inference import run_inference
from lsn.evaluation.report import write_eval_json
from lsn.models import build_from_config


def _build_test_loader(cfg, dataset: str, data_dir: Path):
    if dataset == "grid":
        # Rebuild the same paper-subset test split this model was held out from
        npz_paths = sorted(data_dir.glob("*/*.npz"))
        _, test_paths = create_paper_split(
            npz_paths, speakers=cfg.data.speakers,
            samples_per_speaker=cfg.data.samples_per_speaker,
            train_size=cfg.data.train_size, seed=cfg.data.seed,
        )
        ds = GridLipReadingDataset(test_paths)
    elif dataset == "lrs2":
        # All clips in the directory — no split logic for LRS2
        ds = LRS2Dataset(sorted(data_dir.glob("*.npz")))
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    return DataLoader(
        ds, batch_size=cfg.training.batch_size, shuffle=False,
        collate_fn=grid_collate_fn, num_workers=cfg.training.num_workers,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--dataset", required=True, choices=["grid", "lrs2"])
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("results/predictions"))
    p.add_argument("--decoder", default="beam", choices=["beam", "greedy"])
    p.add_argument("--device", default=None, choices=["cuda", "cpu", None])
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    set_seed(cfg.data.seed)
    configure_cudnn(benchmark=False)
    device = get_device(args.device)

    model = build_from_config(cfg.model, device=device)
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Carry history forward for report.py
    history = ckpt.get("history", [])
    final_epoch = ckpt.get("epoch", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))

    loader = _build_test_loader(cfg, args.dataset, args.data_dir)
    preds = run_inference(model, loader, device=device, decoder=args.decoder)

    out_path = args.output_dir / f"{cfg.experiment_name}_{args.dataset}_eval.json"
    write_eval_json(
        out_path,
        experiment_name=cfg.experiment_name,
        display_name=cfg.model.display_name or cfg.experiment_name,
        color=cfg.model.color or "#3498DB",
        dataset=args.dataset, decoder=args.decoder,
        final_epoch=final_epoch, best_val_loss=best_val_loss,
        history=history, predictions=preds,
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
