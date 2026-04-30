"""YAML config schema (typed dataclasses) + loader + CLI override merge.

See spec §4.1 for the full schema. CLI flags override only env-specific
fields; hyperparameters live in YAML (spec §6).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelCfg:
    backend: str
    vocab_size: int = 40
    freeze_early_effnet: bool = True
    use_self_attn: bool = False
    backend_kwargs: dict[str, Any] = field(default_factory=dict)
    display_name: str | None = None
    color: str | None = None


@dataclass
class TrainingCfg:
    num_epochs: int
    learning_rate: float
    batch_size: int
    accum_steps: int
    max_grad_norm: float
    use_amp: bool
    num_workers: int
    prefetch: int


@dataclass
class DataCfg:
    dataset: str
    speakers: list[str]
    samples_per_speaker: int
    train_size: int
    seed: int


@dataclass
class CkptCfg:
    hf_repo: str | None = None
    hf_subfolder: str | None = None


@dataclass
class Config:
    experiment_name: str
    model: ModelCfg
    training: TrainingCfg
    data: DataCfg
    checkpointing: CkptCfg


def load_config(path: str | Path) -> Config:
    """Load YAML at `path` into a typed Config."""
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return Config(
        experiment_name=raw["experiment_name"],
        model=ModelCfg(**raw["model"]),
        training=TrainingCfg(**raw["training"]),
        data=DataCfg(**raw["data"]),
        checkpointing=CkptCfg(**raw["checkpointing"]),
    )


def apply_cli_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Mutate config in-place from CLI args. Only fields explicitly listed in
    spec §6 (overridable) are touched; everything else stays YAML-driven.

    Recognized args: hf_repo, epochs.
    """
    hf_repo = getattr(args, "hf_repo", None)
    if hf_repo is not None:
        cfg.checkpointing.hf_repo = hf_repo

    epochs = getattr(args, "epochs", None)
    if epochs is not None:
        cfg.training.num_epochs = epochs

    return cfg
