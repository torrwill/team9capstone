import argparse
from pathlib import Path

import pytest

from lsn.config import (
    Config, ModelCfg, TrainingCfg, DataCfg, CkptCfg,
    load_config, apply_cli_overrides,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_config.yaml"


def test_load_config_returns_typed_dataclass():
    cfg = load_config(FIXTURE)
    assert isinstance(cfg, Config)
    assert cfg.experiment_name == "run_test_v1"
    assert isinstance(cfg.model, ModelCfg)
    assert isinstance(cfg.training, TrainingCfg)
    assert isinstance(cfg.data, DataCfg)
    assert isinstance(cfg.checkpointing, CkptCfg)


def test_load_config_parses_model_fields():
    cfg = load_config(FIXTURE)
    assert cfg.model.backend == "identity"
    assert cfg.model.vocab_size == 40
    assert cfg.model.use_self_attn is False
    assert cfg.model.color == "#000000"


def test_load_config_parses_training_fields():
    cfg = load_config(FIXTURE)
    assert cfg.training.num_epochs == 5
    assert cfg.training.learning_rate == 1e-4
    assert cfg.training.batch_size == 2
    assert cfg.training.use_amp is True


def test_load_config_parses_data_fields():
    cfg = load_config(FIXTURE)
    assert cfg.data.speakers == ["s1", "s2", "s3", "s4", "s5"]
    assert cfg.data.train_size == 450
    assert cfg.data.seed == 42


def test_load_config_handles_null_hf():
    cfg = load_config(FIXTURE)
    assert cfg.checkpointing.hf_repo is None
    assert cfg.checkpointing.hf_subfolder is None


def test_apply_cli_overrides_only_touches_allowed_fields():
    cfg = load_config(FIXTURE)
    args = argparse.Namespace(
        hf_repo="ranro1/test", epochs=99,
    )
    cfg2 = apply_cli_overrides(cfg, args)
    assert cfg2.checkpointing.hf_repo == "ranro1/test"
    assert cfg2.training.num_epochs == 99
    # Other fields untouched
    assert cfg2.training.learning_rate == 1e-4


def test_apply_cli_overrides_ignores_none_values():
    cfg = load_config(FIXTURE)
    args = argparse.Namespace(hf_repo=None, epochs=None)
    cfg2 = apply_cli_overrides(cfg, args)
    assert cfg2.checkpointing.hf_repo is None
    assert cfg2.training.num_epochs == 5  # unchanged
