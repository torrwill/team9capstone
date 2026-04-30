import torch

from lsn.config import ModelCfg
from lsn.models import (
    build_paper_model, build_variant, count_parameters, build_from_config,
)
from lsn.models.lipsyncnet import LipSyncNetPaper, LipSyncNetVariant


def test_build_paper_model_returns_paper_class():
    m = build_paper_model(vocab_size=40, use_self_attn=False, device="cpu")
    assert isinstance(m, LipSyncNetPaper)


def test_build_variant_returns_variant_class():
    for backend in ("bilstm", "identity"):
        m = build_variant(backend=backend, vocab_size=40, device="cpu")
        assert isinstance(m, LipSyncNetVariant)


def test_count_parameters_keys():
    m = build_variant(backend="identity", vocab_size=40)
    counts = count_parameters(m)
    assert set(counts.keys()) == {"total", "trainable", "frozen"}
    assert counts["total"] == counts["trainable"] + counts["frozen"]


def test_build_from_config_dispatches_paper():
    cfg = ModelCfg(backend="paper", vocab_size=40, use_self_attn=False)
    m = build_from_config(cfg, device=torch.device("cpu"))
    assert isinstance(m, LipSyncNetPaper)


def test_build_from_config_dispatches_variant():
    for backend in ("identity", "bilstm", "transformer", "transformer_perstream"):
        kwargs = {}
        if backend.startswith("transformer"):
            kwargs = {"d_model": 1024, "nhead": 4, "num_layers": 2}
        cfg = ModelCfg(backend=backend, vocab_size=40, backend_kwargs=kwargs)
        m = build_from_config(cfg, device=torch.device("cpu"))
        assert isinstance(m, LipSyncNetVariant)


def test_build_from_config_unknown_backend_raises():
    import pytest
    cfg = ModelCfg(backend="bogus", vocab_size=40)
    with pytest.raises(ValueError):
        build_from_config(cfg, device=torch.device("cpu"))
