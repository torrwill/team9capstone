import torch

from lsn.models.backends import (
    _BACKEND_REGISTRY, BiLSTMBackend, IdentityBackend,
    TransformerBackend, TransformerBackendPerStream,
    _SinusoidalPE,
)


def test_registry_keys():
    """Registry keys are public dispatch surface — DO NOT rename."""
    assert set(_BACKEND_REGISTRY.keys()) == {
        "bilstm", "identity", "transformer", "transformer_perstream",
    }


def test_bilstm_backend_shapes():
    b = BiLSTMBackend(input_dim=70912, hidden=512, dropout=0.5)
    x = torch.randn(2, 75, 70912)
    y = b(x)
    assert y.shape == (2, 75, 1024)
    assert b.out_dim == 1024


def test_identity_backend_passthrough():
    b = IdentityBackend(input_dim=70912)
    x = torch.randn(2, 75, 70912)
    y = b(x)
    assert torch.equal(y, x)
    assert b.out_dim == 70912


def test_transformer_backend_shapes():
    b = TransformerBackend(input_dim=70912)
    x = torch.randn(2, 75, 70912)
    y = b(x)
    assert y.shape[0] == 2
    assert y.shape[1] == 75
    # default d_model=1024
    assert y.shape[2] == 1024


def test_transformer_perstream_backend_shapes():
    b = TransformerBackendPerStream(d_model=1024, nhead=4, num_layers=2)
    cnn_dim = 8192
    eff_dim = 62720
    fused = torch.randn(2, 75, cnn_dim + eff_dim)
    y = b(fused)
    assert y.shape == (2, 75, 1024)


def test_sinusoidal_pe_buffer_shape():
    pe = _SinusoidalPE(d_model=1024, max_len=75)
    assert pe.pe.shape == (1, 75, 1024)
