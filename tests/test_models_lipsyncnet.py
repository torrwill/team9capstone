import torch

from lsn.models.lipsyncnet import (
    LipSyncNetPaper, LipSyncNetVariant, SelfAttentionBlock,
)


def test_self_attention_block_shape_preserving():
    sab = SelfAttentionBlock(embed_dim=1024, num_heads=8)
    x = torch.randn(2, 75, 1024)
    y = sab(x)
    assert y.shape == x.shape


def test_lipsyncnet_paper_no_self_attn_forward():
    """The paper checkpoint was trained with use_self_attn=False (spec §5).
    This is the variant that must exist for checkpoint compatibility."""
    model = LipSyncNetPaper(vocab_size=40, use_self_attn=False)
    x = torch.randn(2, 75, 46, 140)
    y = model(x)
    # Expected output: (T, B, vocab+1) = (75, 2, 41)
    assert y.shape == (75, 2, 41)


def test_lipsyncnet_paper_with_self_attn_forward():
    model = LipSyncNetPaper(vocab_size=40, use_self_attn=True)
    x = torch.randn(2, 75, 46, 140)
    y = model(x)
    assert y.shape == (75, 2, 41)
    assert hasattr(model, "self_attn")


def test_lipsyncnet_variant_identity_no_lstm():
    model = LipSyncNetVariant(backend="identity", vocab_size=40)
    n_lstm = sum(1 for m in model.modules() if isinstance(m, torch.nn.LSTM))
    assert n_lstm == 0


def test_lipsyncnet_variant_all_backends_forward_shape():
    for backend in ("bilstm", "identity", "transformer", "transformer_perstream"):
        kwargs = {}
        if backend in ("transformer", "transformer_perstream"):
            kwargs = {"d_model": 1024, "nhead": 4, "num_layers": 2}
        model = LipSyncNetVariant(backend=backend, vocab_size=40, **kwargs)
        x = torch.randn(2, 75, 46, 140)
        y = model(x)
        assert y.shape == (75, 2, 41), f"backend={backend} returned {y.shape}"


def test_lipsyncnet_state_dict_keys_top_level():
    """Spec §10 — top-level attribute names must be preserved."""
    paper = LipSyncNetPaper(vocab_size=40, use_self_attn=False)
    keys = set(paper.state_dict().keys())
    # Expect cnn3d.*, efficientnet.*, lstm1.*, lstm2.*, classifier.* present
    assert any(k.startswith("cnn3d.") for k in keys)
    assert any(k.startswith("efficientnet.") for k in keys)
    assert any(k.startswith("lstm1.") for k in keys)
    assert any(k.startswith("lstm2.") for k in keys)
    assert any(k.startswith("classifier.") for k in keys)
    # No self_attn keys when use_self_attn=False
    assert not any(k.startswith("self_attn.") for k in keys)
