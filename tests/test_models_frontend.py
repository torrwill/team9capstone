import torch

from lsn.models.frontend import Conv3DBlock, Frontend3DCNN, EfficientNet


def test_conv3d_block_forward_shape():
    block = Conv3DBlock(in_channels=1, out_channels=64,
                        kernel_size=(3, 5, 5), padding=(1, 2, 2))
    x = torch.randn(1, 1, 75, 46, 140)
    y = block(x)
    # MaxPool3d (1,2,2) halves spatial dims
    assert y.shape == (1, 64, 75, 23, 70)


def test_frontend3dcnn_forward_shape():
    fe = Frontend3DCNN()
    x = torch.randn(2, 1, 75, 46, 140)
    y = fe(x)
    # 4 blocks -> (B, T, 8192) per Frontend3DCNN docstring
    assert y.shape == (2, 75, 8192)
    assert Frontend3DCNN.FLAT_DIM == 8192


def test_efficientnet_forward_shape():
    eff = EfficientNet(freeze_early=True)
    x = torch.randn(2, 1, 46, 140)   # (B*T, 1, H, W)
    y = eff(x)
    assert y.shape == (2, 62720)
    assert EfficientNet.FEATURE_DIM == 62720


def test_efficientnet_freeze_policy():
    """Stages 0-6 frozen, 7-8 trainable (spec section 10, notebook cell 5)."""
    eff = EfficientNet(freeze_early=True)
    for stage_idx in range(7):
        for p in eff.features[stage_idx].parameters():
            assert p.requires_grad is False
    for stage_idx in (7, 8):
        for p in eff.features[stage_idx].parameters():
            assert p.requires_grad is True
