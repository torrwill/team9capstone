"""Refactored from notebook cells 3-5. Class names + attribute names preserved
per state_dict invariant (spec section 10). EfficientNet stages 0-6 frozen."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class Conv3DBlock(nn.Module):
    """Single conv3d + BN + ReLU + MaxPool block."""

    def __init__(self,
                 in_channels:  int,
                 out_channels: int,
                 kernel_size:  tuple,
                 padding:      tuple,
                 pool_kernel:  tuple = (1, 2, 2),
                 pool_stride:  tuple = (1, 2, 2)):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=False)
        self.bn   = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class Frontend3DCNN(nn.Module):
    """
    Four-block 3D-CNN front-end (Table 1).

    Input : (B, 1, T, 46, 140)
    Output: (B, T, 8192) -- 512 * 2 * 8 per timestep

    Block  filters  kernel    padding   pool(1,2,2)   spatial out
    ──────────────────────────────────────────────────────────────
      1      64    (3,5,5)   (1,2,2)    (1,2,2)       23 * 70
      2     128    (3,5,5)   (1,2,2)    (1,2,2)       11 * 35
      3     256    (3,3,3)   (1,1,1)    (1,2,2)        5 * 17
      4     512    (3,3,3)   (1,1,1)    (1,2,2)        2 *  8     -> flat 8,192
    """

    OUT_C    = 512
    OUT_H    = 2
    OUT_W    = 8
    FLAT_DIM = OUT_C * OUT_H * OUT_W   # 8,192

    def __init__(self):
        super().__init__()
        self.block1 = Conv3DBlock(1,   64,  (3, 5, 5), (1, 2, 2))
        self.block2 = Conv3DBlock(64,  128, (3, 5, 5), (1, 2, 2))
        self.block3 = Conv3DBlock(128, 256, (3, 3, 3), (1, 1, 1))
        self.block4 = Conv3DBlock(256, 512, (3, 3, 3), (1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)                          # (B,  64, T, 23, 70)
        x = self.block2(x)                          # (B, 128, T, 11, 35)
        x = self.block3(x)                          # (B, 256, T,  5, 17)
        x = self.block4(x)                          # (B, 512, T,  2,  8)
        B, C, T, Hf, Wf = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, Hf, Wf)
        return x.view(B, T, C * Hf * Wf)           # (B, T, 8192)


class EfficientNet(nn.Module):
    """
    EfficientNet-B0 feature extractor, applied frame-by-frame.

    Input : (B*T, 1, H, W) -- raw grayscale lip frames (NOT 3D-CNN output)
    Output: (B*T, 62720)   -- flattened spatial map (1280 * 7 * 7)

    Freezing policy [D1]:
        Freeze features[0..6] -> 2,878,156 frozen params.
        Unfreeze features[7,8] -> trained end-to-end.

        The paper states 2,780,531 frozen, which no torchvision stage boundary produces.
        The 97,625-param discrepancy is attributed to TF/Keras vs. torchvision weight layout differences.
    """

    FEATURE_DIM = 7 * 7 * 1280   # 62,720

    def __init__(self, freeze_early: bool = True):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = base.features   # indices 0-8; output (B, 1280, 7, 7)

        if freeze_early:
            # Freeze all parameters first
            for param in self.features.parameters():
                param.requires_grad = False
            # Unfreeze stages 7 and 8 (last two MBConv blocks + head)
            for stage_idx in (7, 8):
                for param in self.features[stage_idx].parameters():
                    param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.expand(-1, 3, -1, -1)                            # (B*T, 3, H, W)
        x = F.interpolate(x, size=(224, 224),
                          mode="bilinear", align_corners=False)  # (B*T, 3, 224, 224)
        x = self.features(x)                                    # (B*T, 1280, 7, 7)
        return torch.flatten(x, 1)                              # (B*T, 62720)
