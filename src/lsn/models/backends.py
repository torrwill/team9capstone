"""Temporal backbones + registry — refactored from notebook cells 6-8.

Class names, attribute names, and registry keys preserved per state_dict
invariant (spec §10). DO NOT rename anything — trained checkpoints depend
on the exact names below.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from lsn.models.frontend import EfficientNet, Frontend3DCNN


class BiLSTMBackend(nn.Module):
    """
    2× Bi-LSTM with input projection and dropout.
    Used inside LipSyncNetVariant — NOT inside LipSyncNetPaper.

    input_dim → Linear(1024) → BiLSTM(512×2) → Dropout → BiLSTM(512×2) → Dropout
    Output dim: 1,024.
    """

    def __init__(self, input_dim: int, hidden: int = 512, dropout: float = 0.5):
        super().__init__()
        self.out_dim    = hidden * 2
        self.input_proj = nn.Linear(input_dim, self.out_dim)
        self.lstm1      = nn.LSTM(self.out_dim, hidden, batch_first=True,
                                  bidirectional=True)
        self.drop1      = nn.Dropout(dropout)
        self.lstm2      = nn.LSTM(self.out_dim, hidden, batch_first=True,
                                  bidirectional=True)
        self.drop2      = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm1(self.input_proj(x))
        x     = self.drop1(x)
        x, _ = self.lstm2(x)
        return self.drop2(x)


class IdentityBackend(nn.Module):
    """
    No temporal modeling — ablation baseline.
    Passes fused features directly to the classifier unchanged.
    Isolates the contribution of the temporal backend.
    """

    def __init__(self, input_dim: int, **_):
        super().__init__()
        self.out_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding ("Attention Is All You Need" paper; Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 75, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerBackend(nn.Module):
    """
    2-layer bidirectional Transformer encoder.
    Linear projection → sinusoidal PE → 2× TransformerEncoderLayer.
    Output dim: d_model (default 1,024).

    NOTE: max_len in _SinusoidalPE must be ≥ the longest sequence T in the dataset;
    Update if LRS2 clips exceed 75 frames (see TODO-PRE-5).
    """

    def __init__(self, input_dim: int, d_model: int = 1024,
                 nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.out_dim    = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = _SinusoidalPE(d_model, dropout=dropout)
        self.encoder    = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                       dim_feedforward=d_model * 4,
                                       dropout=dropout, batch_first=True),
            num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.pos_enc(self.input_proj(x)))


# ============================================================
# TransformerBackendPerStream — matches friend's transformer variant
# ============================================================
# Per-stream projection (CNN and EfficientNet streams each get their
# own Linear+LayerNorm to d_model/2), then concat, then standard
# transformer encoder. Different parameter budget and inductive bias
# than our single-projection TransformerBackend, but architecturally
# clean.

class TransformerBackendPerStream(nn.Module):
    """
    Forward path:
        cnn (B, T, 8192)   → Linear(8192, proj_dim) → LayerNorm(proj_dim)
        eff (B, T, 62720)  → Linear(62720, proj_dim) → LayerNorm(proj_dim)
        concat along last dim                                  → (B, T, d_model)
        sinusoidal PE (additive)                               → (B, T, d_model)
        TransformerEncoder (num_layers layers)                 → (B, T, d_model)

    Requires d_model == 2 * proj_dim (the two streams concat).

    Matches state-dict keys:
        backend.cnn_proj.{0.weight, 0.bias, 1.weight, 1.bias}
        backend.eff_proj.{0.weight, 0.bias, 1.weight, 1.bias}
        backend.encoder.layers.{L}.*
        backend.pos_enc.pe
    """

    # Input dims coming from the frontend (do NOT make these configurable —
    # they're determined by the upstream 3D-CNN and EfficientNet-B0 outputs).
    CNN_DIM = Frontend3DCNN.FLAT_DIM      # 8,192
    EFF_DIM = EfficientNet.FEATURE_DIM    # 62,720

    def __init__(self,
                 input_dim: int = None,       # ignored; kept for registry compat
                 d_model:   int = 1024,
                 nhead:     int = 4,          # TODO: confirm with friend
                 num_layers: int = 2,
                 dropout:    float = 0.1):
        super().__init__()

        assert d_model % 2 == 0, f"d_model ({d_model}) must be even (two streams concat)"
        proj_dim = d_model // 2                # 512 for d_model=1024

        self.out_dim = d_model

        # Per-stream projection + LayerNorm
        self.cnn_proj = nn.Sequential(
            nn.Linear(self.CNN_DIM, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.eff_proj = nn.Sequential(
            nn.Linear(self.EFF_DIM, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # Reuse the existing _SinusoidalPE from the notebook.
        self.pos_enc = _SinusoidalPE(d_model, dropout=dropout)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True,
            ),
            num_layers=num_layers,
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """
        IMPORTANT: unlike the other backends in the registry, this one does
        NOT take the already-fused (B, T, 70912) tensor as input. It needs
        the two streams separately. LipSyncNetVariant currently fuses before
        calling the backend — we handle this by splitting fused back into
        its two components. Feasible because fused is just concat([cnn, eff])
        along the last dim in a known order.
        """
        cnn = fused[..., :self.CNN_DIM]                                # (B, T, 8192)
        eff = fused[..., self.CNN_DIM:self.CNN_DIM + self.EFF_DIM]     # (B, T, 62720)

        cnn_out = self.cnn_proj(cnn)      # (B, T, d_model/2)
        eff_out = self.eff_proj(eff)      # (B, T, d_model/2)

        x = torch.cat([cnn_out, eff_out], dim=-1)    # (B, T, d_model)
        x = self.pos_enc(x)
        return self.encoder(x)


# Unified registry — combines cell 6's initial dict + cell 8's addition.
_BACKEND_REGISTRY: dict[str, type] = {
    "bilstm":               BiLSTMBackend,
    "identity":             IdentityBackend,
    "transformer":          TransformerBackend,
    "transformer_perstream": TransformerBackendPerStream,
}
