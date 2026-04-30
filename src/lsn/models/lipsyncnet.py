"""Top-level LipSyncNet models. Refactored from notebook cells 9-11.
Class names + attribute names preserved per state_dict invariant (spec §10)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lsn.models.frontend import EfficientNet, Frontend3DCNN
from lsn.models.backends import _BACKEND_REGISTRY


class SelfAttentionBlock(nn.Module):
    """
    Post-norm multi-head self-attention block placed between Bi-LSTM-2 and
    the classifier, as shown in Figure 9 of the paper.

    Design rationale (all choices forced by context; none are free):

    embed_dim = 1024
        Forced by the Bi-LSTM-2 output shape (B, T, 1024) and the classifier
        input shape (B, T, 1024).  The block must be dimensionality-preserving.

    num_heads = 8
        8 heads * 128 head_dim = 1024.

    Residual connection  (out = LayerNorm(x + Dropout(MHA(x))))
        Standard post-norm formulation.  A bare MHA call with no residual
        would discard positional information accumulated by the LSTMs and
        is not used anywhere in the sequence modelling literature at this
        position.  The paper does not specify, so the conventional choice
        is made and documented.

    dropout = 0.1
        The paper states no dropout for this layer.  0.1 is the standard
        attention dropout used in the original Transformer (Vaswani et al.).
        Keeping it low (vs. the 0.5 used in the LSTM stack) because attention
        weights are already an implicit regulariser. It is configurable.

    No feedforward sublayer
        The paper's block diagram shows a single "Self-Attention" box,
        use TransformerBackend instead during ablation.
    """

    def __init__(self, embed_dim: int = 1024, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(embed_dim=embed_dim,
                                           num_heads=num_heads,
                                           dropout=dropout,
                                           batch_first=True)
        self.norm  = nn.LayerNorm(embed_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # query = key = value = x  (self-attention over the temporal dimension)
        attn_out, _ = self.attn(x, x, x,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)
        # Post-norm residual:  LayerNorm(x + Dropout(Attn(x)))
        return self.norm(x + self.drop(attn_out))              # (B, T, 1024)


class LipSyncNetPaper(nn.Module):
    """
    Re-implementation of LipSyncNet (Table 1 / Figure 8 & 9).

    Explicit decisions for all ambiguities:
      [1] EfficientNet: stages 0-6 frozen (2,878,156 params; closest to paper).
      [2] Self-attention included (Figure 9), implemented as SelfAttentionBlock
           (embed_dim=1024, num_heads=8).  Absent from Table 1.
           Controlled by `use_self_attn` flag so the no-attention variant (matching Table 1) can still be instantiated for ablation.
      [3] No input projection before LSTM-1; raw 70,912-dim concat fed directly.
    """

    CNN_DIM   = Frontend3DCNN.FLAT_DIM              # 8,192
    EFF_DIM   = EfficientNet.FEATURE_DIM      # 62,720
    FUSED_DIM = CNN_DIM + EFF_DIM                   # 70,912

    def __init__(self,
                 vocab_size:          int  = 40,
                 freeze_early_effnet: bool = True,
                 use_self_attn:       bool = True):
        """
        Args:
            vocab_size          : output classes excluding CTC blank.
            freeze_early_effnet : freeze EfficientNet stages 0-6.
            use_self_attn       : include SelfAttentionBlock after Bi-LSTM-2.
        """
        super().__init__()
        self.vocab_size    = vocab_size
        self.use_self_attn = use_self_attn

        self.cnn3d        = Frontend3DCNN()
        self.efficientnet = EfficientNet(freeze_early=freeze_early_effnet)

        self.lstm1 = nn.LSTM(self.FUSED_DIM, 512, batch_first=True,
                             bidirectional=True)
        self.drop1 = nn.Dropout(0.5)
        self.lstm2 = nn.LSTM(1024, 512, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(0.5)

        if use_self_attn:
            self.self_attn = SelfAttentionBlock(embed_dim=1024, num_heads=8,
                                                dropout=0.1)

        self.classifier = nn.Linear(1024, vocab_size + 1)  # +1 = CTC blank

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, H, W = x.shape

        # 3D-CNN branch (spatiotemporal)
        cnn_out = self.cnn3d(x.unsqueeze(1))           # (B, T, 8192)

        # EfficientNet branch (per-frame, on raw input)
        frames  = x.view(B * T, 1, H, W)               # (B*T, 1, 46, 140)
        eff_out = self.efficientnet(frames)             # (B*T, 62720)
        eff_out = eff_out.view(B, T, self.EFF_DIM)     # (B, T, 62720)

        fused = torch.cat([cnn_out, eff_out], dim=-1)  # (B, T, 70912)

        # Bi-LSTM back-end
        out, _ = self.lstm1(fused)                     # (B, T, 1024)
        out     = self.drop1(out)
        out, _ = self.lstm2(out)                       # (B, T, 1024)
        out     = self.drop2(out)

        # Skipped when use_self_attn=False.
        if self.use_self_attn:
            out = self.self_attn(out, key_padding_mask=key_padding_mask)
                                                       # (B, T, 1024)
        # Classifier
        logits    = self.classifier(out)               # (B, T, vocab+1)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.permute(1, 0, 2)              # (T, B, vocab+1)


class LipSyncNetVariant(nn.Module):
    """
    Backend selected at construction from {"bilstm", "transformer", "identity"}.

    Note: All backends receive the raw FUSED_DIM (70,912) vector for fair comparison.
    The BiLSTM variant adds a projection (Linear 70912 -> 1024) before the LSTMs;
    this differs slightly from LipSyncNetPaper (i.e., no projection).
    For paper: Both should be reported with the distinction made explicit.
    """

    CNN_DIM   = Frontend3DCNN.FLAT_DIM          # 8,192
    EFF_DIM   = EfficientNet.FEATURE_DIM  # 62,720
    FUSED_DIM = CNN_DIM + EFF_DIM               # 70,912

    def __init__(self,
                 backend:             str  = "bilstm",
                 vocab_size:          int  = 40,
                 freeze_early_effnet: bool = True,
                 **backend_kwargs):
        super().__init__()

        if backend not in _BACKEND_REGISTRY:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Valid options: {list(_BACKEND_REGISTRY)}")

        self.vocab_size   = vocab_size
        self.cnn3d        = Frontend3DCNN()
        self.efficientnet = EfficientNet(freeze_early=freeze_early_effnet)
        self.backend      = _BACKEND_REGISTRY[backend](
                                input_dim=self.FUSED_DIM, **backend_kwargs)
        self.classifier   = nn.Linear(self.backend.out_dim, vocab_size + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W = x.shape

        cnn_out = self.cnn3d(x.unsqueeze(1))

        frames  = x.view(B * T, 1, H, W)

        eff_out = self.efficientnet(frames).view(B, T, self.EFF_DIM)

        fused   = torch.cat([cnn_out, eff_out], dim=-1)   # (B, T, 70912)

        out       = self.backend(fused)

        logits    = self.classifier(out)
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs.permute(1, 0, 2)                 # (T, B, vocab+1)
