from pathlib import Path

import torch
import torch.nn as nn

from lsn.training.checkpoint import (
    LAST_CKPT_NAME, BEST_CKPT_NAME,
    save_checkpoint_safe, try_resume, freeze_bn_stats,
)


def _make_scaler():
    """Cross-version disabled GradScaler (torch 2.3+ uses torch.amp.GradScaler,
    older versions only have torch.cuda.amp.GradScaler). Tests run on CPU; the
    scaler is disabled either way."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cpu", enabled=False)
    return torch.cuda.amp.GradScaler(enabled=False)


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(8)
        self.lin = nn.Linear(8, 8)

    def forward(self, x):
        return self.lin(self.bn(x))


def test_save_resume_roundtrip(tmp_path):
    model = _Tiny()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = _make_scaler()

    save_path = tmp_path / LAST_CKPT_NAME
    save_checkpoint_safe(
        model, opt, scaler,
        epoch=3, train_loss=2.0, val_loss=2.5, best_val_loss=2.5,
        history=[{"epoch": 1, "train_loss": 3.0, "val_loss": 3.1}],
        save_path=save_path, remote=None,
    )
    assert save_path.exists()

    model2 = _Tiny()
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-4)
    scaler2 = _make_scaler()
    epoch, best, history = try_resume(
        model2, opt2, scaler2, device=torch.device("cpu"),
        local_dir=tmp_path, remote=None,
    )
    assert epoch == 3
    assert best == 2.5
    assert len(history) == 1


def test_try_resume_fresh_start(tmp_path):
    model = _Tiny()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = _make_scaler()
    epoch, best, history = try_resume(
        model, opt, scaler, device=torch.device("cpu"),
        local_dir=tmp_path, remote=None,
    )
    assert epoch == 0
    assert best == float("inf")
    assert history == []


def test_freeze_bn_stats_only_freezes_frozen_param_bn():
    """Spec §4.5: BN goes to eval IFF its direct params are all frozen."""
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn_trainable = nn.BatchNorm1d(8)         # params trainable → stays in train
            self.bn_frozen = nn.BatchNorm1d(8)            # params frozen → goes to eval
            for p in self.bn_frozen.parameters():
                p.requires_grad = False

    m = M()
    m.train()
    n = freeze_bn_stats(m)
    assert n == 1
    assert m.bn_trainable.training is True
    assert m.bn_frozen.training is False
