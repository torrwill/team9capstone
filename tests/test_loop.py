import torch
import torch.nn as nn

from lsn.training.loop import validate_one_epoch


def test_validate_one_epoch_returns_dict_with_loss():
    """Smoke test: validate runs end-to-end on dummy data and returns shape."""
    class TinyCTC(nn.Module):
        def forward(self, x):
            B, T, _, _ = x.shape
            # log_probs shape (T, B, C=41) — matches LipSyncNet output
            return torch.log_softmax(torch.randn(T, B, 41), dim=-1)

    model = TinyCTC()

    def fake_loader():
        yield {
            "frames": torch.randn(2, 75, 46, 140),
            "targets": torch.randint(1, 27, (10,), dtype=torch.long),
            "input_lengths": torch.tensor([75, 75], dtype=torch.long),
            "target_lengths": torch.tensor([5, 5], dtype=torch.long),
        }

    loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    out = validate_one_epoch(
        model, fake_loader(), loss_fn, device=torch.device("cpu"), use_amp=False,
    )
    assert "loss" in out
    assert "time_sec" in out
