import torch
import torch.nn as nn

from lsn.evaluation.inference import Prediction, run_inference


def test_run_inference_smoke():
    """End-to-end shape check on dummy model + loader."""
    class Tiny(nn.Module):
        def forward(self, x):
            B, T, _, _ = x.shape
            return torch.log_softmax(torch.randn(T, B, 41), dim=-1)

    def fake_loader():
        yield {
            "frames": torch.randn(2, 75, 46, 140),
            "input_lengths": torch.tensor([75, 75], dtype=torch.long),
            "texts": ["hello", "world"],
            "paths": ["/x/a.npz", "/x/b.npz"],
        }

    preds = run_inference(
        Tiny(), fake_loader(), device=torch.device("cpu"), decoder="greedy",
    )
    assert isinstance(preds, list)
    assert len(preds) == 2
    assert all(isinstance(p, Prediction) for p in preds)
    assert preds[0].path == "/x/a.npz"
    assert preds[0].reference == "hello"
