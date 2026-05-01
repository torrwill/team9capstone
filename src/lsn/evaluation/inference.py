"""Run a model over a DataLoader, decode each batch, return list[Prediction].

Refactored from notebook Stage D (cell 37). The output is the input to
write_eval_json (spec §7).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from tqdm import tqdm

from lsn.evaluation.decoders import beam_decode, greedy_decode


@dataclass
class Prediction:
    path: str
    reference: str
    hypothesis: str


def run_inference(model, loader, device: torch.device,
                  decoder: str = "beam") -> list[Prediction]:
    """Iterate the loader under torch.no_grad(), decode each batch."""
    model.eval()
    out: list[Prediction] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="inference", leave=False):
            frames = batch["frames"].to(device, non_blocking=True)
            log_probs = model(frames)        # (T, B, C)
            input_lengths = batch["input_lengths"].to(log_probs.device)

            if decoder == "greedy":
                hyps = greedy_decode(log_probs)
            elif decoder == "beam":
                hyps = beam_decode(log_probs, input_lengths, beam_width=100)
            else:
                raise ValueError(f"unknown decoder {decoder!r}")

            for path, ref, hyp in zip(batch["paths"], batch["texts"], hyps):
                out.append(Prediction(path=path, reference=ref, hypothesis=hyp))
    return out
