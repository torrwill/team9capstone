"""JSON write + plots + tables — pure consumer of <experiment>_<dataset>_eval.json.

The single non-JSON side-channel: PAPER_BASELINES_GRID is a hardcoded
constant (spec §7). Refactored from notebook Stages A and E.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on Colab/Kaggle/headless
import matplotlib.pyplot as plt
import pandas as pd

from lsn.data.normalize import normalize_lrs2
from lsn.evaluation.inference import Prediction
from lsn.evaluation.metrics import cer, sentence_acc, wer, word_acc


# Spec §7: Table 5 paper-baseline rows. Provenance: notebook Stage E (cell 40).
PAPER_BASELINES_GRID: list[dict] = [
    {"model": "Xu et al. [29]",               "dataset": "GRID",
     "method": "Cascaded Attention-CTC",       "word_acc": 0.896, "wer": None},
    {"model": "Gergen et al. [31]",            "dataset": "GRID",
     "method": "—",                            "word_acc": 0.864, "wer": None},
    {"model": "Margam et al. [30]",            "dataset": "GRID",
     "method": "3D-2D-CNN BLSTM-HMM",         "word_acc": 0.914, "wer": None},
    {"model": "LipSyncNet (paper-reported)",   "dataset": "GRID",
     "method": "—",                            "word_acc": 0.967, "wer": 0.082},
]


def write_eval_json(out_path: Path, *,
                    experiment_name: str, display_name: str, color: str,
                    dataset: str, decoder: str,
                    final_epoch: int, best_val_loss: float,
                    history: list[dict], predictions: list[Prediction]) -> None:
    """Write the self-describing eval JSON consumed by report.* functions."""
    payload = {
        "experiment_name": experiment_name,
        "display_name": display_name,
        "color": color,
        "dataset": dataset,
        "decoder": decoder,
        "final_epoch": final_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
        "predictions": [asdict(p) for p in predictions],
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def _load_eval_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def plot_learning_curves(eval_jsons: list[Path], out_dir: Path) -> None:
    """Per-model curves + comparison plot. Refactored from notebook Stage A."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = [_load_eval_json(p) for p in eval_jsons]

    # Per-model
    for d in loaded:
        history = d["history"]
        if not history:
            continue
        epochs = [h["epoch"] for h in history]
        train = [h["train_loss"] for h in history]
        val = [h["val_loss"] for h in history]
        color = d.get("color") or "#3498DB"

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(epochs, train, label="train loss", color=color, linewidth=1.8)
        ax.plot(epochs, val, label="val loss", color=color,
                linewidth=1.8, linestyle="--", alpha=0.7)
        best_idx = val.index(min(val))
        ax.scatter([epochs[best_idx]], [val[best_idx]], color=color, s=60,
                   zorder=5, edgecolor="black", linewidth=1.0,
                   label=f"best (ep {epochs[best_idx]}, {val[best_idx]:.3f})")
        ax.set_xlabel("epoch")
        ax.set_ylabel("CTC loss")
        ax.set_title(f"{d['display_name']} — train vs val loss")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.savefig(out_dir / f"learning_curve_{d['experiment_name']}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Comparison
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for d in loaded:
        history = d["history"]
        if not history:
            continue
        epochs = [h["epoch"] for h in history]
        val = [h["val_loss"] for h in history]
        color = d.get("color") or "#3498DB"
        ax.plot(epochs, val, label=f"{d['display_name']} (best={min(val):.3f})",
                color=color, linewidth=1.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation CTC loss")
    ax.set_title("Validation loss — model comparison")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.savefig(out_dir / "learning_curves_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _normalize_for_metrics(refs: list[str], hyps: list[str], dataset: str
                            ) -> tuple[list[str], list[str]]:
    """Spec §7 — LRS2 normalization is applied at metric-compute time."""
    if dataset == "lrs2":
        refs = [normalize_lrs2(r) for r in refs]
        hyps = [normalize_lrs2(h) for h in hyps]
    return refs, hyps


def write_results_table(eval_jsons: list[Path], out_path: Path) -> None:
    """Table 5 reproduction (per dataset). Refactored from notebook Stage E."""
    rows = []
    by_dataset: dict[str, list[dict]] = {}
    for p in eval_jsons:
        d = _load_eval_json(p)
        by_dataset.setdefault(d["dataset"], []).append(d)

    for dataset, models in by_dataset.items():
        for d in models:
            refs = [pr["reference"] for pr in d["predictions"]]
            hyps = [pr["hypothesis"] for pr in d["predictions"]]
            refs, hyps = _normalize_for_metrics(refs, hyps, dataset)
            rows.append({
                "model": d["display_name"], "dataset": dataset.upper(),
                "method": "—",
                "cer": cer(refs, hyps), "wer": wer(refs, hyps),
                "word_acc": word_acc(refs, hyps),
                "sentence_acc": sentence_acc(refs, hyps),
            })

        # Append paper baselines for GRID
        if dataset == "grid":
            for b in PAPER_BASELINES_GRID:
                rows.append({**b, "cer": None, "sentence_acc": None})

    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def write_qualitative_examples(eval_jsons: list[Path], out_path: Path,
                                n_per_model: int = 5) -> None:
    """Table 6 reproduction — n example clips per model. Stage E."""
    rows = []
    for p in eval_jsons:
        d = _load_eval_json(p)
        for pr in d["predictions"][:n_per_model]:
            rows.append({
                "model": d["display_name"], "dataset": d["dataset"].upper(),
                "reference": pr["reference"], "hypothesis": pr["hypothesis"],
                "path": pr["path"],
            })
    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
