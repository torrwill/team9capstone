import json
import platform
from pathlib import Path

import pytest

from lsn.evaluation.inference import Prediction
from lsn.evaluation.report import (
    PAPER_BASELINES_GRID, write_eval_json,
    plot_learning_curves, write_results_table, write_qualitative_examples,
)


def test_paper_baselines_grid_constant_present():
    """Spec §7 — Table 5 paper baselines come from a hardcoded constant."""
    assert isinstance(PAPER_BASELINES_GRID, (list, tuple))
    assert len(PAPER_BASELINES_GRID) >= 3   # Xu, Gergen, Margam at minimum


def test_write_eval_json_roundtrip(tmp_path):
    out_path = tmp_path / "eval.json"
    write_eval_json(
        out_path,
        experiment_name="run_test",
        display_name="Test",
        color="#000000",
        dataset="grid",
        decoder="greedy",
        final_epoch=5,
        best_val_loss=2.5,
        history=[{"epoch": 1, "train_loss": 3.0, "val_loss": 3.1}],
        predictions=[Prediction("/x/a.npz", "hello", "hello")],
    )
    data = json.loads(out_path.read_text())
    assert data["experiment_name"] == "run_test"
    assert data["dataset"] == "grid"
    assert data["predictions"][0]["reference"] == "hello"
    assert data["history"][0]["epoch"] == 1


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="matplotlib rendering crashes on this Windows install (Colab/Kaggle OK)"
)
def test_plot_learning_curves_writes_files(tmp_path):
    eval_json = tmp_path / "run_test_grid_eval.json"
    write_eval_json(
        eval_json, experiment_name="run_test", display_name="Test",
        color="#000000", dataset="grid", decoder="greedy",
        final_epoch=2, best_val_loss=1.0,
        history=[
            {"epoch": 1, "train_loss": 2.0, "val_loss": 2.1},
            {"epoch": 2, "train_loss": 1.5, "val_loss": 1.0},
        ],
        predictions=[],
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    plot_learning_curves([eval_json], out_dir)
    assert (out_dir / "learning_curve_run_test.png").exists()


def test_write_results_table_writes_csv(tmp_path):
    eval_json = tmp_path / "run_test_grid_eval.json"
    write_eval_json(
        eval_json, experiment_name="run_test", display_name="Test",
        color="#000000", dataset="grid", decoder="greedy",
        final_epoch=2, best_val_loss=1.0, history=[],
        predictions=[
            Prediction("/x/a.npz", "hello world", "hello world"),
            Prediction("/x/b.npz", "foo bar baz", "foo bar baz"),
        ],
    )
    out_csv = tmp_path / "results_table_grid.csv"
    write_results_table([eval_json], out_csv)
    assert out_csv.exists()
    text = out_csv.read_text()
    assert "Test" in text
