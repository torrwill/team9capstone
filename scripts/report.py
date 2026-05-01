"""Read predictions JSONs → write plots + CSV tables.

Usage:
    python scripts/report.py --predictions-dir results/predictions \
        [--output-dir results]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lsn.evaluation.report import (
    plot_learning_curves, write_qualitative_examples, write_results_table,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions-dir", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    eval_jsons = sorted(args.predictions_dir.glob("*_eval.json"))
    if not eval_jsons:
        print(f"no *_eval.json files in {args.predictions_dir}",
              file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Group by dataset for the per-dataset CSVs
    grid_jsons = [j for j in eval_jsons if "_grid_eval.json" in j.name]
    lrs2_jsons = [j for j in eval_jsons if "_lrs2_eval.json" in j.name]

    plot_learning_curves(eval_jsons, args.output_dir)

    if grid_jsons:
        write_results_table(grid_jsons, args.output_dir / "results_table_grid.csv")
        write_qualitative_examples(
            grid_jsons, args.output_dir / "qualitative_examples_grid.csv",
        )
    if lrs2_jsons:
        write_results_table(lrs2_jsons, args.output_dir / "results_table_lrs2.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
