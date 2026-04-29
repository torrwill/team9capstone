"""One-off helper: extract a specific code cell from the source notebook.

Usage: python tools/dump_cell.py <code-cell-index>
Used by later refactor tasks to copy notebook code into .py files.
"""
import json
import sys
from pathlib import Path

NB = Path("notebooks/legacy/LSN_TRAINING_EVAL.ipynb")

def main(idx: int) -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    src = "".join(code_cells[idx]["source"])
    sys.stdout.write(src)

if __name__ == "__main__":
    main(int(sys.argv[1]))
