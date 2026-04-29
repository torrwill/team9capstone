# LSN Notebook → Codebase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `LSN_TRAINING_EVAL.ipynb` (54 code cells) into a presentable Python codebase with clean module boundaries, YAML-driven experiments, and a CLI workflow that runs identically on local, Colab, and Kaggle.

**Architecture:** Pure `.py` codebase with three CLI entry points (`train.py`, `infer.py`, `report.py`) over a `src/lsn/` package. Three committed YAML configs are the reproducibility artifact; per-environment glue is the user's responsibility (documented in README). HuggingFace is opt-in via `--hf-repo`. State_dict layout stays byte-identical to the existing notebook so trained checkpoints load via `strict=True`.

**Tech Stack:** Python 3.10+, PyTorch ≥2.1, torchaudio (CTC beam decoder), torchvision (EfficientNet-B0 weights), HuggingFace Hub (opt-in), pytest, PyYAML, matplotlib, pandas, tqdm.

**Source spec:** `docs/superpowers/specs/2026-04-29-lsn-notebook-to-codebase-design.md`

---

## File-structure overview

See the spec §3 for the full tree. Tasks below are ordered so each task produces a self-contained, testable change. The dependency-critical sequencing is:

1. Repo scaffolding + git init (Task 1)
2. Foundations with no internal deps: `vocab`, `env`, `config` (Tasks 2-4)
3. Models — `frontend` → `backends` → `lipsyncnet` → `__init__` (Tasks 5-8)
4. **Backward-compat canary** — load existing `.pt` checkpoints with `strict=True` (Task 9). This catches state_dict drift before any other work depends on the model classes.
5. Data — datasets, splits, normalize (Tasks 10-12)
6. Training infra — hf_store, checkpoint, loop, runner (Tasks 13-16)
7. Evaluation — decoders, metrics, inference, report (Tasks 17-20)
8. Scripts — train, infer, report CLI wrappers (Tasks 21-23)
9. Committed configs (Task 24)
10. Documentation: data-format, future-work, README (Tasks 25-27)
11. Final smoke-test pass + manual end-to-end verification (Task 28)

**TDD framing for a refactor:** The "feature" being added in each task is a faithful Python module copied/adapted from a specific source-notebook cell. The "test" verifies behavior preservation: identical state_dict keys, identical forward-pass shapes, deterministic round-trips. For modules where the only meaningful test is "does it import and produce the right shapes," the smoke test is short by design — that's correct, not a deficiency.

**Source-notebook reference:** All cell numbers below reference `notebooks/legacy/LSN_TRAINING_EVAL.ipynb` after Task 1 moves it. A standalone helper `tools/dump_cell.py` is created in Task 1 so each subsequent task can extract a specific cell's source code in one command.

---

## Task 1: Repo scaffolding + git init

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `src/lsn/__init__.py` (empty)
- Create: `tools/dump_cell.py` (one-off helper used by later tasks to extract code from the source notebook)
- Move: `LSN_TRAINING_EVAL.ipynb` → `notebooks/legacy/LSN_TRAINING_EVAL.ipynb`
- Init: git repo

- [ ] **Step 1: Initialize git and stage existing files**

```bash
git init
git add CLAUDE.md docs/
```

Note: the other notebooks in the repo root (`Copy of VSR_notebook_v1.ipynb`, `LipSyncNet_Model_effnet.ipynb`, `model_LSN_DRAFT.ipynb`, `(Ran)_model_LSN_DRAFT.ipynb`, `LSN Training.ipynb`) are out of scope for this conversion and are tracked but otherwise untouched. They get added in Step 4 below.

- [ ] **Step 2: Write `.gitignore`**

Create `.gitignore`:

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.venv/
venv/

# Editor
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# LSN-specific outputs (large; per-user)
results/checkpoints/
results/predictions/

# Local Drive/Kaggle paths (in case the user puts data here)
*.npz
data/
```

- [ ] **Step 3: Write `pyproject.toml`**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "lsn"
version = "0.1.0"
description = "LipSyncNet — lip-reading on GRID/LRS2 (capstone refactor)"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1",
    "torchvision>=0.16",
    "torchaudio>=2.1",
    "numpy>=1.24",
    "tqdm",
    "pyyaml",
    "matplotlib",
    "pandas",
    "huggingface_hub",
]

[project.optional-dependencies]
dev = ["pytest>=7"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 4: Create directory skeleton and move legacy notebook**

```bash
mkdir -p src/lsn/models src/lsn/data src/lsn/training src/lsn/evaluation
mkdir -p scripts tests configs results docs notebooks/legacy tools

mv "LSN_TRAINING_EVAL.ipynb" "notebooks/legacy/LSN_TRAINING_EVAL.ipynb"
touch src/lsn/__init__.py
touch src/lsn/models/__init__.py src/lsn/data/__init__.py src/lsn/training/__init__.py src/lsn/evaluation/__init__.py
touch tests/__init__.py
```

- [ ] **Step 5: Create `tools/dump_cell.py` (one-off helper for later tasks)**

Create `tools/dump_cell.py`:

```python
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
```

Verify: `python tools/dump_cell.py 0` should print the imports cell from the notebook.

- [ ] **Step 6: Install package in editable mode**

```bash
pip install -e ".[dev]"
```

Expected: success. Verifies the package layout is valid.

- [ ] **Step 7: Verify pytest discovers no tests yet (clean baseline)**

```bash
pytest -q
```

Expected: `no tests ran`. Confirms the test harness is wired up.

- [ ] **Step 8: First commit**

```bash
git add -A
git commit -m "chore: initialize repo scaffolding

- pyproject.toml with pinned dependencies (spec §9.3)
- src/lsn/ package skeleton
- tests/ + configs/ + results/ + docs/ + notebooks/legacy/ dirs
- .gitignore for python + per-user run outputs
- move LSN_TRAINING_EVAL.ipynb to notebooks/legacy/
- tools/dump_cell.py helper for cell-by-cell refactor"
```

---

## Task 2: `lsn/data/vocab.py` — vocabulary constants and encode/decode

**Files:**
- Create: `src/lsn/data/vocab.py`
- Create: `tests/test_vocab.py`

**Source notebook reference:** code cells 14-16 (`# SETUP Configuration`, `VOCAB`, `encode_text`, `decode_ids`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_vocab.py`:

```python
import torch
from lsn.data import vocab as v


def test_vocab_constants_load_bearing():
    # vocab_size=40 is preserved verbatim from trained checkpoints (spec §4.1).
    # CHARS is the actual 27-char alphabet; vocab_size>len(CHARS) is intentional.
    assert v.VOCAB_SIZE == 40
    assert v.NUM_CLASSES == 41
    assert v.BLANK_INDEX == 0
    assert v.CHARS[0] == " "
    assert v.CHARS[-1] == "z"
    assert len(v.CHARS) == 27


def test_encode_decode_roundtrip():
    text = "set white at b nine again"
    ids = v.encode_text(text)
    assert isinstance(ids, torch.Tensor)
    assert ids.dtype == torch.long
    decoded = v.decode_ids(ids.tolist())
    assert decoded == text


def test_decode_skips_blank():
    # blank index 0 must be filtered out during decode (CTC convention)
    ids = [0, v.char_to_idx["a"], 0, v.char_to_idx["b"], 0]
    assert v.decode_ids(ids) == "ab"


def test_encode_lowercases():
    assert v.decode_ids(v.encode_text("HELLO").tolist()) == "hello"
```

- [ ] **Step 2: Run test, verify failure**

```bash
pytest tests/test_vocab.py -v
```

Expected: `ImportError` / `ModuleNotFoundError`. Module doesn't exist yet.

- [ ] **Step 3: Implement `lsn/data/vocab.py`**

Create `src/lsn/data/vocab.py`:

```python
"""Character vocabulary and encode/decode helpers.

Note on `VOCAB_SIZE = 40` (load-bearing — see spec §4.1):
The trained checkpoints were instantiated with `vocab_size=40`, producing a
classifier of shape `Linear(1024, 41)` (40 + 1 CTC blank). The actual
encodable alphabet (CHARS below) is only 27 characters, so the classifier
has 41 channels of which 28 are populated (blank + space + 26 letters).
Do NOT "fix" VOCAB_SIZE to len(CHARS) — that would change the classifier's
weight shape and break strict=True load of every existing .pt file.
"""
from __future__ import annotations

import string

import torch

# 27-char alphabet: space + a-z (lowercase, no digits/punctuation in GRID).
CHARS: list[str] = [" "] + list(string.ascii_lowercase)

# Reserved for CTC blank — PyTorch CTCLoss(blank=0) convention.
BLANK_INDEX: int = 0

# IDs start at 1 (0 is reserved for blank).
char_to_idx: dict[str, int] = {ch: i + 1 for i, ch in enumerate(CHARS)}
idx_to_char: dict[int, str] = {i + 1: ch for i, ch in enumerate(CHARS)}

# Load-bearing constants — see module docstring.
VOCAB_SIZE: int = 40
NUM_CLASSES: int = VOCAB_SIZE + 1   # 41 — includes blank


def encode_text(text: str) -> torch.Tensor:
    """Encode lowercased text → tensor of int ids."""
    text = text.lower().strip()
    ids = [char_to_idx[ch] for ch in text]
    return torch.tensor(ids, dtype=torch.long)


def decode_ids(ids) -> str:
    """Decode a sequence of int ids → string. Skips BLANK_INDEX (CTC)."""
    chars = []
    for idx in ids:
        idx = int(idx)
        if idx == BLANK_INDEX:
            continue
        chars.append(idx_to_char[idx])
    return "".join(chars)
```

- [ ] **Step 4: Run test, verify pass**

```bash
pytest tests/test_vocab.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/lsn/data/vocab.py tests/test_vocab.py
git commit -m "feat(data): vocab constants + encode/decode

VOCAB_SIZE=40 preserved as load-bearing checkpoint constant per spec §4.1.
Refactored from notebook cells 14-16."
```

---

## Task 3: `lsn/env.py` — device, seeding, cudnn config

**Files:**
- Create: `src/lsn/env.py`
- Create: `tests/test_env.py`

**Source notebook reference:** code cell 2 (cudnn benchmark + seeding + device printing).

- [ ] **Step 1: Write the failing test**

Create `tests/test_env.py`:

```python
import random

import numpy as np
import torch

from lsn import env


def test_set_seed_makes_python_random_deterministic():
    env.set_seed(42)
    a = [random.random() for _ in range(5)]
    env.set_seed(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_seed_makes_numpy_deterministic():
    env.set_seed(42)
    a = np.random.rand(5)
    env.set_seed(42)
    b = np.random.rand(5)
    assert (a == b).all()


def test_set_seed_makes_torch_deterministic():
    env.set_seed(42)
    a = torch.randn(5)
    env.set_seed(42)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_get_device_returns_torch_device():
    d = env.get_device(None)
    assert isinstance(d, torch.device)
    assert d.type in ("cuda", "cpu")


def test_get_device_respects_cpu_override():
    d = env.get_device("cpu")
    assert d.type == "cpu"


def test_configure_cudnn_does_not_crash():
    env.configure_cudnn(benchmark=True)
    env.configure_cudnn(benchmark=False)
```

- [ ] **Step 2: Run test, verify failure**

```bash
pytest tests/test_env.py -v
```

Expected: `ModuleNotFoundError: No module named 'lsn.env'`.

- [ ] **Step 3: Implement `src/lsn/env.py`**

```python
"""Environment helpers: deterministic seeding, device selection, cudnn config.

Refactored from notebook cell 2. The codebase deliberately has no
environment-detection branches (no `if 'google.colab' in sys.modules`) — all
env-specific glue is the user's responsibility (README has setup blocks).
"""
from __future__ import annotations

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python random, numpy, and torch (CPU + CUDA all visible devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(override: str | None) -> torch.device:
    """Return torch.device. Honors --device CLI override; otherwise auto-detect."""
    if override is not None:
        return torch.device(override)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            logger.info("cuda:%d = %s, %.1f GB", i, p.name, p.total_memory / 1e9)
        return device
    return torch.device("cpu")


def configure_cudnn(benchmark: bool) -> None:
    """Set cudnn.benchmark.

    True for training (matches notebook): faster, non-deterministic conv algos.
    False for inference: deterministic eval.
    """
    torch.backends.cudnn.benchmark = benchmark
```

- [ ] **Step 4: Run test, verify pass**

```bash
pytest tests/test_env.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/lsn/env.py tests/test_env.py
git commit -m "feat(env): seeding, device selection, cudnn config

Refactored from notebook cell 2. No env-detection branches per spec §9.2."
```

---

## Task 4: `lsn/config.py` — YAML schema + loader + CLI override merge

**Files:**
- Create: `src/lsn/config.py`
- Create: `tests/test_config.py`
- Create: `tests/fixtures/sample_config.yaml`

**Source notebook reference:** none (config is new — replaces hardcoded notebook constants).

- [ ] **Step 1: Create the test fixture**

Create `tests/fixtures/sample_config.yaml`:

```yaml
experiment_name: run_test_v1

model:
  backend: identity
  vocab_size: 40
  freeze_early_effnet: true
  use_self_attn: false
  backend_kwargs: {}
  display_name: "Test"
  color: "#000000"

training:
  num_epochs: 5
  learning_rate: 1.0e-4
  batch_size: 2
  accum_steps: 4
  max_grad_norm: 5.0
  use_amp: true
  num_workers: 2
  prefetch: 4

data:
  dataset: grid
  speakers: [s1, s2, s3, s4, s5]
  samples_per_speaker: 200
  train_size: 450
  seed: 42

checkpointing:
  hf_repo: null
  hf_subfolder: null
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_config.py`:

```python
import argparse
from pathlib import Path

import pytest

from lsn.config import (
    Config, ModelCfg, TrainingCfg, DataCfg, CkptCfg,
    load_config, apply_cli_overrides,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_config.yaml"


def test_load_config_returns_typed_dataclass():
    cfg = load_config(FIXTURE)
    assert isinstance(cfg, Config)
    assert cfg.experiment_name == "run_test_v1"
    assert isinstance(cfg.model, ModelCfg)
    assert isinstance(cfg.training, TrainingCfg)
    assert isinstance(cfg.data, DataCfg)
    assert isinstance(cfg.checkpointing, CkptCfg)


def test_load_config_parses_model_fields():
    cfg = load_config(FIXTURE)
    assert cfg.model.backend == "identity"
    assert cfg.model.vocab_size == 40
    assert cfg.model.use_self_attn is False
    assert cfg.model.color == "#000000"


def test_load_config_parses_training_fields():
    cfg = load_config(FIXTURE)
    assert cfg.training.num_epochs == 5
    assert cfg.training.learning_rate == 1e-4
    assert cfg.training.batch_size == 2
    assert cfg.training.use_amp is True


def test_load_config_parses_data_fields():
    cfg = load_config(FIXTURE)
    assert cfg.data.speakers == ["s1", "s2", "s3", "s4", "s5"]
    assert cfg.data.train_size == 450
    assert cfg.data.seed == 42


def test_load_config_handles_null_hf():
    cfg = load_config(FIXTURE)
    assert cfg.checkpointing.hf_repo is None
    assert cfg.checkpointing.hf_subfolder is None


def test_apply_cli_overrides_only_touches_allowed_fields():
    cfg = load_config(FIXTURE)
    args = argparse.Namespace(
        hf_repo="ranro1/test", epochs=99,
    )
    cfg2 = apply_cli_overrides(cfg, args)
    assert cfg2.checkpointing.hf_repo == "ranro1/test"
    assert cfg2.training.num_epochs == 99
    # Other fields untouched
    assert cfg2.training.learning_rate == 1e-4


def test_apply_cli_overrides_ignores_none_values():
    cfg = load_config(FIXTURE)
    args = argparse.Namespace(hf_repo=None, epochs=None)
    cfg2 = apply_cli_overrides(cfg, args)
    assert cfg2.checkpointing.hf_repo is None
    assert cfg2.training.num_epochs == 5  # unchanged
```

- [ ] **Step 3: Run test, verify failure**

```bash
pytest tests/test_config.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Implement `src/lsn/config.py`**

```python
"""YAML config schema (typed dataclasses) + loader + CLI override merge.

See spec §4.1 for the full schema. CLI flags override only env-specific
fields; hyperparameters live in YAML (spec §6).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelCfg:
    backend: str
    vocab_size: int = 40
    freeze_early_effnet: bool = True
    use_self_attn: bool = False
    backend_kwargs: dict[str, Any] = field(default_factory=dict)
    display_name: str | None = None
    color: str | None = None


@dataclass
class TrainingCfg:
    num_epochs: int
    learning_rate: float
    batch_size: int
    accum_steps: int
    max_grad_norm: float
    use_amp: bool
    num_workers: int
    prefetch: int


@dataclass
class DataCfg:
    dataset: str
    speakers: list[str]
    samples_per_speaker: int
    train_size: int
    seed: int


@dataclass
class CkptCfg:
    hf_repo: str | None = None
    hf_subfolder: str | None = None


@dataclass
class Config:
    experiment_name: str
    model: ModelCfg
    training: TrainingCfg
    data: DataCfg
    checkpointing: CkptCfg


def load_config(path: str | Path) -> Config:
    """Load YAML at `path` into a typed Config."""
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return Config(
        experiment_name=raw["experiment_name"],
        model=ModelCfg(**raw["model"]),
        training=TrainingCfg(**raw["training"]),
        data=DataCfg(**raw["data"]),
        checkpointing=CkptCfg(**raw["checkpointing"]),
    )


def apply_cli_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Mutate config in-place from CLI args. Only fields explicitly listed in
    spec §6 (overridable) are touched; everything else stays YAML-driven.

    Recognized args: hf_repo, epochs.
    """
    hf_repo = getattr(args, "hf_repo", None)
    if hf_repo is not None:
        cfg.checkpointing.hf_repo = hf_repo

    epochs = getattr(args, "epochs", None)
    if epochs is not None:
        cfg.training.num_epochs = epochs

    return cfg
```

- [ ] **Step 5: Run test, verify pass**

```bash
pytest tests/test_config.py -v
```

Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add src/lsn/config.py tests/test_config.py tests/fixtures/sample_config.yaml
git commit -m "feat(config): YAML schema + loader + CLI override merge (spec §4.1, §6)"
```

---

## Task 5: `lsn/models/frontend.py` — Conv3DBlock, Frontend3DCNN, EfficientNet

**Files:**
- Create: `src/lsn/models/frontend.py`
- Create: `tests/test_models_frontend.py`

**Source notebook reference:** code cells 3, 4, 5.

**Implementation note:** Copy the class definitions verbatim from cells 3–5 (including docstrings — they contain load-bearing rationale per spec §12). The notebook freezes EfficientNet stages 0–6 and trains stages 7–8 — preserve this exactly.

- [ ] **Step 1: Write the failing test**

Create `tests/test_models_frontend.py`:

```python
import torch

from lsn.models.frontend import Conv3DBlock, Frontend3DCNN, EfficientNet


def test_conv3d_block_forward_shape():
    block = Conv3DBlock(in_channels=1, out_channels=64,
                        kernel_size=(3, 5, 5), padding=(1, 2, 2))
    x = torch.randn(1, 1, 75, 46, 140)
    y = block(x)
    # MaxPool3d (1,2,2) halves spatial dims
    assert y.shape == (1, 64, 75, 23, 70)


def test_frontend3dcnn_forward_shape():
    fe = Frontend3DCNN()
    x = torch.randn(2, 1, 75, 46, 140)
    y = fe(x)
    # 4 blocks → (B, T, 8192) per Frontend3DCNN docstring
    assert y.shape == (2, 75, 8192)
    assert Frontend3DCNN.FLAT_DIM == 8192


def test_efficientnet_forward_shape():
    eff = EfficientNet(freeze_early=True)
    x = torch.randn(2, 1, 46, 140)   # (B*T, 1, H, W)
    y = eff(x)
    assert y.shape == (2, 62720)
    assert EfficientNet.FEATURE_DIM == 62720


def test_efficientnet_freeze_policy():
    """Stages 0-6 frozen, 7-8 trainable (spec §10, notebook cell 5)."""
    eff = EfficientNet(freeze_early=True)
    for stage_idx in range(7):
        for p in eff.features[stage_idx].parameters():
            assert p.requires_grad is False
    for stage_idx in (7, 8):
        for p in eff.features[stage_idx].parameters():
            assert p.requires_grad is True
```

- [ ] **Step 2: Run test, verify failure**

```bash
pytest tests/test_models_frontend.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/lsn/models/frontend.py`**

Extract from notebook:

```bash
python tools/dump_cell.py 3 > /tmp/cell3.py
python tools/dump_cell.py 4 > /tmp/cell4.py
python tools/dump_cell.py 5 > /tmp/cell5.py
```

Create `src/lsn/models/frontend.py` by concatenating:
- A module docstring referencing spec §10 (state_dict invariant) and the load-bearing freeze policy
- The required imports: `torch`, `torch.nn as nn`, `torch.nn.functional as F`, `from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights`
- The three classes from cells 3, 4, 5 (Conv3DBlock, Frontend3DCNN, EfficientNet) verbatim — DO NOT modify class names, attribute names, or __init__ argument order. State_dict invariant per spec §10.

Drop notebook prints; add no logging in this module.

- [ ] **Step 4: Run test, verify pass**

```bash
pytest tests/test_models_frontend.py -v
```

Expected: 4 passed. (First run downloads EfficientNet-B0 ImageNet weights — ~20 MB, may take a minute.)

- [ ] **Step 5: Commit**

```bash
git add src/lsn/models/frontend.py tests/test_models_frontend.py
git commit -m "feat(models): frontend — Conv3DBlock, Frontend3DCNN, EfficientNet

Refactored verbatim from notebook cells 3-5. Class names + attribute names
preserved per state_dict invariant (spec §10). Stages 0-6 frozen."
```

---

## Task 6: `lsn/models/backends.py` — `_SinusoidalPE`, all four backends, registry

**Files:**
- Create: `src/lsn/models/backends.py`
- Create: `tests/test_models_backends.py`

**Source notebook reference:** code cells 6, 7, 8.

**Implementation note:** Cell 6 has BiLSTMBackend, IdentityBackend, _SinusoidalPE, TransformerBackend. Cell 7 has TransformerBackendPerStream. Cell 8 registers `transformer_perstream` in `_BACKEND_REGISTRY`. Combine all into `backends.py`. **Critical:** the registry's keys `"bilstm"`, `"identity"`, `"transformer"`, `"transformer_perstream"` must be preserved exactly — they're the public dispatch surface.

- [ ] **Step 1: Write the failing test**

Create `tests/test_models_backends.py`:

```python
import torch

from lsn.models.backends import (
    _BACKEND_REGISTRY, BiLSTMBackend, IdentityBackend,
    TransformerBackend, TransformerBackendPerStream,
    _SinusoidalPE,
)


def test_registry_keys():
    """Registry keys are public dispatch surface — DO NOT rename."""
    assert set(_BACKEND_REGISTRY.keys()) == {
        "bilstm", "identity", "transformer", "transformer_perstream",
    }


def test_bilstm_backend_shapes():
    b = BiLSTMBackend(input_dim=70912, hidden=512, dropout=0.5)
    x = torch.randn(2, 75, 70912)
    y = b(x)
    assert y.shape == (2, 75, 1024)
    assert b.out_dim == 1024


def test_identity_backend_passthrough():
    b = IdentityBackend(input_dim=70912)
    x = torch.randn(2, 75, 70912)
    y = b(x)
    assert torch.equal(y, x)
    assert b.out_dim == 70912


def test_transformer_backend_shapes():
    b = TransformerBackend(input_dim=70912)
    x = torch.randn(2, 75, 70912)
    y = b(x)
    assert y.shape[0] == 2
    assert y.shape[1] == 75
    # default d_model=1024
    assert y.shape[2] == 1024


def test_transformer_perstream_backend_shapes():
    b = TransformerBackendPerStream(d_model=1024, nhead=4, num_layers=2)
    cnn_dim = 8192
    eff_dim = 62720
    fused = torch.randn(2, 75, cnn_dim + eff_dim)
    y = b(fused)
    assert y.shape == (2, 75, 1024)


def test_sinusoidal_pe_buffer_shape():
    pe = _SinusoidalPE(d_model=1024, max_len=75)
    assert pe.pe.shape == (1, 75, 1024)
```

- [ ] **Step 2: Run test, verify failure**

```bash
pytest tests/test_models_backends.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/lsn/models/backends.py`**

Extract:

```bash
python tools/dump_cell.py 6 > /tmp/cell6.py   # BiLSTMBackend, IdentityBackend, _SinusoidalPE, TransformerBackend
python tools/dump_cell.py 7 > /tmp/cell7.py   # TransformerBackendPerStream
python tools/dump_cell.py 8 > /tmp/cell8.py   # registry registration
```

Create `src/lsn/models/backends.py`:
- Module docstring referencing spec §10 (state_dict invariant for `backend.input_proj`, `backend.cnn_proj`, `backend.eff_proj`, `backend.encoder.layers`, `backend.pos_enc.pe`)
- Imports from `frontend.py`: `from lsn.models.frontend import Frontend3DCNN, EfficientNet` — needed because `TransformerBackendPerStream` references `Frontend3DCNN.FLAT_DIM` and `EfficientNet.FEATURE_DIM`
- Other imports: `math`, `torch`, `torch.nn as nn`
- Classes from cells 6, 7 verbatim
- The registry definition from cell 6 PLUS the registration from cell 8:
  ```python
  _BACKEND_REGISTRY = {
      "bilstm":      BiLSTMBackend,
      "identity":    IdentityBackend,
      "transformer": TransformerBackend,
      "transformer_perstream": TransformerBackendPerStream,
  }
  ```

Note: in cell 6 the registry initially has only 3 entries; cell 8 adds the 4th. Combine both upfront in the .py file.

- [ ] **Step 4: Run test, verify pass**

```bash
pytest tests/test_models_backends.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/lsn/models/backends.py tests/test_models_backends.py
git commit -m "feat(models): backends + _SinusoidalPE + _BACKEND_REGISTRY

Refactored from notebook cells 6-8. Registry keys (bilstm, identity,
transformer, transformer_perstream) preserved per spec §10."
```

---

## Task 7: `lsn/models/lipsyncnet.py` — `SelfAttentionBlock`, `LipSyncNetPaper`, `LipSyncNetVariant`

**Files:**
- Create: `src/lsn/models/lipsyncnet.py`
- Create: `tests/test_models_lipsyncnet.py`

**Source notebook reference:** code cells 9, 10, 11.

- [ ] **Step 1: Write the failing test**

Create `tests/test_models_lipsyncnet.py`:

```python
import torch

from lsn.models.lipsyncnet import (
    LipSyncNetPaper, LipSyncNetVariant, SelfAttentionBlock,
)


def test_self_attention_block_shape_preserving():
    sab = SelfAttentionBlock(embed_dim=1024, num_heads=8)
    x = torch.randn(2, 75, 1024)
    y = sab(x)
    assert y.shape == x.shape


def test_lipsyncnet_paper_no_self_attn_forward():
    """The paper checkpoint was trained with use_self_attn=False (spec §5).
    This is the variant that must exist for checkpoint compatibility."""
    model = LipSyncNetPaper(vocab_size=40, use_self_attn=False)
    x = torch.randn(2, 75, 46, 140)
    y = model(x)
    # Expected output: (T, B, vocab+1) = (75, 2, 41)
    assert y.shape == (75, 2, 41)


def test_lipsyncnet_paper_with_self_attn_forward():
    model = LipSyncNetPaper(vocab_size=40, use_self_attn=True)
    x = torch.randn(2, 75, 46, 140)
    y = model(x)
    assert y.shape == (75, 2, 41)
    assert hasattr(model, "self_attn")


def test_lipsyncnet_variant_identity_no_lstm():
    model = LipSyncNetVariant(backend="identity", vocab_size=40)
    n_lstm = sum(1 for m in model.modules() if isinstance(m, torch.nn.LSTM))
    assert n_lstm == 0


def test_lipsyncnet_variant_all_backends_forward_shape():
    for backend in ("bilstm", "identity", "transformer", "transformer_perstream"):
        kwargs = {}
        if backend in ("transformer", "transformer_perstream"):
            kwargs = {"d_model": 1024, "nhead": 4, "num_layers": 2}
        model = LipSyncNetVariant(backend=backend, vocab_size=40, **kwargs)
        x = torch.randn(2, 75, 46, 140)
        y = model(x)
        assert y.shape == (75, 2, 41), f"backend={backend} returned {y.shape}"


def test_lipsyncnet_state_dict_keys_top_level():
    """Spec §10 — top-level attribute names must be preserved."""
    paper = LipSyncNetPaper(vocab_size=40, use_self_attn=False)
    keys = set(paper.state_dict().keys())
    # Expect cnn3d.*, efficientnet.*, lstm1.*, lstm2.*, classifier.* present
    assert any(k.startswith("cnn3d.") for k in keys)
    assert any(k.startswith("efficientnet.") for k in keys)
    assert any(k.startswith("lstm1.") for k in keys)
    assert any(k.startswith("lstm2.") for k in keys)
    assert any(k.startswith("classifier.") for k in keys)
    # No self_attn keys when use_self_attn=False
    assert not any(k.startswith("self_attn.") for k in keys)
```

- [ ] **Step 2: Run test, verify failure**

```bash
pytest tests/test_models_lipsyncnet.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/lsn/models/lipsyncnet.py`**

Extract from notebook:

```bash
python tools/dump_cell.py 9  > /tmp/cell9.py    # SelfAttentionBlock
python tools/dump_cell.py 10 > /tmp/cell10.py   # LipSyncNetPaper
python tools/dump_cell.py 11 > /tmp/cell11.py   # LipSyncNetVariant
```

Create `src/lsn/models/lipsyncnet.py`:
- Module docstring with the cell-1/cell-2 markdown rationale (the load-bearing paper-vs-impl divergences — see spec §12 last bullet). Inline these notes near the relevant classes:
  - `LipSyncNetPaper` docstring: paper-Table-1 freeze count discrepancy + LSTM count anomaly + self-attn ambiguity
  - `SelfAttentionBlock` docstring: design choices forced by surrounding shapes (already in notebook)
- Imports: `torch`, `nn`, `F`, plus `from lsn.models.frontend import Frontend3DCNN, EfficientNet` and `from lsn.models.backends import _BACKEND_REGISTRY`
- Three classes from cells 9, 10, 11 verbatim — DO NOT rename `cnn3d`, `efficientnet`, `lstm1`, `drop1`, `lstm2`, `drop2`, `self_attn`, `classifier`, `backend`. State_dict invariant per spec §10.

- [ ] **Step 4: Run test, verify pass**

```bash
pytest tests/test_models_lipsyncnet.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/lsn/models/lipsyncnet.py tests/test_models_lipsyncnet.py
git commit -m "feat(models): LipSyncNetPaper, LipSyncNetVariant, SelfAttentionBlock

Refactored from notebook cells 9-11. Top-level attribute names preserved
per state_dict invariant (spec §10). Paper-vs-impl rationale moved into
class docstrings (spec §12)."
```

---

## Task 8: `lsn/models/__init__.py` — public builders + `build_from_config`

**Files:**
- Modify: `src/lsn/models/__init__.py`
- Create: `tests/test_models_builders.py`

**Source notebook reference:** code cell 12 (`build_paper_model`, `build_variant`, `count_parameters`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_models_builders.py`:

```python
import torch

from lsn.config import ModelCfg
from lsn.models import (
    build_paper_model, build_variant, count_parameters, build_from_config,
)
from lsn.models.lipsyncnet import LipSyncNetPaper, LipSyncNetVariant


def test_build_paper_model_returns_paper_class():
    m = build_paper_model(vocab_size=40, use_self_attn=False, device="cpu")
    assert isinstance(m, LipSyncNetPaper)


def test_build_variant_returns_variant_class():
    for backend in ("bilstm", "identity"):
        m = build_variant(backend=backend, vocab_size=40, device="cpu")
        assert isinstance(m, LipSyncNetVariant)


def test_count_parameters_keys():
    m = build_variant(backend="identity", vocab_size=40)
    counts = count_parameters(m)
    assert set(counts.keys()) == {"total", "trainable", "frozen"}
    assert counts["total"] == counts["trainable"] + counts["frozen"]


def test_build_from_config_dispatches_paper():
    cfg = ModelCfg(backend="paper", vocab_size=40, use_self_attn=False)
    m = build_from_config(cfg, device=torch.device("cpu"))
    assert isinstance(m, LipSyncNetPaper)


def test_build_from_config_dispatches_variant():
    for backend in ("identity", "bilstm", "transformer", "transformer_perstream"):
        kwargs = {}
        if backend.startswith("transformer"):
            kwargs = {"d_model": 1024, "nhead": 4, "num_layers": 2}
        cfg = ModelCfg(backend=backend, vocab_size=40, backend_kwargs=kwargs)
        m = build_from_config(cfg, device=torch.device("cpu"))
        assert isinstance(m, LipSyncNetVariant)


def test_build_from_config_unknown_backend_raises():
    import pytest
    cfg = ModelCfg(backend="bogus", vocab_size=40)
    with pytest.raises(ValueError):
        build_from_config(cfg, device=torch.device("cpu"))
```

- [ ] **Step 2: Run test, verify failure**

```bash
pytest tests/test_models_builders.py -v
```

Expected: `ImportError` — `cannot import name 'build_paper_model' from 'lsn.models'`.

- [ ] **Step 3: Implement `src/lsn/models/__init__.py`**

```python
"""Public API for the lsn.models package.

Mirrors notebook cell 12. Adds `build_from_config` for YAML-driven dispatch.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lsn.config import ModelCfg
from lsn.models.lipsyncnet import LipSyncNetPaper, LipSyncNetVariant


def build_paper_model(vocab_size: int = 40,
                      use_self_attn: bool = False,
                      device: str = "cpu") -> LipSyncNetPaper:
    """Instantiate the paper-faithful model."""
    return LipSyncNetPaper(
        vocab_size=vocab_size, use_self_attn=use_self_attn,
    ).to(device)


def build_variant(backend: str = "bilstm",
                  vocab_size: int = 40,
                  device: str = "cpu",
                  **backend_kwargs) -> LipSyncNetVariant:
    """Instantiate the modular variant with chosen backend."""
    return LipSyncNetVariant(
        backend=backend, vocab_size=vocab_size, **backend_kwargs,
    ).to(device)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total / trainable / frozen parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def build_from_config(cfg: ModelCfg, device: torch.device) -> nn.Module:
    """Dispatch from a ModelCfg to the right builder.

    `backend == "paper"` routes to LipSyncNetPaper; everything else routes
    to LipSyncNetVariant via the `_BACKEND_REGISTRY`.
    """
    if cfg.backend == "paper":
        return build_paper_model(
            vocab_size=cfg.vocab_size,
            use_self_attn=cfg.use_self_attn,
            device=str(device),
        )
    # Validate against the registry without leaking it
    from lsn.models.backends import _BACKEND_REGISTRY
    if cfg.backend not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend {cfg.backend!r}. "
            f"Valid options: paper, {', '.join(sorted(_BACKEND_REGISTRY))}"
        )
    return build_variant(
        backend=cfg.backend, vocab_size=cfg.vocab_size, device=str(device),
        **cfg.backend_kwargs,
    )


__all__ = [
    "build_paper_model", "build_variant", "count_parameters",
    "build_from_config",
]
```

- [ ] **Step 4: Run test, verify pass**

```bash
pytest tests/test_models_builders.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/lsn/models/__init__.py tests/test_models_builders.py
git commit -m "feat(models): public builders + build_from_config

Refactored from notebook cell 12. Adds YAML-driven dispatch (spec §4.3)."
```

---

## Task 9: 🚨 Backward-compat canary — load existing checkpoints with `strict=True`

**Critical step.** Per spec §14, this verifies the §10 invariant before downstream work depends on the model classes. If state_dict drift is going to bite, it bites here.

**Files:**
- Create: `tests/test_checkpoint_compat.py`

**Pre-requisite:** the user has the three existing `best_model.pt` files. Two acceptable locations:
- Local: `<some-path>/{paper,identity,transformer}_best_model.pt`
- HuggingFace: `ranro1/lipsyncnet-checkpoints/run_<name>_v1/best_model.pt`

The test reads the path from an env var so it's gated cleanly on availability.

- [ ] **Step 1: Write the test**

Create `tests/test_checkpoint_compat.py`:

```python
"""Backward-compat canary — load each existing best_model.pt with strict=True.

This test enforces the spec §10 state_dict invariant. Any rename of a model
attribute (e.g., cnn3d → frontend) breaks this test.

Set LSN_CKPT_DIR to a local directory containing:
    {LSN_CKPT_DIR}/paper_best_model.pt
    {LSN_CKPT_DIR}/identity_best_model.pt
    {LSN_CKPT_DIR}/transformer_best_model.pt

If LSN_CKPT_DIR is unset, this test is skipped (fine for CI without secrets).
"""
import os
from pathlib import Path

import pytest
import torch

from lsn.models import build_from_config
from lsn.config import ModelCfg


CKPT_DIR = os.environ.get("LSN_CKPT_DIR")
pytestmark = pytest.mark.skipif(
    CKPT_DIR is None,
    reason="LSN_CKPT_DIR not set — see test docstring",
)


CONFIGS = {
    "paper": ModelCfg(
        backend="paper", vocab_size=40,
        freeze_early_effnet=True, use_self_attn=False,
    ),
    "identity": ModelCfg(
        backend="identity", vocab_size=40, freeze_early_effnet=True,
    ),
    "transformer": ModelCfg(
        backend="transformer_perstream", vocab_size=40,
        freeze_early_effnet=True,
        backend_kwargs={"d_model": 1024, "nhead": 4, "num_layers": 2},
    ),
}


@pytest.mark.parametrize("name,cfg", list(CONFIGS.items()))
def test_existing_checkpoint_loads_strict(name: str, cfg: ModelCfg):
    ckpt_path = Path(CKPT_DIR) / f"{name}_best_model.pt"
    if not ckpt_path.exists():
        pytest.skip(f"checkpoint not found at {ckpt_path}")

    model = build_from_config(cfg, device=torch.device("cpu"))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    # Assert strict load — any drift raises RuntimeError.
    model.load_state_dict(state_dict, strict=True)
```

- [ ] **Step 2: Run test (skipped if no checkpoints)**

```bash
pytest tests/test_checkpoint_compat.py -v
```

Expected (no env var set): 3 skipped.

- [ ] **Step 3: Run test against actual checkpoints**

The user puts the three existing `best_model.pt` files in a local directory (e.g., copy from Drive or download from HF), then:

```bash
LSN_CKPT_DIR=/path/to/checkpoints pytest tests/test_checkpoint_compat.py -v
```

Expected: 3 passed.

**If this fails:** state_dict drift was introduced. The error message names the missing/unexpected keys. Diff against the spec §10 frozen-key list and revert the offending rename. DO NOT proceed to subsequent tasks until this passes — every downstream module assumes these checkpoints load.

- [ ] **Step 4: Commit**

```bash
git add tests/test_checkpoint_compat.py
git commit -m "test: checkpoint backward-compat canary

Enforces state_dict invariant (spec §10). Skipped without LSN_CKPT_DIR;
when set, loads each existing best_model.pt with strict=True.

This is the gold-standard verification that the model refactor preserved
the trained-checkpoint contract."
```

---

## Task 10: `lsn/data/datasets.py` — Grid + LRS2 datasets, collate

**Files:**
- Create: `src/lsn/data/datasets.py`
- Create: `tests/test_datasets.py`

**Source notebook reference:** code cells 17 (`GridLipReadingDataset`), 19 (`grid_collate_fn`), and the LRS2 dataset code that appears in the LRS2 stages (look for `LRS2LipReadingDataset` definition near Stage G).

- [ ] **Step 1: Locate LRS2Dataset source**

```bash
python -c "
import json
nb = json.load(open('notebooks/legacy/LSN_TRAINING_EVAL.ipynb', encoding='utf-8'))
codes = [c for c in nb['cells'] if c['cell_type']=='code']
for i, c in enumerate(codes):
    src = ''.join(c['source'])
    if 'LRS2' in src and 'class' in src:
        print(f'cell {i}:'); print(src[:200]); print('---')
"
```

Note the cell index for the LRS2 dataset class. Refer to it as `LRS2_DATASET_CELL` below (replace with the real number).

- [ ] **Step 2: Write the failing test**

Create `tests/test_datasets.py`:

```python
from pathlib import Path

import numpy as np
import pytest
import torch

from lsn.data.datasets import GridLipReadingDataset, grid_collate_fn


def _write_npz(path: Path, label: str = "set white at b nine again",
               with_channel: bool = False) -> None:
    if with_channel:
        frames = np.zeros((75, 46, 140, 1), dtype=np.float32)
    else:
        frames = np.zeros((75, 46, 140), dtype=np.float32)
    np.savez(path, frames=frames, label=label)


def test_grid_dataset_loads_npz_no_channel(tmp_path):
    p = tmp_path / "s1_001.npz"
    _write_npz(p)
    ds = GridLipReadingDataset([p])
    sample = ds[0]
    assert sample["frames"].shape == (75, 46, 140)
    assert sample["frames"].dtype == torch.float32
    assert sample["text"] == "set white at b nine again"
    assert sample["target_length"] == len(sample["text"])
    assert sample["path"] == str(p)


def test_grid_dataset_loads_npz_with_channel(tmp_path):
    p = tmp_path / "s1_001.npz"
    _write_npz(p, with_channel=True)
    ds = GridLipReadingDataset([p])
    sample = ds[0]
    # trailing channel-1 dim is squeezed
    assert sample["frames"].shape == (75, 46, 140)


def test_grid_dataset_validates_shape(tmp_path):
    """Spec §9.6 — boundary validation with clear ValueError."""
    p = tmp_path / "bad.npz"
    np.savez(p, frames=np.zeros((75, 50, 140), dtype=np.float32),
             label="hello")
    ds = GridLipReadingDataset([p])
    with pytest.raises(ValueError):
        ds[0]


def test_grid_collate_fn_shapes(tmp_path):
    paths = []
    for i in range(2):
        p = tmp_path / f"s1_{i}.npz"
        _write_npz(p, label="hello")
        paths.append(p)

    ds = GridLipReadingDataset(paths)
    batch = [ds[0], ds[1]]
    out = grid_collate_fn(batch)

    assert out["frames"].shape == (2, 75, 46, 140)
    assert out["targets"].shape == (10,)             # 5 chars × 2 samples
    assert out["target_lengths"].tolist() == [5, 5]
    assert out["input_lengths"].tolist() == [75, 75]
    assert len(out["texts"]) == 2
    assert len(out["paths"]) == 2
```

- [ ] **Step 3: Run test, verify failure**

```bash
pytest tests/test_datasets.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Implement `src/lsn/data/datasets.py`**

Extract from notebook:

```bash
python tools/dump_cell.py 17 > /tmp/cell17.py   # GridLipReadingDataset (use the second class in this cell — the canonical one with FRAME_KEY/LABEL_KEY)
python tools/dump_cell.py 19 > /tmp/cell19.py   # grid_collate_fn
python tools/dump_cell.py LRS2_DATASET_CELL > /tmp/lrs2.py
```

Create `src/lsn/data/datasets.py`:
- Module docstring referencing spec §11 (.npz contract: shape `(75, 46, 140)` or `(75, 46, 140, 1)`, keys `frames` and `label`)
- Imports: `Path`, `numpy as np`, `torch`, `Dataset` from `torch.utils.data`, `from lsn.data.vocab import encode_text`
- The canonical `GridLipReadingDataset` class from cell 17 (the second one — the one with `FRAME_KEY` / `LABEL_KEY` constants). Drop the earlier `GridLipReadingDatasetParquet` — unused.
- **Add shape validation** to `__getitem__` per spec §9.6: after squeezing the trailing channel dim, assert `frames.shape == (75, 46, 140)` and raise `ValueError(f"unexpected frames shape {frames.shape} in {path}; see docs/data-format.md")` if not.
- The `grid_collate_fn` from cell 19 verbatim
- `LRS2Dataset` from the LRS2 cell (renamed from `LRS2LipReadingDataset` per spec §12 last bullet)

- [ ] **Step 5: Run test, verify pass**

```bash
pytest tests/test_datasets.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/lsn/data/datasets.py tests/test_datasets.py
git commit -m "feat(data): GridLipReadingDataset, LRS2Dataset, grid_collate_fn

Refactored from notebook cells 17, 19, and the LRS2 stage. Renamed
LRS2LipReadingDataset → LRS2Dataset (spec §12). Added shape validation
(spec §9.6)."
```

---

## Task 11: `lsn/data/splits.py` — `create_paper_split`

**Files:**
- Create: `src/lsn/data/splits.py`
- Create: `tests/test_splits.py`

**Source notebook reference:** code cell 21 (`create_paper_split`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_splits.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from lsn.data.splits import create_paper_split


def _make_synthetic_dataset(tmp_path: Path, speakers: list[str], n_per: int = 200):
    """Build a fake speaker_dir/*.npz layout."""
    paths = []
    for sid in speakers:
        d = tmp_path / sid
        d.mkdir(exist_ok=True)
        for i in range(n_per):
            p = d / f"{sid}_{i:04d}.npz"
            np.savez(p, frames=np.zeros((75, 46, 140), dtype=np.float32),
                     label="hi")
            paths.append(p)
    return paths


def test_paper_split_sizes(tmp_path):
    paths = _make_synthetic_dataset(
        tmp_path, ["s1", "s2", "s3", "s4", "s5"], n_per=200,
    )
    train, test = create_paper_split(
        paths, speakers=["s1", "s2", "s3", "s4", "s5"],
        samples_per_speaker=200, train_size=450, seed=42,
    )
    assert len(train) == 450
    assert len(test) == 550


def test_paper_split_deterministic(tmp_path):
    paths = _make_synthetic_dataset(
        tmp_path, ["s1", "s2", "s3", "s4", "s5"], n_per=200,
    )
    train1, test1 = create_paper_split(
        paths, speakers=["s1", "s2", "s3", "s4", "s5"],
        samples_per_speaker=200, train_size=450, seed=42,
    )
    train2, test2 = create_paper_split(
        paths, speakers=["s1", "s2", "s3", "s4", "s5"],
        samples_per_speaker=200, train_size=450, seed=42,
    )
    assert [str(p) for p in train1] == [str(p) for p in train2]
    assert [str(p) for p in test1] == [str(p) for p in test2]


def test_paper_split_disjoint(tmp_path):
    paths = _make_synthetic_dataset(
        tmp_path, ["s1", "s2", "s3", "s4", "s5"], n_per=200,
    )
    train, test = create_paper_split(
        paths, speakers=["s1", "s2", "s3", "s4", "s5"],
        samples_per_speaker=200, train_size=450, seed=42,
    )
    assert set(map(str, train)).isdisjoint(set(map(str, test)))
```

- [ ] **Step 2: Run test, verify failure**

```bash
pytest tests/test_splits.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/lsn/data/splits.py`**

Extract from notebook cell 21 (the function `create_paper_split`, NOT the surrounding `train_paths, val_paths = ...` invocation). Drop the per-speaker print statements; replace with a single `logger.info` summary at the end:

```python
"""Paper-subset 450/550 split with balanced speaker representation.

Refactored from notebook cell 21.
"""
from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def create_paper_split(npz_paths: list[Path],
                       speakers: list[str] | None = None,
                       samples_per_speaker: int = 200,
                       train_size: int = 450,
                       seed: int = 42) -> tuple[list[Path], list[Path]]:
    """Reproduce the paper's 1,000-sample / 450-550 split with balanced
    speaker representation. Per speaker: 90 train, 110 test (= 450/5, 550/5).
    """
    speaker_files: dict[str, list[Path]] = defaultdict(list)
    for p in npz_paths:
        sid = p.parent.name
        if speakers and sid not in speakers:
            continue
        speaker_files[sid].append(p)

    if speakers is None:
        speakers = sorted(speaker_files.keys(), key=lambda x: int(x[1:]))

    n_speakers = len(speakers)
    train_per_speaker = train_size // n_speakers
    test_per_speaker = samples_per_speaker - train_per_speaker

    rng = random.Random(seed)
    train_paths: list[Path] = []
    test_paths: list[Path] = []

    for sid in sorted(speakers, key=lambda x: int(x[1:])):
        files = sorted(speaker_files[sid])
        assert len(files) >= samples_per_speaker, \
            f"{sid} has {len(files)} files, need {samples_per_speaker}"

        sampled = rng.sample(files, samples_per_speaker)
        rng.shuffle(sampled)

        train_paths.extend(sampled[:train_per_speaker])
        test_paths.extend(sampled[train_per_speaker:])

    rng.shuffle(train_paths)
    rng.shuffle(test_paths)

    logger.info(
        "split: %d train / %d test across %d speakers (%d/%d per speaker)",
        len(train_paths), len(test_paths), n_speakers,
        train_per_speaker, test_per_speaker,
    )

    return train_paths, test_paths
```

- [ ] **Step 4: Run test, verify pass**

```bash
pytest tests/test_splits.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/lsn/data/splits.py tests/test_splits.py
git commit -m "feat(data): create_paper_split

Refactored from notebook cell 21. Per-speaker prints downgraded to one
logger.info summary (spec §12)."
```

---

## Task 12: `lsn/data/normalize.py` — LRS2 text normalization

**Files:**
- Create: `src/lsn/data/normalize.py`
- Create: `tests/test_normalize.py`

**Source notebook reference:** Stage H markdown describes "LRS2-specific text-normalization function." Find the corresponding code cell.

- [ ] **Step 1: Locate the LRS2 normalize function and dump it for verbatim paste**

```bash
python -c "
import json
nb = json.load(open('notebooks/legacy/LSN_TRAINING_EVAL.ipynb', encoding='utf-8'))
codes = [c for c in nb['cells'] if c['cell_type']=='code']
for i, c in enumerate(codes):
    src = ''.join(c['source'])
    if 'def normalize' in src.lower() or ('lrs2' in src.lower() and 'normaliz' in src.lower()):
        print(f'cell {i}:'); print(src); print('---')
"
```

Note the cell index — call it `LRS2_NORM_CELL`. Dump it:

```bash
python tools/dump_cell.py LRS2_NORM_CELL
```

**Read the actual function body** before writing tests. LRS2 normalization typically does some combination of digit-to-word ("7" → "seven"), apostrophe/punctuation handling, and case folding. **The exact rules are load-bearing for metric correctness** on LRS2 — guessing silently corrupts the Table 5 LRS2 row. Do NOT proceed with placeholder tests.

If the function name in the notebook differs (e.g., `normalize_text_lrs2`), preserve its body verbatim under the new name `normalize_lrs2`.

- [ ] **Step 2: Write the failing tests — base set + per-rule assertions from the cell body**

Create `tests/test_normalize.py`:

```python
from lsn.data.normalize import normalize_lrs2


def test_normalize_returns_str():
    assert isinstance(normalize_lrs2("hello world"), str)


def test_normalize_invariant_on_clean_input():
    """Already-lowercase, no-punct, no-digit input round-trips unchanged."""
    s = "hello world"
    assert normalize_lrs2(s) == s


def test_normalize_lowercases():
    assert normalize_lrs2("HELLO") == normalize_lrs2("hello")


# IMPORTANT: After dumping the source cell in Step 1, ADD per-rule assertions
# here that match the actual transforms. Examples (uncomment + adapt):
#
# def test_normalize_strips_apostrophes():
#     assert normalize_lrs2("don't") == "dont"
#
# def test_normalize_digit_to_word():
#     assert normalize_lrs2("we have 7 cats") == "we have seven cats"
#
# def test_normalize_strips_punctuation():
#     assert normalize_lrs2("hello, world!") == "hello world"
#
# Each rule the actual notebook applies needs a corresponding assertion here.
# The base tests above are not sufficient — they would pass on a stub.
```

**Do not skip the per-rule assertions.** They are the only protection against silent normalization regressions in LRS2 metrics.

- [ ] **Step 3: Run test, verify failure**

Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Implement `src/lsn/data/normalize.py`**

Paste the function body from the located cell **verbatim** into the module skeleton below. Do not paraphrase, refactor, or "improve" the rules — they're load-bearing for metric correctness.

```python
"""LRS2-specific text normalization.

Applied at metric-compute time inside lsn.evaluation.report (spec §7),
not at JSON-write time. JSON predictions contain raw decoder output.
"""
from __future__ import annotations


def normalize_lrs2(text: str) -> str:
    """Refactored verbatim from notebook Stage H.

    <After pasting: replace this line with a one-line summary of what the
    function does — e.g., 'Lowercase + strip punctuation + digit-to-word'.>
    """
    # PASTE the cell body here verbatim. If the original used helper
    # constants (regex tables, digit-word maps), paste those at module
    # scope above this function.
    raise NotImplementedError("paste cell body from Step 1 dump")
```

The `NotImplementedError` is a deliberate trip-wire — Step 5 will fail until the body is pasted.

- [ ] **Step 5: Run test, verify pass**

```bash
pytest tests/test_normalize.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/lsn/data/normalize.py tests/test_normalize.py
git commit -m "feat(data): normalize_lrs2

Refactored from LRS2 Stage H of notebook (spec §4.4)."
```

---

## Task 13: `lsn/training/hf_store.py` — HFStore class

**Files:**
- Create: `src/lsn/training/hf_store.py`
- Create: `tests/test_hf_store.py`

**Source notebook reference:** parts of cell 22 (HF upload / hf_hub_download usage).

- [ ] **Step 1: Write the failing test**

Create `tests/test_hf_store.py`:

```python
import os

import pytest

from lsn.training.hf_store import HFStore


def test_hf_store_uses_token_arg():
    """When token is passed explicitly, it's used."""
    s = HFStore(repo="x/y", subfolder="run_v1", token="explicit-tok")
    assert s.token == "explicit-tok"


def test_hf_store_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "env-tok")
    s = HFStore(repo="x/y", subfolder="run_v1", token=None)
    assert s.token == "env-tok"


def test_hf_store_no_token_no_env(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    s = HFStore(repo="x/y", subfolder="run_v1", token=None)
    assert s.token is None
```

(Network-touching `upload`/`try_download` are not unit-tested here — covered by the manual end-to-end run in Task 28.)

- [ ] **Step 2: Run test, verify failure**

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/lsn/training/hf_store.py`**

```python
"""HuggingFace Hub adapter — the ONLY module that imports huggingface_hub.

Spec §9.3: gated by config.checkpointing.hf_repo. When unset, the runner
passes None everywhere and zero HF code paths are exercised.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

logger = logging.getLogger(__name__)


class HFStore:
    """Thin wrapper around HfApi for upload/download of checkpoints.

    `token=None` reads the HF_TOKEN env var. The README documents how each
    environment (Colab/Kaggle/local) populates HF_TOKEN.
    """

    def __init__(self, repo: str, subfolder: str, token: str | None = None):
        self.repo = repo
        self.subfolder = subfolder
        self.token = token if token is not None else os.environ.get("HF_TOKEN")
        self._api: HfApi | None = None

    @property
    def api(self) -> HfApi:
        if self._api is None:
            self._api = HfApi(token=self.token)
        return self._api

    def upload(self, local_path: Path, remote_filename: str, *,
               commit_message: str) -> bool:
        """Upload `local_path` to `<subfolder>/<remote_filename>`. Returns True
        on success; logs a warning and returns False on failure (never raises —
        HF hiccups must not kill training)."""
        try:
            self.api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=f"{self.subfolder}/{remote_filename}",
                repo_id=self.repo,
                repo_type="model",
                commit_message=commit_message,
            )
            return True
        except Exception as e:
            logger.warning("HF upload failed: %s", e)
            return False

    def try_download(self, remote_filename: str, local_dest: Path) -> Path | None:
        """Download `<subfolder>/<remote_filename>` into `local_dest` directory.
        Returns the local Path on success; None if the file isn't on the Hub
        (clean miss, not an error)."""
        try:
            local_dest.mkdir(parents=True, exist_ok=True)
            path = hf_hub_download(
                repo_id=self.repo,
                filename=f"{self.subfolder}/{remote_filename}",
                token=self.token,
                local_dir=str(local_dest),
            )
            return Path(path)
        except (EntryNotFoundError, RepositoryNotFoundError):
            return None
        except Exception as e:
            logger.info("HF download error for %s: %s", remote_filename, e)
            return None
```

- [ ] **Step 4: Run test, verify pass**

```bash
pytest tests/test_hf_store.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/lsn/training/hf_store.py tests/test_hf_store.py
git commit -m "feat(training): HFStore — sole importer of huggingface_hub (spec §9.3)"
```

---

## Task 14: `lsn/training/checkpoint.py` — save, resume, freeze_bn_stats

**Files:**
- Create: `src/lsn/training/checkpoint.py`
- Create: `tests/test_checkpoint.py`

**Source notebook reference:** code cell 22 (the bulk of the checkpointing infra: `freeze_bn_stats`, `_model_state_dict`, `_load_into_model`, `save_checkpoint_safe`, `try_resume`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_checkpoint.py`:

```python
from pathlib import Path

import torch
import torch.nn as nn

from lsn.training.checkpoint import (
    LAST_CKPT_NAME, BEST_CKPT_NAME,
    save_checkpoint_safe, try_resume, freeze_bn_stats,
)


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
    scaler = torch.amp.GradScaler("cpu", enabled=False)

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
    scaler2 = torch.amp.GradScaler("cpu", enabled=False)
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
    scaler = torch.amp.GradScaler("cpu", enabled=False)
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
```

- [ ] **Step 2: Run test, verify failure**

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/lsn/training/checkpoint.py`**

Extract from cell 22:

```bash
python tools/dump_cell.py 22 > /tmp/cell22.py
```

Cell 22 contains both checkpointing logic AND the HF-Hub-using `save_checkpoint_safe` / `try_resume` bodies. Refactor:
- HF-related code goes through the `remote: HFStore | None` parameter (already extracted to `hf_store.py` in Task 13). Replace direct `hf_api.upload_file` / `hf_hub_download` calls with `remote.upload(...)` / `remote.try_download(...)`.
- Add module-level constants `LAST_CKPT_NAME = "last_checkpoint.pt"`, `BEST_CKPT_NAME = "best_model.pt"` (spec §4.5).
- `try_resume` uses the priority order in spec §4.5: HF last → HF best → local last → local best → fresh.
- All `torch.load` calls use `weights_only=False` (spec §14 risk).
- `freeze_bn_stats` returns `int` (count of BN modules frozen) per spec §4.5 docstring.
- Drop the print statements — replace with `logger.info` for resume confirmation, `logger.warning` for HF errors.

Key API:

```python
LAST_CKPT_NAME = "last_checkpoint.pt"
BEST_CKPT_NAME = "best_model.pt"

def save_checkpoint_safe(model, optimizer, scaler, *,
                         epoch, train_loss, val_loss, best_val_loss, history,
                         save_path, remote=None) -> None: ...

def try_resume(model, optimizer, scaler, device,
               local_dir, remote) -> tuple[int, float, list]: ...

def freeze_bn_stats(model) -> int: ...

def _model_state_dict(model): ...   # DataParallel unwrap
def _load_into_model(model, state): ...
```

- [ ] **Step 4: Run test, verify pass**

```bash
pytest tests/test_checkpoint.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/lsn/training/checkpoint.py tests/test_checkpoint.py
git commit -m "feat(training): save_checkpoint_safe, try_resume, freeze_bn_stats

Refactored from notebook cell 22. HF-side ops dispatched via HFStore;
torch.load uses weights_only=False (spec §14)."
```

---

## Task 15: `lsn/training/loop.py` — `train_one_epoch`, `validate_one_epoch`

**Files:**
- Create: `src/lsn/training/loop.py`
- Create: `tests/test_loop.py`

**Source notebook reference:** code cell 23 (`train_one_epoch`, `validate_one_epoch`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_loop.py` (smoke test only — full training requires real data):

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lsn.training.loop import validate_one_epoch


def test_validate_one_epoch_returns_dict_with_loss():
    """Smoke test: validate runs end-to-end on dummy data and returns shape."""
    class TinyCTC(nn.Module):
        def forward(self, x):
            B, T, _, _ = x.shape
            # log_probs shape (T, B, C=41) — matches LipSyncNet output
            return torch.log_softmax(torch.randn(T, B, 41), dim=-1)

    model = TinyCTC()

    # Build a tiny batch matching grid_collate_fn output
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
```

- [ ] **Step 2: Run test, verify failure**

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/lsn/training/loop.py`**

```bash
python tools/dump_cell.py 23 > /tmp/cell23.py
```

Copy `train_one_epoch` and `validate_one_epoch` verbatim. Imports: `time`, `torch`, `from torch.amp import autocast`, `from tqdm import tqdm`, `from lsn.training.checkpoint import freeze_bn_stats`. Module docstring summarizes the AMP-fp16-cast-to-fp32-for-CTC trick (spec §3 / notebook cell 23 docstring).

- [ ] **Step 4: Run test, verify pass**

```bash
pytest tests/test_loop.py -v
```

Expected: 1 passed (validate-only smoke; training-loop full check happens in end-to-end Task 28).

- [ ] **Step 5: Commit**

```bash
git add src/lsn/training/loop.py tests/test_loop.py
git commit -m "feat(training): train_one_epoch + validate_one_epoch

Refactored from notebook cell 23. AMP forward + fp32 CTC preserved."
```

---

## Task 16: `lsn/training/runner.py` — high-level `run()`

**Files:**
- Create: `src/lsn/training/runner.py`
- (No unit test — `run()` is integration-tested via Task 28's end-to-end smoke run)

**Source notebook reference:** code cells 24, 25 (DataLoader setup + main training loop).

- [ ] **Step 1: Implement `src/lsn/training/runner.py`**

```python
"""High-level training runner — wires Config → model + data + loop + checkpoint.

Refactored from notebook cells 24-25. Called by scripts/train.py.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from lsn.config import Config
from lsn.data.datasets import GridLipReadingDataset, grid_collate_fn
from lsn.data.splits import create_paper_split
from lsn.data.vocab import BLANK_INDEX
from lsn.env import configure_cudnn, set_seed
from lsn.models import build_from_config, count_parameters
from lsn.training.checkpoint import (
    BEST_CKPT_NAME, LAST_CKPT_NAME,
    save_checkpoint_safe, try_resume,
)
from lsn.training.hf_store import HFStore
from lsn.training.loop import train_one_epoch, validate_one_epoch

logger = logging.getLogger(__name__)


def run(cfg: Config, *, data_dir: Path, ckpt_dir: Path,
        device: torch.device) -> None:
    """End-to-end training run from a Config.

    Builds the per-experiment subdirectory `ckpt_dir / cfg.experiment_name`
    and passes it to checkpoint.save_*/try_resume. Caller (scripts/train.py)
    supplies the parent ckpt_dir; runner.run owns per-experiment naming.
    """
    set_seed(cfg.data.seed)
    configure_cudnn(benchmark=True)

    # Per-experiment ckpt dir
    run_ckpt_dir = ckpt_dir / cfg.experiment_name
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # HF gating
    remote: HFStore | None = None
    if cfg.checkpointing.hf_repo:
        subfolder = cfg.checkpointing.hf_subfolder or cfg.experiment_name
        remote = HFStore(cfg.checkpointing.hf_repo, subfolder)
        logger.info("HF enabled: repo=%s subfolder=%s",
                    cfg.checkpointing.hf_repo, subfolder)
    else:
        logger.info("HF disabled — local-only checkpointing")

    # Dataset + split
    npz_paths = sorted(Path(data_dir).glob("*/*.npz"))
    if not npz_paths:
        raise FileNotFoundError(
            f"No .npz files found in {data_dir} (expected speaker-subdir layout)"
        )
    train_paths, val_paths = create_paper_split(
        npz_paths,
        speakers=cfg.data.speakers,
        samples_per_speaker=cfg.data.samples_per_speaker,
        train_size=cfg.data.train_size,
        seed=cfg.data.seed,
    )

    train_loader = DataLoader(
        GridLipReadingDataset(train_paths),
        batch_size=cfg.training.batch_size, shuffle=True,
        collate_fn=grid_collate_fn,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=cfg.training.num_workers > 0,
        prefetch_factor=cfg.training.prefetch if cfg.training.num_workers > 0 else None,
        drop_last=False,
    )
    val_loader = DataLoader(
        GridLipReadingDataset(val_paths),
        batch_size=cfg.training.batch_size, shuffle=False,
        collate_fn=grid_collate_fn,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=cfg.training.num_workers > 0,
        prefetch_factor=cfg.training.prefetch if cfg.training.num_workers > 0 else None,
        drop_last=False,
    )
    logger.info("train=%d batches=%d  val=%d batches=%d  effective_batch=%d",
                len(train_loader.dataset), len(train_loader),
                len(val_loader.dataset), len(val_loader),
                cfg.training.batch_size * cfg.training.accum_steps)

    # Model
    model = build_from_config(cfg.model, device=device)
    counts = count_parameters(model)
    logger.info("params: total=%(total)d trainable=%(trainable)d frozen=%(frozen)d",
                counts)

    # Loss / optim / scaler
    loss_fn = nn.CTCLoss(blank=BLANK_INDEX, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.learning_rate,
    )
    scaler = GradScaler("cuda", enabled=cfg.training.use_amp and device.type == "cuda")

    # Resume
    start_epoch, best_val_loss, history = try_resume(
        model, optimizer, scaler, device, run_ckpt_dir, remote,
    )

    # Training loop
    for epoch in range(start_epoch + 1, cfg.training.num_epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            loss_fn=loss_fn, device=device, scaler=scaler,
            max_grad_norm=cfg.training.max_grad_norm,
            accum_steps=cfg.training.accum_steps,
            use_amp=cfg.training.use_amp,
        )
        val_stats = validate_one_epoch(
            model=model, loader=val_loader, loss_fn=loss_fn,
            device=device, use_amp=cfg.training.use_amp,
        )

        train_loss = train_stats["loss"]
        val_loss = val_stats["loss"]

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "grad_norm": train_stats["grad_norm"],
            "train_time": train_stats["time_sec"],
            "val_time": val_stats["time_sec"],
        })

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss

        save_checkpoint_safe(
            model, optimizer, scaler,
            epoch=epoch, train_loss=train_loss, val_loss=val_loss,
            best_val_loss=best_val_loss, history=history,
            save_path=run_ckpt_dir / LAST_CKPT_NAME, remote=remote,
        )
        if improved:
            save_checkpoint_safe(
                model, optimizer, scaler,
                epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                best_val_loss=best_val_loss, history=history,
                save_path=run_ckpt_dir / BEST_CKPT_NAME, remote=remote,
            )

        logger.info(
            "epoch %d/%d | train=%.4f val=%.4f | gnorm=%.2f | "
            "train_t=%.1fs val_t=%.1fs | best=%.4f%s",
            epoch, cfg.training.num_epochs, train_loss, val_loss,
            train_stats["grad_norm"], train_stats["time_sec"],
            val_stats["time_sec"], best_val_loss, "  ↓" if improved else "",
        )
```

- [ ] **Step 2: Verify imports work**

```bash
python -c "from lsn.training.runner import run; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/lsn/training/runner.py
git commit -m "feat(training): runner.run — wires config → model + data + loop + ckpt

Refactored from notebook cells 24-25 (DataLoader setup + main training loop)."
```

---

## Task 17: `lsn/evaluation/decoders.py` — greedy + beam=100

**Files:**
- Create: `src/lsn/evaluation/decoders.py`
- Create: `tests/test_decoders.py`

**Source notebook reference:** Stage C cells (greedy + beam decoder definitions).

> **Critical for this task:** the beam decoder's exact `ctc_decoder(...)` arguments — `tokens` list, `blank_token`, `sil_token`, `lexicon` — are load-bearing. Wrong values silently corrupt Table 5 numbers. The Step 1 locator is required, not optional, and the implementation must paste the notebook's exact arguments rather than reconstructing them.

- [ ] **Step 0 (REQUIRED): Locate and dump the Stage C decoder cells**

```bash
python -c "
import json
nb = json.load(open('notebooks/legacy/LSN_TRAINING_EVAL.ipynb', encoding='utf-8'))
codes = [c for c in nb['cells'] if c['cell_type']=='code']
for i, c in enumerate(codes):
    s = ''.join(c['source'])
    if ('def greedy' in s.lower()
        or ('beam' in s.lower() and 'decode' in s.lower())
        or 'ctc_decoder' in s.lower()):
        print(f'=== cell {i} ==='); print(s); print()
"
```

Note all matching cell indices (likely 2–3 cells in Stage C). Dump each:

```bash
python tools/dump_cell.py <N1>
python tools/dump_cell.py <N2>
```

**Read every line.** Specifically capture:
- The exact `tokens` argument passed to `ctc_decoder` (typically `["-"] + CHARS` or `["<blank>"] + CHARS` — the notebook's choice is canonical, not the plan's guess)
- The `blank_token` and `sil_token` strings
- Whether `lexicon` is `None` or a path
- Any post-processing applied to `Hypothesis.tokens` (does the notebook strip blanks again? join with space? something custom?)
- The exact greedy decoder loop (how it collapses repeats, when it drops blanks)

- [ ] **Step 1: Write the failing test**

Create `tests/test_decoders.py`:

```python
import torch

from lsn.evaluation.decoders import beam_decode, greedy_decode
from lsn.data.vocab import BLANK_INDEX, char_to_idx


def test_greedy_decode_shapes_and_types():
    T, B, C = 75, 2, 41
    log_probs = torch.log_softmax(torch.randn(T, B, C), dim=-1)
    out = greedy_decode(log_probs)
    assert isinstance(out, list)
    assert len(out) == B
    assert all(isinstance(s, str) for s in out)


def test_greedy_decode_deterministic_input():
    """Hand-construct an argmax sequence and verify the decoded string.

    Sequence (T=6): [a, a, blank, b, b, c]
    CTC collapse-then-drop-blanks → "abc"
    """
    a = char_to_idx["a"]
    b = char_to_idx["b"]
    c = char_to_idx["c"]
    seq = [a, a, BLANK_INDEX, b, b, c]
    T, B, C = len(seq), 1, 41
    log_probs = torch.full((T, B, C), -10.0)
    for t, idx in enumerate(seq):
        log_probs[t, 0, idx] = 0.0
    log_probs = torch.log_softmax(log_probs, dim=-1)

    decoded = greedy_decode(log_probs)
    assert decoded == ["abc"], f"expected ['abc'], got {decoded!r}"


def test_beam_decode_requires_input_lengths():
    """beam_decode signature includes input_lengths (spec §4.6, B3 fix)."""
    T, B, C = 75, 2, 41
    log_probs = torch.log_softmax(torch.randn(T, B, C), dim=-1)
    input_lengths = torch.tensor([T, T], dtype=torch.long)
    out = beam_decode(log_probs, input_lengths=input_lengths, beam_width=10)
    assert isinstance(out, list)
    assert len(out) == B
    assert all(isinstance(s, str) for s in out)


def test_beam_decode_deterministic_input():
    """A peaked argmax sequence should produce the same string under greedy
    and beam decoding (when the hypothesis is unambiguous)."""
    a = char_to_idx["a"]
    seq = [a, a, BLANK_INDEX, a]
    T, B, C = len(seq), 1, 41
    log_probs = torch.full((T, B, C), -10.0)
    for t, idx in enumerate(seq):
        log_probs[t, 0, idx] = 0.0
    log_probs = torch.log_softmax(log_probs, dim=-1)
    input_lengths = torch.tensor([T], dtype=torch.long)

    g = greedy_decode(log_probs)
    b = beam_decode(log_probs, input_lengths=input_lengths, beam_width=10)
    assert g == b == ["aa"], f"greedy={g}, beam={b}"
```

The deterministic-input tests above catch wrong tokens / blank-symbol bugs.
A stub that returns shape-correct gibberish would pass `test_greedy_decode_shapes_and_types` but fail `test_greedy_decode_deterministic_input`.

- [ ] **Step 2: Run test, verify failure**

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/lsn/evaluation/decoders.py` — paste from Step 0 dumps**

Use the dumps from Step 0 as the canonical source. The skeleton below shows the module structure and the key signature (B3 fix: `input_lengths` is required). The bodies marked `<paste from cell N>` MUST come from the dumped cells, not from the plan's guess at the API.

```python
"""CTC decoders — greedy + beam=100 (paper setting).

Refactored from notebook Stage C cells (see Step 0 dumps).
beam_decode requires input_lengths (torchaudio API, spec §4.6).
"""
from __future__ import annotations

import torch
from torch import Tensor
from torchaudio.models.decoder import ctc_decoder

from lsn.data.vocab import BLANK_INDEX, CHARS, decode_ids


def greedy_decode(log_probs: Tensor) -> list[str]:
    """Greedy CTC decoding. Input shape (T, B, C); returns list of B strings.

    <Paste body from notebook greedy-decode cell. Must use BLANK_INDEX from
    vocab (not a hardcoded 0) and must collapse repeats before dropping
    blanks — the test_greedy_decode_deterministic_input test enforces this.>
    """
    raise NotImplementedError("paste body from Step 0 dump")


_BEAM_DECODER_CACHE: dict[int, object] = {}


def _get_beam_decoder(beam_width: int):
    """Lazy-construct + cache the torchaudio decoder by beam width.

    <Paste the EXACT ctc_decoder(...) call from the notebook here. The
    `tokens`, `blank_token`, `sil_token`, and `lexicon` arguments are
    load-bearing — copy them verbatim. DO NOT pass a tokens FILE; pass the
    in-memory list (spec §9.3).>
    """
    if beam_width not in _BEAM_DECODER_CACHE:
        # tokens list: paste from notebook (likely ["-"] + CHARS or similar)
        # Replace the line below with the exact call from the dump:
        raise NotImplementedError("paste ctc_decoder(...) call from Step 0 dump")
    return _BEAM_DECODER_CACHE[beam_width]


def beam_decode(log_probs: Tensor, input_lengths: Tensor,
                beam_width: int = 100) -> list[str]:
    """Beam-search CTC decode.

    log_probs: (T, B, C) — same convention as model.forward output.
    input_lengths: (B,) long — actual T per sample (75 for fixed-T GRID).

    <Paste body from notebook beam-decode cell. Pay attention to:
    - Whether log_probs needs permute(1, 0, 2) before passing to the decoder
      (torchaudio expects (B, T, C)).
    - Whether tokens are joined with "" or " " in the post-processing.
    - Whether the notebook calls .cpu() before the decoder.>
    """
    raise NotImplementedError("paste body from Step 0 dump")
```

The `NotImplementedError`s are deliberate trip-wires. Step 4 will fail loudly until each is replaced with the actual notebook code.

- [ ] **Step 4: Run test, verify pass**

```bash
pytest tests/test_decoders.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/lsn/evaluation/decoders.py tests/test_decoders.py
git commit -m "feat(evaluation): greedy + beam=100 CTC decoders (spec §4.6)"
```

---

## Task 18: `lsn/evaluation/metrics.py` — edit_distance, CER, WER, accuracies

**Files:**
- Create: `src/lsn/evaluation/metrics.py`
- Create: `tests/test_metrics.py`

**Source notebook reference:** Stage E cells.

- [ ] **Step 1: Write the failing test**

Create `tests/test_metrics.py`:

```python
from lsn.evaluation.metrics import (
    edit_distance, cer, wer, word_acc, sentence_acc,
)


def test_edit_distance_identical():
    assert edit_distance(list("abc"), list("abc")) == 0


def test_edit_distance_single_sub():
    assert edit_distance(list("abc"), list("abd")) == 1


def test_edit_distance_insertion_deletion():
    assert edit_distance(list("ab"), list("abc")) == 1
    assert edit_distance(list("abc"), list("ab")) == 1


def test_cer_zero_for_identical():
    refs = ["hello", "world"]
    hyps = ["hello", "world"]
    assert cer(refs, hyps) == 0.0


def test_wer_zero_for_identical():
    refs = ["hello world"]
    hyps = ["hello world"]
    assert wer(refs, hyps) == 0.0


def test_word_acc_inverts_wer():
    refs = ["a b c"]
    hyps = ["a b c"]
    assert word_acc(refs, hyps) == 1.0


def test_sentence_acc_exact_match():
    refs = ["foo", "bar", "baz"]
    hyps = ["foo", "BAR", "baz"]   # one mismatch
    # implementation should compare lowercased/stripped — verify per notebook
    acc = sentence_acc(refs, hyps)
    assert 0.0 <= acc <= 1.0
```

- [ ] **Step 2: Run test, verify failure**

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/lsn/evaluation/metrics.py`**

Copy `edit_distance` from Stage E. Implement CER/WER/word_acc/sentence_acc using it:

```python
"""CTC metrics — refactored from notebook Stage E."""
from __future__ import annotations


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein distance over the two token sequences."""
    # <copy from notebook Stage E>
    ...


def cer(refs: list[str], hyps: list[str]) -> float:
    total_d, total_n = 0, 0
    for r, h in zip(refs, hyps):
        total_d += edit_distance(list(r), list(h))
        total_n += len(r)
    return total_d / max(total_n, 1)


def wer(refs: list[str], hyps: list[str]) -> float:
    total_d, total_n = 0, 0
    for r, h in zip(refs, hyps):
        total_d += edit_distance(r.split(), h.split())
        total_n += len(r.split())
    return total_d / max(total_n, 1)


def word_acc(refs: list[str], hyps: list[str]) -> float:
    """1 - WER (paper convention)."""
    return 1.0 - wer(refs, hyps)


def sentence_acc(refs: list[str], hyps: list[str]) -> float:
    """Exact-match rate over (ref, hyp) pairs after strip+lower normalize."""
    n_match = sum(1 for r, h in zip(refs, hyps)
                  if r.strip().lower() == h.strip().lower())
    return n_match / max(len(refs), 1)
```

- [ ] **Step 4: Run test, verify pass**

```bash
pytest tests/test_metrics.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/lsn/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat(evaluation): edit_distance, CER, WER, word_acc, sentence_acc"
```

---

## Task 19: `lsn/evaluation/inference.py` — `run_inference` + `Prediction`

**Files:**
- Create: `src/lsn/evaluation/inference.py`
- Create: `tests/test_inference.py`

**Source notebook reference:** Stage D cells (the `run_inference` function or its inline equivalent).

- [ ] **Step 1: Write the failing test**

Create `tests/test_inference.py`:

```python
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
```

- [ ] **Step 2: Run test, verify failure**

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/lsn/evaluation/inference.py`**

```python
"""Run a model over a DataLoader, decode each batch, return list[Prediction].

Refactored from notebook Stage D. The output is the input to write_eval_json
(spec §7).
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
```

- [ ] **Step 4: Run test, verify pass**

```bash
pytest tests/test_inference.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/lsn/evaluation/inference.py tests/test_inference.py
git commit -m "feat(evaluation): run_inference + Prediction (spec §4.6)"
```

---

## Task 20: `lsn/evaluation/report.py` — JSON write, plots, tables

**Files:**
- Create: `src/lsn/evaluation/report.py`
- Create: `tests/test_report.py`

**Source notebook reference:** Stage A (learning curves), Stage E (metrics tables, qualitative examples).

- [ ] **Step 1: Write the failing test**

Create `tests/test_report.py`:

```python
import json
from pathlib import Path

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
```

- [ ] **Step 2: Run test, verify failure**

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/lsn/evaluation/report.py`**

```python
"""JSON write + plots + tables — pure consumer of <experiment>_<dataset>_eval.json.

The single non-JSON side-channel: PAPER_BASELINES_GRID is a hardcoded
constant (spec §7).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from lsn.data.normalize import normalize_lrs2
from lsn.evaluation.inference import Prediction
from lsn.evaluation.metrics import cer, sentence_acc, wer, word_acc


# Spec §7: Table 5 paper-baseline rows. Provenance: notebook Stage E text.
PAPER_BASELINES_GRID: list[dict] = [
    {"model": "Xu et al. [29]",         "dataset": "GRID",
     "method": "Cascaded Attention-CTC", "word_acc": 0.896, "wer": None},
    {"model": "Gergen et al. [31]",      "dataset": "GRID",
     "method": "—",                      "word_acc": 0.864, "wer": None},
    {"model": "Margam et al. [30]",      "dataset": "GRID",
     "method": "3D-2D-CNN BLSTM-HMM",    "word_acc": 0.914, "wer": None},
    {"model": "LipSyncNet (paper-reported)", "dataset": "GRID",
     "method": "—",                      "word_acc": 0.967, "wer": 0.082},
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def _load_eval_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def plot_learning_curves(eval_jsons: list[Path], out_dir: Path) -> None:
    """Per-model curves + comparison plot. Refactored from notebook Stage A."""
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
        ax.set_xlabel("epoch"); ax.set_ylabel("CTC loss")
        ax.set_title(f"{d['display_name']} — train vs val loss")
        ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
        plt.tight_layout()
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
    ax.set_xlabel("epoch"); ax.set_ylabel("validation CTC loss")
    ax.set_title("Validation loss — model comparison")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
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
    by_dataset = {}
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
```

- [ ] **Step 4: Run test, verify pass**

```bash
pytest tests/test_report.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/lsn/evaluation/report.py tests/test_report.py
git commit -m "feat(evaluation): write_eval_json, plots, results+qualitative tables

Refactored from notebook Stages A and E. PAPER_BASELINES_GRID is the only
non-JSON side-channel (spec §7)."
```

---

## Task 21: `scripts/train.py` — argparse + runner.run wrapper

**Files:**
- Create: `scripts/train.py`

**No unit test** — it's a thin argparse wrapper. Manual run in Task 28 verifies.

- [ ] **Step 1: Implement `scripts/train.py`**

```python
"""Entry point for training. Usage:

    python scripts/train.py --config configs/identity.yaml \\
        --data-dir <path> [--ckpt-dir results/checkpoints] \\
        [--hf-repo ranro1/lipsyncnet-checkpoints] \\
        [--device cuda] [--epochs N]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lsn.config import apply_cli_overrides, load_config
from lsn.env import get_device
from lsn.training.runner import run


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data-dir", required=True, type=Path,
                   help="Directory containing speaker-subdir layout: <dir>/s1/*.npz")
    p.add_argument("--ckpt-dir", type=Path, default=Path("results/checkpoints"),
                   help="Parent directory; per-experiment subdir built from experiment_name")
    p.add_argument("--hf-repo", default=None,
                   help="Enable HF resume/upload (overrides YAML)")
    p.add_argument("--device", default=None, choices=["cuda", "cpu", None])
    p.add_argument("--epochs", type=int, default=None,
                   help="Override training.num_epochs (smoke-test convenience)")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = apply_cli_overrides(load_config(args.config), args)
    device = get_device(args.device)
    run(cfg, data_dir=args.data_dir, ckpt_dir=args.ckpt_dir, device=device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify --help works**

```bash
python scripts/train.py --help
```

Expected: argparse usage block printed.

- [ ] **Step 3: Commit**

```bash
git add scripts/train.py
git commit -m "feat(scripts): train.py CLI entry point"
```

---

## Task 22: `scripts/infer.py`

**Files:**
- Create: `scripts/infer.py`

- [ ] **Step 1: Implement `scripts/infer.py`**

```python
"""Run inference on a test split. Writes <experiment>_<dataset>_eval.json.

Usage:

    python scripts/infer.py --config configs/identity.yaml \\
        --weights <path-to-best_model.pt> \\
        --dataset {grid|lrs2} --data-dir <path> \\
        [--output-dir results/predictions] \\
        [--decoder {beam|greedy}] [--device cuda]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lsn.config import load_config
from lsn.data.datasets import (
    GridLipReadingDataset, LRS2Dataset, grid_collate_fn,
)
from lsn.data.splits import create_paper_split
from lsn.env import configure_cudnn, get_device, set_seed
from lsn.evaluation.inference import run_inference
from lsn.evaluation.report import write_eval_json
from lsn.models import build_from_config


def _build_test_loader(cfg, dataset: str, data_dir: Path):
    if dataset == "grid":
        # Rebuild the same paper-subset test split this model was held out from
        npz_paths = sorted(data_dir.glob("*/*.npz"))
        _, test_paths = create_paper_split(
            npz_paths, speakers=cfg.data.speakers,
            samples_per_speaker=cfg.data.samples_per_speaker,
            train_size=cfg.data.train_size, seed=cfg.data.seed,
        )
        ds = GridLipReadingDataset(test_paths)
    elif dataset == "lrs2":
        # All clips in the directory — no split logic for LRS2
        ds = LRS2Dataset(sorted(data_dir.glob("*.npz")))
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    return DataLoader(
        ds, batch_size=cfg.training.batch_size, shuffle=False,
        collate_fn=grid_collate_fn, num_workers=cfg.training.num_workers,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--dataset", required=True, choices=["grid", "lrs2"])
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("results/predictions"))
    p.add_argument("--decoder", default="beam", choices=["beam", "greedy"])
    p.add_argument("--device", default=None, choices=["cuda", "cpu", None])
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    set_seed(cfg.data.seed)
    configure_cudnn(benchmark=False)
    device = get_device(args.device)

    model = build_from_config(cfg.model, device=device)
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Carry history forward for report.py
    history = ckpt.get("history", [])
    final_epoch = ckpt.get("epoch", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))

    loader = _build_test_loader(cfg, args.dataset, args.data_dir)
    preds = run_inference(model, loader, device=device, decoder=args.decoder)

    out_path = args.output_dir / f"{cfg.experiment_name}_{args.dataset}_eval.json"
    write_eval_json(
        out_path,
        experiment_name=cfg.experiment_name,
        display_name=cfg.model.display_name or cfg.experiment_name,
        color=cfg.model.color or "#3498DB",
        dataset=args.dataset, decoder=args.decoder,
        final_epoch=final_epoch, best_val_loss=best_val_loss,
        history=history, predictions=preds,
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify --help**

```bash
python scripts/infer.py --help
```

Expected: argparse usage block.

- [ ] **Step 3: Commit**

```bash
git add scripts/infer.py
git commit -m "feat(scripts): infer.py CLI — inference + JSON write"
```

---

## Task 23: `scripts/report.py`

**Files:**
- Create: `scripts/report.py`

- [ ] **Step 1: Implement `scripts/report.py`**

```python
"""Read predictions JSONs → write plots + CSV tables.

Usage:
    python scripts/report.py --predictions-dir results/predictions \\
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
```

- [ ] **Step 2: Verify --help**

```bash
python scripts/report.py --help
```

- [ ] **Step 3: Commit**

```bash
git add scripts/report.py
git commit -m "feat(scripts): report.py CLI — plots + tables from JSONs"
```

---

## Task 24: Three committed YAML configs

**Files:**
- Create: `configs/identity.yaml`
- Create: `configs/paper.yaml`
- Create: `configs/transformer.yaml`

- [ ] **Step 1: Create `configs/identity.yaml`**

```yaml
experiment_name: run_identity_v1

model:
  backend: identity
  vocab_size: 40
  freeze_early_effnet: true
  use_self_attn: false
  backend_kwargs: {}
  display_name: "No temporal (identity)"
  color: "#E74C3C"

training:
  num_epochs: 100
  learning_rate: 1.0e-4
  batch_size: 2
  accum_steps: 4
  max_grad_norm: 5.0
  use_amp: true
  num_workers: 2
  prefetch: 4

data:
  dataset: grid
  speakers: [s1, s2, s3, s4, s5]
  samples_per_speaker: 200
  train_size: 450
  seed: 42

checkpointing:
  hf_repo: null
  hf_subfolder: null
```

- [ ] **Step 2: Create `configs/paper.yaml`**

(Differs from identity.yaml in 4 fields below.)

```yaml
experiment_name: run_paper_v1

model:
  backend: paper
  vocab_size: 40
  freeze_early_effnet: true
  use_self_attn: false   # spec §5: existing paper_best_model.pt was trained without self-attn
  backend_kwargs: {}
  display_name: "LipSyncNet (paper)"
  color: "#8E44AD"

training:
  num_epochs: 100
  learning_rate: 1.0e-4
  batch_size: 2
  accum_steps: 4
  max_grad_norm: 5.0
  use_amp: true
  num_workers: 2
  prefetch: 4

data:
  dataset: grid
  speakers: [s1, s2, s3, s4, s5]
  samples_per_speaker: 200
  train_size: 450
  seed: 42

checkpointing:
  hf_repo: null
  hf_subfolder: null
```

- [ ] **Step 3: Create `configs/transformer.yaml`**

```yaml
experiment_name: run_transformer_v1

model:
  backend: transformer_perstream
  vocab_size: 40
  freeze_early_effnet: true
  use_self_attn: false
  backend_kwargs:
    nhead: 4
    num_layers: 2
    d_model: 1024
    dropout: 0.1
  display_name: "Transformer encoder (per-stream)"
  color: "#27AE60"

training:
  num_epochs: 100
  learning_rate: 1.0e-4
  batch_size: 2
  accum_steps: 4
  max_grad_norm: 5.0
  use_amp: true
  num_workers: 2
  prefetch: 4

data:
  dataset: grid
  speakers: [s1, s2, s3, s4, s5]
  samples_per_speaker: 200
  train_size: 450
  seed: 42

checkpointing:
  hf_repo: null
  hf_subfolder: null
```

- [ ] **Step 4: Verify all three configs load**

```bash
python -c "
from lsn.config import load_config
for p in ['configs/identity.yaml', 'configs/paper.yaml', 'configs/transformer.yaml']:
    cfg = load_config(p)
    print(p, '→', cfg.experiment_name, cfg.model.backend)
"
```

Expected: three lines, no errors.

- [ ] **Step 5: Verify each config can build its model**

```bash
python -c "
import torch
from lsn.config import load_config
from lsn.models import build_from_config
for p in ['configs/identity.yaml', 'configs/paper.yaml', 'configs/transformer.yaml']:
    cfg = load_config(p)
    m = build_from_config(cfg.model, device=torch.device('cpu'))
    print(p, '→', type(m).__name__)
"
```

Expected: three lines (LipSyncNetVariant, LipSyncNetPaper, LipSyncNetVariant).

- [ ] **Step 6: Commit**

```bash
git add configs/
git commit -m "feat(configs): three YAML experiments — identity, paper, transformer

Reproducibility artifact (spec §1, §5)."
```

---

## Task 25: `docs/data-format.md`

**Files:**
- Create: `docs/data-format.md`

- [ ] **Step 1: Write the doc**

```markdown
# Data format — `.npz` contract

The training and inference codebase consumes a directory of `.npz` files.

## File contents

Each `.npz` must contain two arrays:

| Key      | Type         | Shape                         | Notes                            |
|----------|--------------|-------------------------------|----------------------------------|
| `frames` | `float32`    | `(75, 46, 140)` or `(75, 46, 140, 1)` | The trailing channel-1 dim is squeezed on load. |
| `label`  | `str`        | scalar                        | Lowercased + stripped on load.   |

## Directory layout

`GridLipReadingDataset` expects a speaker-subdirectory layout:

```
<data-dir>/
├── s1/
│   ├── s1_<clip-id>.npz
│   └── ...
├── s2/...
└── s5/...
```

`scripts/train.py` and `scripts/infer.py` glob `<data-dir>/*/*.npz`.

For LRS2, `LRS2Dataset` expects a flat directory of `.npz` files; pass the
LRS2 test directory directly to `scripts/infer.py --dataset lrs2 --data-dir <flat-dir>`.

## Validation

`GridLipReadingDataset.__getitem__` validates `frames.shape == (75, 46, 140)`
after squeezing the channel dim. A `ValueError` referencing this document is
raised on shape mismatch.

## Producing the files

The `.npz` files are produced by the preprocessing notebook
(`Copy of VSR_notebook_v1.ipynb`), which is out of scope for the current
codebase conversion. A future port will be tracked separately.
```

- [ ] **Step 2: Commit**

```bash
git add docs/data-format.md
git commit -m "docs: data-format.md — the .npz contract (spec §11)"
```

---

## Task 26: `docs/future-work.md`

**Files:**
- Create: `docs/future-work.md`

- [ ] **Step 1: Write the doc**

```markdown
# Future work

Items deferred from the LSN_TRAINING_EVAL.ipynb refactor. See the source
notebook's TODO markdown cells for full rationale.

## Preprocessing port (highest leverage)

`Copy of VSR_notebook_v1.ipynb` (and friends) currently produce the `.npz`
clips this codebase consumes. Port them into a `lsn.preprocessing` package
with a CLI: `python scripts/preprocess.py --input <mp4-dir> --output <npz-dir>`.

## Variable-T support for LRS2 (notebook TODO-PRE-4)

The current code uses fixed `T=75` for both GRID and LRS2. Proper LRS2
support requires:
- Computing the length distribution and choosing `MAX_T = 95th-percentile`
- Padding shorter clips to `MAX_T` and recording true lengths
- Passing true lengths as `input_lengths` to `CTCLoss` (padding frames must
  not contribute to the loss)
- Raising `_SinusoidalPE.max_len` to `MAX_T`

The `_SinusoidalPE.pe` buffer's shape is part of the §10 state_dict
invariant — variable-T support requires retraining or weight-init
migration; existing checkpoints assume `(1, 75, 1024)`.

## GRID-full speaker-independent split (notebook TODO-PRE-2)

3 speakers test / 2 val / 28 train, reported in a separate table. Implement
as `create_speaker_independent_split` alongside `create_paper_split`.

## LRS2 official splits (notebook TODO-PRE-3)

Use the official pre-train / train / val / test splits verbatim instead of
the current "all clips in <dir>" approach.

## CI checkpoint-load test

`test_checkpoint_compat.py` runs locally only when `LSN_CKPT_DIR` is set
(spec §11, §14). A CI job that downloads the smallest checkpoint
(`identity_best_model.pt`, ~121 MB) from HF and runs the strict-load test
would enforce the §10 invariant automatically. Out of scope for the
conversion.
```

- [ ] **Step 2: Commit**

```bash
git add docs/future-work.md
git commit -m "docs: future-work.md — deferred TODOs from the notebook (spec §13)"
```

---

## Task 27: `README.md`

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the README**

```markdown
# LipSyncNet (LSN) — Capstone

PyTorch re-implementation of LipSyncNet (3D-CNN + EfficientNet-B0 + temporal
backbone + CTC). This repo trains and evaluates three backend variants on
GRID and reports cross-dataset transfer to LRS2.

> **Source:** refactored from `notebooks/legacy/LSN_TRAINING_EVAL.ipynb` —
> see `docs/superpowers/specs/2026-04-29-lsn-notebook-to-codebase-design.md`
> for the full design.

---

## Results

(Generated by `python scripts/report.py`. Committed PNGs / CSVs live in `results/`.)

- `results/learning_curve_run_paper_v1.png` — paper-faithful run
- `results/learning_curve_run_identity_v1.png` — no-temporal-backbone ablation
- `results/learning_curve_run_transformer_v1.png` — per-stream Transformer encoder
- `results/learning_curves_comparison.png` — all three overlaid (val loss)
- `results/results_table_grid.csv` — Table 5 reproduction (CER, WER, word-acc, sentence-acc per model + paper baselines)
- `results/qualitative_examples_grid.csv` — Table 6 reproduction
- `results/results_table_lrs2.csv` — cross-dataset transfer

---

## Setup — local

Requires Python ≥3.10, PyTorch ≥2.1, CUDA optional (CPU works for inference).

```bash
git clone <repo-url>
cd <repo>
pip install -e .[dev]
```

The `.npz` clips (see `docs/data-format.md` for format) are produced by the
preprocessing notebook (currently out of scope; see `docs/future-work.md`).

## Setup — Colab

Open a fresh Colab notebook with a GPU runtime. Run these setup cells:

```python
# 1. Clone & install
!git clone <repo-url> lsn
%cd lsn
!pip install -e .

# 2. Mount Drive (data lives here)
from google.colab import drive
drive.mount('/content/drive')

# 3. HF token (only if you'll push checkpoints to Hub)
from google.colab import userdata
import os
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
```

Then run the CLI:

```python
!python scripts/train.py --config configs/identity.yaml \
    --data-dir /content/drive/MyDrive/LSN_Data/grid_processed_new \
    --hf-repo ranro1/lipsyncnet-checkpoints
```

## Setup — Kaggle

Add the GRID dataset to your notebook (Add data → search by name).
Add `HF_TOKEN` via Add-ons → Secrets → enable. In the notebook:

```python
!git clone <repo-url> /kaggle/working/lsn
%cd /kaggle/working/lsn
!pip install -e .

from kaggle_secrets import UserSecretsClient
import os
os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')

!python scripts/train.py --config configs/identity.yaml \
    --data-dir /kaggle/input/<your-grid-dataset>/grid_processed_new \
    --hf-repo ranro1/lipsyncnet-checkpoints
```

> Note: Kaggle commits time out at ~9h. The codebase resumes from the most
> recent `last_checkpoint.pt` automatically when `--hf-repo` is set. Disk
> requirement: ~5 GB working space for the largest checkpoint.

---

## Usage

### Train one experiment

```bash
python scripts/train.py --config configs/identity.yaml --data-dir <data-dir>
```

Three committed configs reproduce the three trained models:
`configs/identity.yaml`, `configs/paper.yaml`, `configs/transformer.yaml`.

### Inference (writes JSON predictions)

```bash
python scripts/infer.py --config configs/identity.yaml \
    --weights results/checkpoints/run_identity_v1/best_model.pt \
    --dataset grid --data-dir <test-data-dir>
```

For LRS2 cross-dataset transfer:

```bash
python scripts/infer.py --config configs/identity.yaml \
    --weights results/checkpoints/run_identity_v1/best_model.pt \
    --dataset lrs2 --data-dir <lrs2-test-dir>
```

### Report (plots + CSV tables)

```bash
python scripts/report.py --predictions-dir results/predictions
```

---

## Project layout

| Path                  | Purpose                                                         |
|-----------------------|-----------------------------------------------------------------|
| `src/lsn/models/`     | 3D-CNN, EfficientNet, temporal backends, top-level models       |
| `src/lsn/data/`       | Datasets, splits, vocab, LRS2 normalize                          |
| `src/lsn/training/`   | Loop, checkpoint (HF-gated), runner                              |
| `src/lsn/evaluation/` | Decoders, metrics, inference, report                             |
| `scripts/`            | `train.py`, `infer.py`, `report.py` — CLI entry points           |
| `configs/`            | Three YAML experiments — the reproducibility artifact            |
| `tests/`              | Smoke tests (model shapes, data contract, config roundtrip)      |
| `results/`            | Canonical PNGs + CSVs (committed); checkpoints + predictions are gitignored |
| `docs/`               | `data-format.md`, `future-work.md`, design spec                  |
| `notebooks/legacy/`   | Original `LSN_TRAINING_EVAL.ipynb` — preserved for lineage       |

---

## Tests

```bash
pytest -v
```

Backward-compat canary against existing checkpoints:

```bash
LSN_CKPT_DIR=/path/to/checkpoint-dir pytest tests/test_checkpoint_compat.py -v
```

---

## License

(Add your license of choice.)
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README — overview, setup per env, usage"
```

---

## Task 28: End-to-end smoke run + final pass

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

```bash
pytest -v
```

Expected: all tests pass (or skipped where checkpoints/data are gated by env vars). Specifically:
- `test_vocab.py`: 4 passed
- `test_env.py`: 6 passed
- `test_config.py`: 7 passed
- `test_models_*`: ≥16 passed
- `test_checkpoint_compat.py`: 3 skipped (or 3 passed if `LSN_CKPT_DIR` is set)
- `test_datasets.py`: 4 passed
- `test_splits.py`: 3 passed
- `test_normalize.py`: 2 passed
- `test_hf_store.py`: 3 passed
- `test_checkpoint.py`: 3 passed
- `test_loop.py`: 1 passed
- `test_decoders.py`: 3 passed
- `test_metrics.py`: 7 passed
- `test_inference.py`: 1 passed
- `test_report.py`: 4 passed

- [ ] **Step 2: Smoke-run training for 1 epoch**

If you have local `.npz` data:

```bash
python scripts/train.py --config configs/identity.yaml \
    --data-dir <path-to-npz> --epochs 1 --device cpu
```

Expected: one epoch completes; `results/checkpoints/run_identity_v1/last_checkpoint.pt` and `best_model.pt` are written. (~10 min on CPU; faster on GPU.)

- [ ] **Step 3: Smoke-run inference**

```bash
python scripts/infer.py --config configs/identity.yaml \
    --weights results/checkpoints/run_identity_v1/best_model.pt \
    --dataset grid --data-dir <path-to-npz>
```

Expected: `results/predictions/run_identity_v1_grid_eval.json` written.

- [ ] **Step 4: Smoke-run report**

```bash
python scripts/report.py --predictions-dir results/predictions
```

Expected: `results/learning_curve_run_identity_v1.png`, `results/learning_curves_comparison.png`, `results/results_table_grid.csv`, `results/qualitative_examples_grid.csv` are written.

- [ ] **Step 5: Run backward-compat canary against real checkpoints**

```bash
LSN_CKPT_DIR=/path/to/checkpoints pytest tests/test_checkpoint_compat.py -v
```

Expected: 3 passed. **This is the gold-standard verification of the refactor.** If it fails, the refactor introduced state_dict drift.

- [ ] **Step 6: Final commit (if any cleanup)**

```bash
git status
# Address any uncommitted artifacts
git commit -m "chore: end-to-end verification pass" --allow-empty
```

- [ ] **Step 7: Final manual review**

Open the repo in your editor and verify:
- `README.md` renders correctly
- `results/` PNGs (after a real training run) are committable
- No `print(...)` statements remain in `src/lsn/` (only in `scripts/`)
- No `google.colab` / `kaggle_secrets` imports anywhere in `src/lsn/`

```bash
grep -r "print(" src/lsn/ || echo "no print() calls"
grep -r "google.colab\|kaggle_secrets" src/lsn/ || echo "no env-specific imports"
```

Expected: both echo "no ..." (per spec §9.1, §9.2).

---

## Done

The codebase is now:
- Importable as a package (`pip install -e .`)
- Runnable end-to-end via the three CLI scripts
- Backward-compatible with the user's three trained checkpoints (verified via Task 9 + Task 28 Step 5)
- Documented (README + spec + data-format + future-work)
- Tested (~50 smoke tests covering shapes, contracts, round-trips)
- Committed in small, focused commits suitable for a research-presentation artifact

The user can now: train new variants by adding a YAML to `configs/`; share the repo; present it as their capstone artifact with `results/` figures embedded inline in the README.
