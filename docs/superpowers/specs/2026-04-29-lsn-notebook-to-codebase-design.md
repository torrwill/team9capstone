# LSN Notebook → Codebase Design

**Date:** 2026-04-29
**Status:** Draft for review
**Source notebook:** `LSN_TRAINING_EVAL.ipynb`

---

## 1. Goal

Refactor `LSN_TRAINING_EVAL.ipynb` into a presentable Python codebase such that:

1. Anyone can clone and run end-to-end on local, Colab, or Kaggle without environment-specific code edits.
2. Each experiment's hyperparameter set is a committed YAML — the *reproducibility artifact* a research committee can inspect.
3. Final figures and tables ship inline (in `results/`) so reviewers can assess outputs without installing anything.
4. The notebook's existing behavior is preserved bit-for-bit where it matters (state_dict layouts, training math, decoder behavior). This is a faithful refactor, not a redesign.

## 2. Scope

### 2.1 In scope

Everything currently implemented in `LSN_TRAINING_EVAL.ipynb`:

- **Models.** `LipSyncNetPaper` (paper-faithful, no input projection, optional self-attention) and `LipSyncNetVariant` (modular ablation rig). Backends: `paper` (alias for the BiLSTM+self-attn used inside `LipSyncNetPaper`), `identity`, `bilstm`, `transformer`, `transformer_perstream`. Frontend: `Conv3DBlock`, `Frontend3DCNN`, `EfficientNet` (stages 0–6 frozen). Temporal primitives: `_SinusoidalPE`, `SelfAttentionBlock`.
- **Data.** `GridLipReadingDataset`, `LRS2Dataset`, `grid_collate_fn`, `create_paper_split`, vocab + `encode_text` / `decode_ids`.
- **Training.** AMP forward / fp32 CTC, gradient accumulation, BN-eval-when-params-frozen, gradient clipping (max_norm=5.0), `CTCLoss(blank=0, zero_infinity=True)`, Adam(lr=1e-4), 100 epochs.
- **Checkpointing.** Single self-describing `.pt` containing `{epoch, model_state_dict, optimizer_state_dict, scaler_state_dict, train_loss, val_loss, best_val_loss, history}`. Atomic-ish save via `/tmp`. Resume priority: HF `last` → HF `best` → local `last` → local `best` → fresh.
- **Evaluation.** Greedy + beam=100 CTC decoders, edit-distance-based CER / WER / word_acc / sentence_acc, learning-curve plots (per-model + comparison), Table 5 reproduction (results), Table 6 reproduction (qualitative examples).
- **LRS2 cross-dataset transfer.** `LRS2Dataset` + LRS2 text normalizer + Stage I/J equivalents.
- **Three committed configs.** `configs/identity.yaml`, `configs/paper.yaml`, `configs/transformer.yaml` — one per already-trained run.
- **README.** Per-environment setup (local / Colab / Kaggle), usage, results inline.
- **Smoke tests.** Three tests, ~50 lines total (see §10).
- **Disposition of source notebook.** Move `LSN_TRAINING_EVAL.ipynb` to `notebooks/legacy/` (preserved for lineage, not the entry point).

### 2.2 Out of scope

- Preprocessing (`.mp4` → `.npz`) — `Copy of VSR_notebook_v1.ipynb` continues to handle this. The new codebase consumes `.npz` files via the contract in §11.
- Variable-T support for LRS2 (notebook TODO-PRE-4) — fixed T=75 only, same as today. Documented as future work.
- New backends, new datasets, new metrics, new training tricks.
- Old-checkpoint migration tooling — not needed; the backward-compat invariant in §9 ensures existing `.pt` files load directly.
- Markdown TODO items in the source notebook flagged as future work (PRE-2 GRID-full split, PRE-3/4 LRS2 access + variable-T, etc.) — moved to `docs/future-work.md`, not implemented.

## 3. Repo layout

```
lsn/
├── README.md
├── pyproject.toml
├── .gitignore                      # checkpoints/, predictions/, __pycache__, *.egg-info
├── configs/
│   ├── identity.yaml
│   ├── paper.yaml
│   └── transformer.yaml
├── src/lsn/
│   ├── __init__.py
│   ├── config.py                   # YAML loader, schema dataclasses, CLI override merge
│   ├── env.py                      # device selection, seeding helpers
│   ├── models/
│   │   ├── __init__.py             # public API: build_paper_model, build_variant, count_parameters
│   │   ├── frontend.py             # Conv3DBlock, Frontend3DCNN, EfficientNet
│   │   ├── backends.py             # _SinusoidalPE, BiLSTMBackend, IdentityBackend,
│   │   │                           #   TransformerBackend, TransformerBackendPerStream, _BACKEND_REGISTRY
│   │   └── lipsyncnet.py           # SelfAttentionBlock, LipSyncNetPaper, LipSyncNetVariant
│   ├── data/
│   │   ├── __init__.py
│   │   ├── vocab.py                # CHARS, char_to_idx, idx_to_char, encode_text, decode_ids,
│   │   │                           #   BLANK_INDEX, VOCAB_SIZE, NUM_CLASSES
│   │   ├── datasets.py             # GridLipReadingDataset, LRS2Dataset, grid_collate_fn
│   │   ├── splits.py               # create_paper_split
│   │   └── normalize.py            # LRS2 text normalization
│   ├── training/
│   │   ├── __init__.py
│   │   ├── checkpoint.py           # save_checkpoint_safe, try_resume,
│   │   │                           #   _model_state_dict, _load_into_model, freeze_bn_stats
│   │   ├── hf_store.py             # HFStore class — only file importing huggingface_hub
│   │   ├── loop.py                 # train_one_epoch, validate_one_epoch
│   │   └── runner.py               # run() — wires config → model + data + loop + checkpoint
│   └── evaluation/
│       ├── __init__.py
│       ├── decoders.py             # greedy_decode, beam_decode (beam=100, paper setting)
│       ├── metrics.py              # edit_distance, cer, wer, word_acc, sentence_acc
│       ├── inference.py            # run_inference: model + loader → list[Prediction]
│       └── report.py               # learning curves, Table 5/6 builders
├── scripts/
│   ├── train.py                    # python scripts/train.py --config ...
│   ├── infer.py                    # python scripts/infer.py --config ... --weights ...
│   └── report.py                   # python scripts/report.py --predictions-dir ...
├── tests/
│   ├── test_model_shapes.py
│   ├── test_data_contract.py
│   └── test_config_roundtrip.py
├── results/                        # canonical artifacts (committed PNGs + CSVs)
│   ├── learning_curve_run_identity_v1.png
│   ├── learning_curve_run_paper_v1.png
│   ├── learning_curve_run_transformer_v1.png
│   ├── learning_curves_comparison.png
│   ├── results_table_grid.csv
│   ├── qualitative_examples_grid.csv
│   └── results_table_lrs2.csv
├── docs/
│   ├── data-format.md              # the .npz contract (frames shape + label key + dir layout)
│   └── future-work.md              # carried-over TODOs from the notebook (variable-T, GRID-full split, etc.)
└── notebooks/
    └── legacy/
        └── LSN_TRAINING_EVAL.ipynb # moved from repo root, preserved for lineage
```

## 4. Module contracts

Each module has a clear, narrow purpose. Public API surfaces are listed; internals are free to change.

### 4.1 `lsn.config`

Loads YAML files into typed dataclasses; merges CLI overrides.

```python
@dataclass
class ModelCfg:
    backend: str                 # paper | identity | bilstm | transformer | transformer_perstream
    vocab_size: int = 40         # see note below — DO NOT "fix" to len(CHARS)
    freeze_early_effnet: bool = True
    use_self_attn: bool = False  # only honored when backend == "paper"
    backend_kwargs: dict = field(default_factory=dict)
    display_name: str | None = None   # for plots/tables; defaults to experiment_name
    color: str | None = None          # hex string for plots; required for report.py

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
    dataset: str                 # grid | lrs2 (training is grid-only; lrs2 only for inference)
    speakers: list[str]
    samples_per_speaker: int
    train_size: int
    seed: int

@dataclass
class CkptCfg:
    hf_repo: str | None = None
    hf_subfolder: str | None = None  # defaults to experiment_name when None

@dataclass
class Config:
    experiment_name: str
    model: ModelCfg
    training: TrainingCfg
    data: DataCfg
    checkpointing: CkptCfg

def load_config(path: Path) -> Config: ...
def apply_cli_overrides(cfg: Config, args: argparse.Namespace) -> Config: ...
```

**Note on `vocab_size=40` (load-bearing, do not change).** The notebook sets `VOCAB_SIZE=40` early and uses it as the classifier dimension (`nn.Linear(1024, vocab_size + 1)` → 41 output channels). The actual encodable alphabet (`CHARS = [' '] + ascii_lowercase`) is only 27 characters, so the trained classifier has 41 channels of which only 28 (blank + space + 26 letters) are ever populated. **`vocab_size=40` is preserved verbatim from the trained checkpoints; "fixing" it to `len(CHARS)` would change the classifier's weight shape and break `strict=True` load of every existing `.pt` file.** Annotate this in `ModelCfg`, `lsn.data.vocab`, and the README.

### 4.2 `lsn.env`

```python
def set_seed(seed: int) -> None: ...      # python, numpy, torch (cpu+cuda)
def get_device(override: str | None) -> torch.device: ...   # cuda if available else cpu
def configure_cudnn(benchmark: bool) -> None: ...
```

### 4.3 `lsn.models`

Public API mirrors the notebook's:

```python
def build_paper_model(vocab_size: int = 40,
                      use_self_attn: bool = False,
                      device: str = "cpu") -> LipSyncNetPaper: ...
def build_variant(backend: str = "bilstm",
                  vocab_size: int = 40,
                  device: str = "cpu",
                  **backend_kwargs) -> LipSyncNetVariant: ...
def count_parameters(model: nn.Module) -> dict[str, int]: ...   # total / trainable / frozen
```

Plus a `build_from_config(cfg: ModelCfg, device: torch.device) -> nn.Module` convenience that dispatches to the right builder.

`_BACKEND_REGISTRY` stays internal to `models.backends`; consumers go through `build_variant` / `build_from_config`.

### 4.4 `lsn.data`

```python
class GridLipReadingDataset(Dataset): ...
class LRS2Dataset(Dataset): ...

def grid_collate_fn(batch) -> dict: ...   # signature unchanged from notebook

def create_paper_split(npz_paths: list[Path],
                       speakers: list[str] | None = None,
                       samples_per_speaker: int = 200,
                       train_size: int = 450,
                       seed: int = 42) -> tuple[list[Path], list[Path]]: ...

def normalize_lrs2(text: str) -> str: ...
```

`vocab.py` exports the constants (`CHARS`, `BLANK_INDEX`, `VOCAB_SIZE`, `NUM_CLASSES`) and the `encode_text` / `decode_ids` helpers.

### 4.5 `lsn.training`

```python
# checkpoint.py
LAST_CKPT_NAME = "last_checkpoint.pt"
BEST_CKPT_NAME = "best_model.pt"

def save_checkpoint_safe(model, optimizer, scaler, *,
                         epoch, train_loss, val_loss, best_val_loss, history,
                         save_path: Path,
                         remote: HFStore | None = None) -> None:
    """Atomic-ish save: write to /tmp, copy to save_path, optionally upload to remote.
    save_path's filename should be LAST_CKPT_NAME or BEST_CKPT_NAME."""

def try_resume(model, optimizer, scaler, device,
               local_dir: Path,
               remote: HFStore | None) -> tuple[int, float, list]:
    """Try to resume in priority order:
        1) remote.try_download(LAST_CKPT_NAME)
        2) remote.try_download(BEST_CKPT_NAME)
        3) local_dir / LAST_CKPT_NAME
        4) local_dir / BEST_CKPT_NAME
        5) fresh — return (0, float('inf'), [])
    Loads via torch.load(..., weights_only=False) — required because legacy
    checkpoints contain optimizer_state_dict (not weight tensors only)."""

def freeze_bn_stats(model: nn.Module) -> int:
    """Set BatchNorm modules to eval mode IFF all their direct parameters are
    frozen (requires_grad=False). Used to keep frozen-EfficientNet BN running
    stats from drifting on small batches. Returns count of BN modules frozen.
    The 3D-CNN's BatchNorm3d layers (whose params are trainable) stay in
    training mode and update running stats normally."""

# hf_store.py — the ONLY file importing huggingface_hub
class HFStore:
    def __init__(self, repo: str, subfolder: str, token: str | None = None):
        """token=None → read from HF_TOKEN env var."""
    def upload(self, local_path: Path, remote_filename: str, *, commit_message: str) -> bool: ...
    def try_download(self, remote_filename: str, local_dest: Path) -> Path | None: ...

# loop.py
def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler,
                    max_grad_norm: float, accum_steps: int, use_amp: bool) -> dict: ...
def validate_one_epoch(model, loader, loss_fn, device, use_amp: bool) -> dict: ...

# runner.py
def run(cfg: Config, *, data_dir: Path, ckpt_dir: Path, device: torch.device) -> None:
    """End-to-end training run from a Config. Called by scripts/train.py.

    Builds the per-experiment subdirectory ckpt_dir / cfg.experiment_name and
    passes that to save_checkpoint_safe / try_resume as local_dir. Caller
    (scripts/train.py) supplies the parent ckpt_dir; runner.run owns the
    per-experiment naming."""
```

### 4.6 `lsn.evaluation`

```python
# decoders.py
def greedy_decode(log_probs: Tensor) -> list[str]: ...      # input shape (T, B, C)
def beam_decode(log_probs: Tensor,
                input_lengths: Tensor,                       # (B,) — required by torchaudio
                beam_width: int = 100) -> list[str]: ...

# metrics.py
def edit_distance(ref: list[str], hyp: list[str]) -> int: ...
def cer(refs: list[str], hyps: list[str]) -> float: ...
def wer(refs: list[str], hyps: list[str]) -> float: ...
def word_acc(refs: list[str], hyps: list[str]) -> float: ...   # 1 - WER, paper convention
def sentence_acc(refs: list[str], hyps: list[str]) -> float: ...

# inference.py
@dataclass
class Prediction:
    path: str
    reference: str
    hypothesis: str

def run_inference(model, loader, device, decoder: str = "beam") -> list[Prediction]: ...

# report.py
def write_eval_json(out_path: Path, *, cfg: Config, history: list, predictions: list[Prediction],
                    dataset: str, decoder: str, final_epoch: int, best_val_loss: float) -> None: ...
def plot_learning_curves(eval_jsons: list[Path], out_dir: Path) -> None: ...
def write_results_table(eval_jsons: list[Path], out_path: Path) -> None: ...
def write_qualitative_examples(eval_jsons: list[Path], out_path: Path, n_per_model: int = 5) -> None: ...
```

## 5. Config schema — full example

`configs/identity.yaml`:

```yaml
experiment_name: run_identity_v1

model:
  backend:              identity
  vocab_size:           40
  freeze_early_effnet:  true
  use_self_attn:        false
  backend_kwargs: {}
  display_name: "No temporal (identity)"
  color: "#E74C3C"

training:
  num_epochs:    100
  learning_rate: 1.0e-4
  batch_size:    2
  accum_steps:   4
  max_grad_norm: 5.0
  use_amp:       true
  num_workers:   2
  prefetch:      4

data:
  dataset:             grid
  speakers:            [s1, s2, s3, s4, s5]
  samples_per_speaker: 200
  train_size:          450
  seed:                42

checkpointing:
  hf_repo:      null
  hf_subfolder: null
```

`configs/paper.yaml` differs in: `backend: paper`, `use_self_attn: false` (the existing `paper_best_model.pt` was trained without self-attention per the notebook's `MODELS` registry builder; flipping this to `true` would break checkpoint load), `display_name: "LipSyncNet (paper)"`, `color: "#8E44AD"`.

`configs/transformer.yaml` differs in: `backend: transformer_perstream`, `backend_kwargs: {nhead: 4, num_layers: 2, d_model: 1024, dropout: 0.1}`, `display_name: "Transformer encoder (per-stream)"`, `color: "#27AE60"`.

## 6. CLI commands

```bash
# Training
python scripts/train.py \
  --config configs/identity.yaml \
  --data-dir <path to .npz dir> \
  [--ckpt-dir results/checkpoints/]      \   # default: results/checkpoints/<experiment_name>/
  [--hf-repo ranro1/lipsyncnet-checkpoints]\ # opt-in HF
  [--device cuda]                          \ # default: auto-detect
  [--epochs N]                               # smoke-test override

# Inference
python scripts/infer.py \
  --config configs/identity.yaml \
  --weights <path to best_model.pt> \
  --dataset {grid|lrs2} \
  --data-dir <path to test .npz dir> \
  [--output-dir results/predictions/]      \
  [--decoder {beam|greedy}]                \ # default beam (paper setting)
  [--device cuda]

# Report
python scripts/report.py \
  --predictions-dir results/predictions/ \
  [--output-dir results/]
```

CLI flags override only env-specific fields. Hyperparameters and architectural choices live in YAML and are *not* exposed as flags (with the exception of `--epochs` for smoke tests).

**Exact list of CLI-overridable fields** (anything not on this list belongs in YAML):

| Flag             | Overrides            | Scripts                                 |
|------------------|----------------------|-----------------------------------------|
| `--data-dir`     | (CLI-only, not YAML) | `train.py`, `infer.py`                  |
| `--ckpt-dir`     | (CLI-only, not YAML) | `train.py`                              |
| `--output-dir`   | (CLI-only, not YAML) | `infer.py`, `report.py`                 |
| `--predictions-dir` | (CLI-only, not YAML) | `report.py`                          |
| `--weights`      | (CLI-only, not YAML) | `infer.py`                              |
| `--dataset`      | (CLI-only, not YAML) | `infer.py`                              |
| `--decoder`      | (CLI-only, not YAML) | `infer.py`                              |
| `--device`       | (CLI-only, not YAML) | `train.py`, `infer.py`                  |
| `--hf-repo`      | `cfg.checkpointing.hf_repo`        | `train.py`                |
| `--epochs`       | `cfg.training.num_epochs` (smoke) | `train.py`                 |

`apply_cli_overrides(cfg, args)` mutates only `cfg.checkpointing.hf_repo` and `cfg.training.num_epochs`. Everything else is consumed directly by the script (it doesn't belong in `Config`).

**Why `data_dir` / `ckpt_dir` / `output_dir` are not in YAML:** these are environment-specific (different on each machine). Putting them in YAML would force every clone to edit committed files. Keeping them as CLI flags keeps the YAML environment-portable.

## 7. Predictions JSON contract

Written by `infer.py` to `<output-dir>/<experiment_name>_<dataset>_eval.json`:

```json
{
  "experiment_name": "run_identity_v1",
  "display_name":    "No temporal (identity)",
  "color":           "#E74C3C",
  "dataset":         "grid",
  "decoder":         "beam",
  "final_epoch":     100,
  "best_val_loss":   1.234,
  "history": [
    {"epoch": 1, "train_loss": 3.41, "val_loss": 3.55, "grad_norm": 4.8, "train_time": 280.1, "val_time": 35.2},
    ...
  ],
  "predictions": [
    {"path": ".../s1_pgaq6n.npz", "reference": "set white at b nine again", "hypothesis": "set white at b nine again"},
    ...
  ]
}
```

`report.py` is a near-pure consumer of these JSONs. The single exception: paper-baseline rows in Table 5 (Xu, Gergen, Margam, "LipSyncNet paper-reported") cannot come from any JSON — they're hardcoded as a constant `PAPER_BASELINES_GRID` in `lsn.evaluation.report`, with provenance comments citing the notebook source. No other side-channel.

**LRS2 normalization site.** `predictions[].reference` and `predictions[].hypothesis` in the JSON contain *raw* decoder output (lowercased, stripped). LRS2-specific normalization (digit/punctuation handling) is applied at metric-compute time inside `report.py` — driven by the JSON's `dataset` field — not at JSON-write time. Rationale: keeps the JSON debuggable (raw model outputs preserved), keeps GRID and LRS2 JSONs symmetric in shape, keeps normalization logic colocated with metric logic.

## 8. Results folder layout

```
results/
├── checkpoints/                              # written by train.py — .gitignored
│   ├── run_identity_v1/{last_checkpoint.pt, best_model.pt}
│   ├── run_paper_v1/...
│   └── run_transformer_v1/...
├── predictions/                              # written by infer.py — .gitignored
│   ├── run_identity_v1_grid_eval.json
│   ├── run_identity_v1_lrs2_eval.json
│   └── ...
├── learning_curve_<experiment_name>.png      # written by report.py — committed
├── learning_curves_comparison.png
├── results_table_grid.csv
├── qualitative_examples_grid.csv
└── results_table_lrs2.csv
```

## 9. Cross-cutting concerns

### 9.1 Logging
`logging.getLogger(__name__)` everywhere; `print()` is banned in `src/lsn/`. `scripts/*.py` configure root logger at INFO with `%(asctime)s %(levelname)s %(name)s: %(message)s`. `tqdm` stays for training/inference progress bars.

### 9.2 Environment detection — none
The codebase has zero `if "google.colab" in sys.modules` / `if os.path.exists("/kaggle")` branches. All env-specific glue (Drive mount, secret resolution, dataset path) is the user's responsibility, performed in setup cells before invoking the CLI. The README provides copy-paste blocks per environment.

### 9.3 Dependencies and HuggingFace gating

**Pinned in `pyproject.toml`** (all required, not optional):
- `torch>=2.1`, `torchvision>=0.16`, `torchaudio>=2.1` — `torchaudio.models.decoder.ctc_decoder` powers `beam_decode`
- `numpy`, `tqdm`, `pyyaml`, `matplotlib`, `pandas`
- `huggingface_hub` — installed for everyone, used only when `hf_repo` is set
- (no `jiwer` — metrics are computed via in-house `edit_distance`, matching the notebook)

**HF gating:**
- `lsn.training.hf_store` is the *only* module that imports `huggingface_hub`.
- `runner.run()` instantiates `HFStore` only if `cfg.checkpointing.hf_repo` is set; otherwise passes `None` everywhere.
- `save_checkpoint_safe` and `try_resume` accept `remote: HFStore | None` and skip remote ops when `None`.
- `HFStore.__init__(repo, subfolder, token=None)` — when `token=None`, reads from the `HF_TOKEN` env var. The README's Colab/Kaggle setup blocks export `HF_TOKEN` from `userdata` / `kaggle_secrets`. There is no `google.colab.userdata` import inside `lsn`.

**Beam decoder tokens.** `torchaudio.models.decoder.ctc_decoder` requires a tokens list/file. Pass the in-memory `[blank] + CHARS` list directly (the API accepts a `list[str]`); avoid writing a tokens file to disk. No environment-specific paths.

### 9.4 Determinism
`env.set_seed(seed)` seeds `random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`. `cudnn.benchmark` defaults to `True` for training (matches notebook), `False` for inference. **Reproducibility scope:** model *structure* and split selection are deterministic; full re-training is not bitwise reproducible because `cudnn.benchmark=True` selects non-deterministic conv algorithms. This matches the notebook and is accepted — re-runs land in the same neighborhood, not on the same numbers.

### 9.5 Device selection
`env.get_device(override)` returns `cuda` if available else `cpu`. CLI `--device cpu` forces CPU (for local eval-only on a laptop).

### 9.6 Boundary validation
- `GridLipReadingDataset.__getitem__` validates `frames.shape[-3:] == (75, 46, 140)` with a clear `ValueError` referencing `docs/data-format.md`.
- `lsn.config.load_config` validates required fields and types via dataclass machinery.
- Internal contracts (model output shapes, optimizer state structure) are trusted — no defensive checks.

## 10. Backward-compatibility invariants

The `state_dict` keys produced by the new code MUST be byte-identical to those in the user's existing trained checkpoints (`paper_best_model.pt`, `identity_best_model.pt`, `transformer_best_model.pt`) so they load via the new codebase without conversion.

This freezes the following names:

**`LipSyncNetPaper` top-level attributes:**
- `cnn3d`, `efficientnet`, `lstm1`, `drop1`, `lstm2`, `drop2`, `self_attn` (only present when `use_self_attn=True`), `classifier`

**`LipSyncNetVariant` top-level attributes:**
- `cnn3d`, `efficientnet`, `backend`, `classifier`

**Frontend internals** (in both top-level classes):
- `cnn3d.block{1..4}.conv.weight`, `cnn3d.block{1..4}.bn.{weight,bias,running_mean,running_var,num_batches_tracked}` — `pool` is parameter-free, no state
- `efficientnet.features.{0..8}.*` — torchvision EfficientNet-B0 internal naming, frozen across torchvision versions

**Backend internals (variant-specific):**
- `BiLSTMBackend`: `backend.input_proj.{weight,bias}`, `backend.lstm1.weight_{ih,hh}_l0[_reverse]`, `backend.lstm1.bias_{ih,hh}_l0[_reverse]`, `backend.lstm2.<same pattern>`. (No params for `Dropout`.)
- `IdentityBackend`: no params.
- `TransformerBackend`: `backend.input_proj.{weight,bias}`, `backend.pos_enc.pe` (registered buffer, shape `(1, 75, 1024)` — caps the model at T≤75 by design), `backend.encoder.layers.{L}.*` (PyTorch internal MHA/FFN keys).
- `TransformerBackendPerStream`: `backend.cnn_proj.0.{weight,bias}`, `backend.cnn_proj.1.{weight,bias}` (Linear+LayerNorm), `backend.eff_proj.{0,1}.<same>`, `backend.pos_enc.pe`, `backend.encoder.layers.{L}.*`.

**`SelfAttentionBlock` internals (only present in `LipSyncNetPaper` with `use_self_attn=True`):**
- `self_attn.attn.in_proj_weight`, `self_attn.attn.in_proj_bias`, `self_attn.attn.out_proj.{weight,bias}`, `self_attn.norm.{weight,bias}`. Refactoring this class (renaming `attn`→`mha` or `norm`→`ln`) breaks `strict=True` load — don't.

**LSTM-specific note:** `nn.LSTM` expands into 4 sub-parameter names per layer per direction (`weight_ih`, `weight_hh`, `bias_ih`, `bias_hh`), with `_reverse` suffix for the backward direction in bidirectional mode. These are stable across PyTorch versions and inherited unchanged from `nn.LSTM`.

**Classifier:** `classifier.weight` shape `(41, 1024)`, `classifier.bias` shape `(41,)`. Driven by `vocab_size=40 + 1` blank — see §4.1 note.

`test_model_shapes.py` (§11) loads each existing checkpoint into the corresponding new-codebase model with `strict=True` to enforce this invariant.

## 11. Smoke tests

Three tests in `tests/`, ~50 lines total, runnable on CPU in <30 seconds:

### `test_model_shapes.py`
For each backend in `{paper, identity, bilstm, transformer, transformer_perstream}`:
- Build via `build_from_config`
- Forward pass on `torch.randn(2, 75, 46, 140)`
- Assert output shape `(75, 2, 41)`
- (When committed checkpoints are available locally:) load each existing `best_model.pt` with `strict=True` to enforce the §10 invariant.

### `test_data_contract.py`
- Write a synthetic `.npz` with `frames` of shape `(75, 46, 140)` and `label="set white at b nine again"` to a temp dir.
- Load via `GridLipReadingDataset`.
- Assert sample fields (shape, dtype, decoded text round-trip).
- Write a malformed `.npz` (wrong shape) and assert the dataset raises `ValueError` (assert exception type only, not message text — message wording is allowed to evolve).

### `test_config_roundtrip.py`
- Load each committed YAML in `configs/`.
- Assert dataclass round-trip (load → dump → load yields equal objects).
- Assert each config can build its model via `build_from_config`.

## 12. What gets dropped (cells from the source notebook NOT ported)

### Pure exploration / debugging cells — discarded entirely:
- **Cell 13** — `if __name__ == "__main__":` block that builds an `identity` variant on dummy input and asserts `n_lstm == 0`. Replaced by `test_model_shapes.py`.
- **Cell 18** — `count speakers files`: prints per-speaker `.npz` counts.
- **Cell 20** — `validate Dataset`: prints `dataset[0]` fields. Replaced by `test_data_contract.py`.
- **Cell 26** — `TEST TRANSFORMER CHECKPOINT`: introspects a `.pt` file's pickle internals.

### Conversational `print()` noise — logic ported, prints downgraded to a single `logger.info` line each:
- Cell 21's per-speaker split summary.
- Cell 22's `print("HF ready, repo:", HF_REPO)`.
- Cell 24's DataLoader summary.
- Cell 27's `MODELS` registry status report (replaced wholesale by config files).
- Cell 28's per-model `[ok]` / `[skip]` history-loading prints.

### Markdown TODO cells — split:
- Implemented items → moved into module/function docstrings as design rationale.
- Unimplemented items (variable-T LRS2, GRID-full split, etc.) → moved to `docs/future-work.md`.

### Architectural-discrepancy notes (cells 1–2 markdown) — *load-bearing* documentation:
Moved into class-level docstrings of `EfficientNet`, `LipSyncNetPaper`, `SelfAttentionBlock` so the rationale lives next to the code it justifies.

### Class renames (single-source-of-truth list):
- `LRS2LipReadingDataset` → `LRS2Dataset` (shorter; matches `GridLipReadingDataset` siblinghood loosely — only rename, no behavior change). No other class renames.

## 13. Future work (post-conversion, documented in `docs/future-work.md`)

- Port preprocessing (`Copy of VSR_notebook_v1.ipynb`) into `lsn.preprocessing`.
- Variable-T LRS2 support (notebook TODO-PRE-4) — proper `input_lengths` plumbing through CTC, padded collate, `_SinusoidalPE.max_len` adjustment.
- GRID-full speaker-independent split (TODO-PRE-2) — `create_speaker_independent_split` alongside `create_paper_split`.
- LRS2 official splits (TODO-PRE-3).

## 14. Open questions / risks

- **Existing checkpoints' compatibility — needs verification.** The §10 invariant is the assumed-true compatibility contract. The implementation plan should include a step that loads each of the user's three existing `.pt` files into the new codebase early in the work, before any code is committed, so any state_dict drift is caught immediately.
- **Disk space for resume.** `paper_last_checkpoint.pt` is ~3.67 GB (per the notebook's TEST TRANSFORMER CHECKPOINT cell). `try_resume` downloads via `hf_hub_download` to `local_dir`. The README notes that the user is responsible for provisioning enough disk in the working directory (Kaggle `/kaggle/working` ≈ 20 GB; Colab `/tmp` may be smaller).
- **`torch.load` security.** PyTorch 2.6+ defaults `weights_only=True`. Legacy checkpoints contain `optimizer_state_dict` (Python objects, not just tensors) and will fail to load with the new default. `try_resume` and any other loader of these files MUST pass `weights_only=False` explicitly. Spec §4.5 calls this out in the `try_resume` docstring.
- **CI checkpoint-load test.** §11's strict-load test runs only when checkpoints are available locally. The implementation plan should consider a CI step that downloads the smallest checkpoint (`identity_best_model.pt` ≈ 121 MB) from HF as part of the test job. Without it, §10 is enforced only manually. (Out of scope for this conversion; flagged for future-work.)
- **`backend: paper` vs `LipSyncNetPaper`.** The current notebook has `LipSyncNetPaper` as a separate class (with `cnn3d`, `lstm1`, `drop1`, `lstm2`, `drop2`, `self_attn`, `classifier`) and `LipSyncNetVariant` as the modular rig. We unify config dispatch via `build_from_config` but keep both classes — `backend: paper` in YAML routes to `build_paper_model(...)`, all other backends route to `build_variant(...)`. The configs and checkpoints stay distinguishable.
- **`huggingface_hub` as required dep.** Even local-only users install it (small, well-behaved). Decision: accept this; making it optional adds conditional-import code paths without meaningful benefit.
- **Plot `color` in YAML.** Putting plotting metadata in the run config is mildly cross-cutting. Acceptable because it keeps the report stage purely a JSON consumer; alternative would be a separate `configs/plots.yaml`. Going with the simpler in-line approach.

---

## Acceptance

This spec is approved when:
1. The user confirms (after reading this document).
2. The optional automated review pass returns no blocking issues.

Implementation begins via the `superpowers:writing-plans` skill, which produces a step-by-step implementation plan from this spec.
