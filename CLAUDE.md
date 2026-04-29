# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Capstone re-implementation of **LipSyncNet (LSN)**, a lip-reading model: 3D-CNN frontend + EfficientNet-B0 per-frame branch (fused) → temporal backend → linear classifier → CTC loss. Goal: reproduce the paper's GRID numbers (target ~96.7% word accuracy / 8.2% WER on a 450/550 paper-subset split, seed=42, speakers `s1`–`s5`) and ablate the temporal backend across three variants.

The repository is **not a Python package** — every file is a Jupyter notebook designed to run on **Colab** or **Kaggle** (T4 GPU). There is no `requirements.txt`, no `setup.py`, no test suite. Training happens online; checkpoints are pushed to HuggingFace Hub.

## Repository layout (just notebooks)

| Notebook | Role |
|---|---|
| `Copy of VSR_notebook_v1.ipynb` | **Preprocessing.** GRID + LRS2 → `.npz` clips of shape `(75, 46, 140, 1)`. Uses dlib 68-landmark mouth ROI crop, grayscale, normalize, pad/trim to 75 frames. Writes to `/content/drive/MyDrive/lipsyncnet_baseline/grid_processed`. |
| `LipSyncNet_Model_effnet.ipynb` | Earliest architecture *notes* (Keras-flavored, descriptive only). Superseded by the PyTorch implementations below. |
| `model_LSN_DRAFT.ipynb`, `(Ran)_model_LSN_DRAFT.ipynb`, `LSN Training.ipynb` | Successive drafts of the PyTorch model + training loop. |
| `LSN_TRAINING_EVAL.ipynb` | **Current source of truth.** End-to-end training (Kaggle "Save & Run All" ready) + GRID evaluation (Stages A–E) + LRS2 cross-dataset evaluation (Stages G–J). All other model notebooks are historical drafts. |

When asked to "modify the model" or "change training," default to editing `LSN_TRAINING_EVAL.ipynb` unless the user names another file.

## Architecture (the part that requires reading multiple cells to understand)

The codebase has **two parallel model classes**, both fed the same fused feature vector but wired differently. Don't conflate them:

- **`LipSyncNetPaper`** — paper-faithful. `Linear(70912 → 1024)` is **omitted**: the raw fused 70,912-dim vector is fed directly into LSTM-1. Hardcoded 2× BiLSTM(512) + optional `SelfAttentionBlock` (paper Figure 9, but absent from Table 1 — included by default and toggleable via `use_self_attn`).
- **`LipSyncNetVariant`** — modular ablation rig. The temporal block is selected from `_BACKEND_REGISTRY` by string name: `"bilstm"`, `"identity"`, `"transformer"`, `"transformer_perstream"`. The BiLSTM variant adds an input projection that the paper class does not — this is intentional, and **must be reported as a difference** in any results table.

Frontend pipeline used by both:
```
input (B, 75, 46, 140)
  ├─ Frontend3DCNN  → (B, 75, 8192)        # 4 Conv3D blocks, paper Table 1
  └─ EfficientNet   → (B, 75, 62720)       # B0, ImageNet weights, frame-by-frame
  → concat → fused (B, 75, 70912)
  → backend → classifier(out_dim → vocab+1=41) → log_softmax → permute (T, B, C) for CTC
```

Three deliberate divergences from the paper, documented inside cell-0 markdown — preserve them and the rationale if you touch the model:

1. **EfficientNet freeze policy.** Paper says 2,780,531 frozen params; no torchvision boundary matches. Code freezes stages 0–6 (2,878,156 params) — the closest clean boundary. Discrepancy is attributed to TF/Keras vs. torchvision weight layout.
2. **LSTM parameter count** in Table 1 is internally inconsistent with the stated dims; we follow the architecture, not the count.
3. **Self-attention block** appears in the figure but not Table 1 — implemented as `SelfAttentionBlock` (embed=1024, heads=8, post-norm residual, dropout=0.1). All design choices are forced by surrounding shapes; the docstring on the class enumerates them.

## Training contract

- **Loss:** `nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)`. `zero_infinity=True` is load-bearing — disabling it produces NaN gradients on short sequences.
- **Numerical precision:** Forward pass under `autocast(dtype=fp16)`, but cast `log_probs.float()` **before** CTCLoss. CTC NaNs immediately in fp16. This pattern is intentional, do not "simplify" it.
- **Gradient clipping:** `clip_grad_norm_(..., max_norm=5.0)`. BiLSTM-on-CTC explodes without this.
- **Effective batch:** `BATCH_SIZE=2` × `ACCUM_STEPS=4` = 8. The paper model is ~307M params, so per-step batch is memory-bound on a single T4. Loss is divided by `accum_steps` before backward.
- **Frozen BatchNorm:** `freeze_bn_stats(model)` is called at the top of every training epoch. Required because the EfficientNet stem is frozen but BN running stats would otherwise drift on tiny batches.
- **Multi-GPU:** disabled by default (`USE_MULTI_GPU = False`). With BATCH_SIZE=2 over 2 GPUs you get BN with batch=1 per device (unstable). Don't enable without first raising the batch and measuring >1.7× speedup.

## Checkpointing & resume (this is the load-bearing infra)

Training is designed to survive Kaggle's ~9h commit timeout. The **single source of truth** for run state is one self-describing dict:

```python
{"epoch", "model_state_dict", "optimizer_state_dict", "scaler_state_dict",
 "train_loss", "val_loss", "best_val_loss", "history"}
```

- `last_checkpoint.pt` is pushed to HuggingFace Hub (`ranro1/lipsyncnet-checkpoints`, subfolder per run, e.g. `run_identity_v1`) **every epoch**. `best_model.pt` is pushed only when val loss improves.
- Save is atomic-ish: write to `/tmp/checkpoint_tmp.pt` first, then `shutil.copy2` to the final path.
- Resume order: HF `last` → HF `best` → local `last` → local `best` → fresh.
- `best_val_loss` and full `history` live **inside** the checkpoint. Do not introduce a sidecar history file — losing the in-checkpoint copy reintroduces the "fresh start resets best score" bug that this scheme exists to prevent.
- `_model_state_dict` / `_load_into_model` unwrap `DataParallel` so checkpoints are portable across GPU configs.

When changing a run's identity (different backend, different speakers, etc.), change `HF_SUBFOLDER` — never overwrite an existing run's HF folder.

## Evaluation pipeline (`LSN_TRAINING_EVAL.ipynb`, lower half)

The eval flow is split into pure stages so each can be re-run independently:

- **Stage A** — load per-epoch `history` from each model's `last_checkpoint.pt` (NOT `best_model.pt` — `best` is truncated at the best-val epoch). Plot learning curves.
- **Stage B** — build each architecture via its `MODELS[name]["builder"]`, load `best_model.pt` with `strict=True` (mismatches crash here, not at inference).
- **Stage C** — rebuild the **same** 550-clip test split via `create_paper_split(npz_paths, speakers=['s1'..'s5'], seed=42)`. Define greedy + beam=100 CTC decoders.
- **Stage D** — run inference, write `predictions_{model_name}.json` (one row per clip: `{path, reference, beam}`). **Stage E reads from these JSONs**, never re-runs the model. This is the key separation.
- **Stage E** — compute CER / WER / word-acc / sentence-acc, emit `results_table.csv` (paper Table 5 reproduction) + `qualitative_examples.csv` (paper Table 6 reproduction).
- **Stages G–J** — repeat for LRS2 cross-dataset transfer. Framing is "do GRID-trained representations produce plausible English characters on unseen video," not "does the model generalize." WER is expected near 100%; CER is the primary metric.

The `MODELS` registry (one cell, near "Stage A") is the single place to add/remove a model from every downstream stage. Each entry needs `builder`, `history_file`, `weights_file`, `display_name`, `color`. Setting any path to `None` cleanly skips the model everywhere.

## Conventions worth preserving

- `seed=42` everywhere; `cudnn.benchmark=True` (training is non-deterministic — that's accepted; flip to `False` only if you need a bitwise-reproducible paper table).
- Vocab is `[' '] + a..z`, `BLANK_INDEX=0`, ids start at 1, total `NUM_CLASSES=41`. Don't reorder.
- Every clip is fixed at `T=75`, `H=46`, `W=140`, grayscale. The `_SinusoidalPE(max_len=75)` in `TransformerBackend` will need to be raised if LRS2 variable-length support is added (see TODO-PRE-4 in the cell-16 markdown).
- The cell-1 / cell-2 markdown blocks in `LSN_TRAINING_EVAL.ipynb` enumerate every paper-vs-implementation discrepancy and the rationale. Treat them as primary source, not commentary.

## Working in notebooks (operational notes)

- No build, no lint, no test runner. "Run" = open in Colab/Kaggle, mount Drive, install `huggingface_hub` (and on the preprocessing notebook: dlib + opencv + jiwer + the dlib 68-landmark predictor), set `HF_TOKEN` via `userdata.get`, run cells top-to-bottom.
- Training notebook expects either Colab Drive (`/content/drive/MyDrive/LSN_Data/...`) or Kaggle inputs (`/kaggle/input/datasets/runnscream/grid-data/...`). Both path forms appear in the cells; one is commented out depending on environment. Don't delete the alternate — just toggle the comment.
- `EVAL_OUTPUT_DIR` defaults to `/content/drive/MyDrive/LSN_Data/eval_outputs`. Trained checkpoints used by Stages A–E live in `/content/drive/MyDrive/LSN_Data/eval_models/{name}_{last_checkpoint,best_model}.pt`.
- For local edits to a notebook, prefer programmatic JSON edits over launching Jupyter — these notebooks are large (`LSN_TRAINING_EVAL.ipynb` is ~2 MB) and re-saving from a UI tends to churn unrelated metadata.
