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

The `_SinusoidalPE.pe` buffer's shape is part of the state_dict invariant —
variable-T support requires retraining or weight-init migration; existing
checkpoints assume `(1, 75, 1024)`.

## GRID-full speaker-independent split (notebook TODO-PRE-2)

3 speakers test / 2 val / 28 train, reported in a separate table. Implement
as `create_speaker_independent_split` alongside `create_paper_split`.

## LRS2 official splits (notebook TODO-PRE-3)

Use the official pre-train / train / val / test splits verbatim instead of
the current "all clips in <dir>" approach.

## CI checkpoint-load test

`test_checkpoint_compat.py` runs locally only when `LSN_CKPT_DIR` is set.
A CI job that downloads the smallest checkpoint (`identity_best_model.pt`,
~121 MB) from HF and runs the strict-load test would enforce the state_dict
invariant automatically. Out of scope for the conversion.
