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
