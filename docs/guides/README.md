# User and Usage Guides

Practical, step-by-step guides for the pytorch-lab project. Each guide is runnable
start to finish — commands are taken directly from the Makefiles and source code.

---

## Guides

### [Getting Started](getting-started.md)

Start here. Covers prerequisites (uv, Python 3.12, Apple Silicon Mac), cloning,
syncing the environment, verifying MPS acceleration, setting the MPS fallback
environment variable, and orienting yourself in the project layout.

**Audience:** Anyone setting up the project for the first time.

---

### [ScreenNet: Image Classification](screennet-classification.md)

End-to-end walkthrough of the `screennet/` subproject. Fetches the
twitter/facebook/tiktok classification dataset, trains EfficientNet-B0 for 10 epochs,
views TensorBoard curves, runs prediction on a single image, and ranks the test set by
prediction error.

**Audience:** Developers working on the classification model.

**Key commands:** `make download-dataset`, `make unzip-dataset`, `make train`,
`make tensorboard`, `make predict`, `make worst-first`

---

### [ScreenCropNet: Bounding-Box Localization](screencropnet-localization.md)

End-to-end walkthrough of the `screencropnet/` subproject. Fetches all assets via the
idempotent PEP 723 fetcher, trains ObjLocModel for 40 or 378 epochs, resumes from a
checkpoint, evaluates on the test split, runs prediction with a bounding-box overlay,
and autocrop to the predicted content region.

**Audience:** Developers working on the localization model.

**Key commands:** `make fetch-assets`, `make train`, `make train_100`,
`make transfer_learning`, `make evaluate`, `make predict`, `make autocrop`,
`make tensorboard`

---

### [Development Workflow](development-workflow.md)

Day-to-day development tasks. Covers adding dependencies with `uv add` (never
`uv pip install`), formatting with ruff, linting, type-checking with pyright, running
pytest, and using the `inv` CLI for Invoke tasks (`inv local.clean`,
`inv local.jupyter`, etc.).

**Audience:** All contributors.

**Key commands:** `uv add`, `make format`, `make lint`, `make typecheck`, `make test`,
`make check`, `inv local.*`

---

### [Troubleshooting](troubleshooting.md)

A reference table of common errors with exact fixes. Covers MPS operator failures,
missing datasets, missing weights, import errors, ruff/pyright failures, and TensorBoard
data not appearing.

**Audience:** Anyone hitting an error.

---

## Quick command reference

```bash
# --- Setup ---
uv sync                      # create/update .venv from uv.lock  (make setup)
make env-works               # verify MPS + matplotlib
export PYTORCH_ENABLE_MPS_FALLBACK=1   # set once in ~/.zshrc

# --- Classification dataset ---
make download-dataset        # fetch twitter_facebook_tiktok.zip
make unzip-dataset           # extract and remove zip

# --- Localization assets ---
make fetch-assets            # all: dataset + weights + sample image
make download-localization-dataset  # dataset only

# --- screennet (from screennet/) ---
make train                   # EfficientNet-B0, 10 epochs
make tensorboard             # http://localhost:6006
make predict                 # inference on demo image
make worst-first             # rank test set by error

# --- screencropnet (from screencropnet/) ---
make train                   # ObjLocModel, 40 epochs
make train_100               # 378 epochs
make transfer_learning       # resume from 40-epoch checkpoint
make evaluate                # test-split evaluation
make predict                 # inference + bbox overlay
make autocrop                # crop to predicted content region
make get-best-model          # download ScreenCropNetV1_378_epochs.pth

# --- Quality (from repo root) ---
make format                  # ruff format .
make lint                    # ruff check .
make typecheck               # pyright
make test                    # pytest
make check                   # lint + typecheck + test

# --- Invoke tasks ---
inv local.clean              # remove .pyc / __pycache__
inv local.jupyter            # start Jupyter notebook
inv local.ipython            # start IPython shell
```
