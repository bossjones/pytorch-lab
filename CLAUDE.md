# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A PyTorch experimentation lab built around [Daniel Bourke's PyTorch course](https://github.com/mrdbourke/pytorch-deep-learning), containing two production-style model subprojects and modular training utilities.

## Environment Setup

This project is managed by **uv** (Mac-only: Apple Silicon MPS + CPU), Python 3.12.
Dependencies live in `pyproject.toml`; the lockfile is `uv.lock`. **Only add
dependencies via `uv add` / `uv add --dev` — never `uv pip install`.**

```bash
uv sync                      # create/update .venv from uv.lock
make env-works               # uv run is-mps-available.py + does-matplotlib-work.py
```

On Apple Silicon the default PyPI torch wheels include the MPS backend — no custom
index is needed. Background notes live in `ai_docs/`.

## Common Commands

### Root-level
```bash
uv sync                      # sync env from lockfile        (make setup)
uv lock                      # re-resolve dependencies        (make lock)
uv add <pkg>                 # add a runtime dependency
uv add --dev <pkg>           # add a dev/test dependency
make test                    # uv run pytest
make check                   # ruff check + pyright + pytest
inv local.clean              # remove .pyc / __pycache__
inv local.jupyter            # start Jupyter notebook
```

### screennet/ (image classification — twitter/facebook/tiktok)
```bash
cd screennet
make train                   # train EfficientNet-B0 for 10 epochs
make predict                 # run inference on a single image
make worst-first             # rank test-set images by highest loss
make tensorboard             # launch TensorBoard (logdir: runs/)
make format                  # uv run ruff format .
make lint                    # uv run ruff check .
make typecheck               # uv run pyright main.py
```

### screencropnet/ (object localization / bounding box regression)
```bash
cd screencropnet
make train                   # train ObjLocModel for 40 epochs
make train_100               # train for 378 epochs
make transfer_learning       # resume from a checkpoint
make evaluate                # evaluate with --test flag
make predict                 # run inference + save result to disk
make autocrop                # crop a screenshot using predicted bbox
make get-best-model          # download the pretrained .pth from Dropbox
make format                  # uv run ruff format .
make lint                    # uv run ruff check .
make typecheck               # uv run pyright main.py
make tensorboard             # launch TensorBoard
```

### Quality (run from repo root)
```bash
uv run ruff format .         # formatter (replaces black + isort)
uv run ruff check . --fix    # linter (replaces pylint + autoflake)
uv run pyright               # type checker (replaces mypy)
uv run pytest                # tests (pytest + pytest-mock)
```

All tool config lives in the root `pyproject.toml` (`[tool.ruff]`, `[tool.pyright]`,
`[tool.pytest.ini_options]` — `--capture=no --disable-warnings`).

## Architecture

### Two model subprojects

**`screennet/`** — image **classification** (3 classes: twitter / facebook / tiktok)
- `main.py` — monolithic training/inference entry point; imports `going_modular`,
  `helper_functions`, and `screennet.devices` as installed packages (no `sys.path` hacks)
- `devices.py` — device selection (CUDA → MPS → CPU) and MPS-specific workarounds

**`screencropnet/`** — screenshot **localization** (bounding box regression)
- `arch.py` / `ObjLocModel` — EfficientNet-B0 backbone from `timm`, outputs 4 bbox coordinates; trained with MSELoss
- `data_set.py` / `ObjLocDataset` — loads images + Pascal VOC CSV annotations; applies albumentations transforms including bbox augmentation
- `helpers.py` — IoU utilities (`find_intersection`, `find_jaccard_overlap`, `intersection_over_union`, `non_max_suppression`, `mean_average_precision`)
  - `intersection_over_union` / `iou_width_height` / `non_max_suppression` /
    `mean_average_precision` are ported from the Kaggle *YOLOv3 for Pascal VOC*
    notebook (`# SOURCE:` at `screencropnet/helpers.py:59`) — a YOLOv3-derived
    eval/post-processing layer only; `ObjLocModel` is plain single-box
    regression, not a YOLO network.
- `ml_types.py` — typed aliases (`ImageNdarrayBGR`, `ImageNdarrayHWC`, `TensorCHW`)
- `main.py` — training/inference entry point; package-qualified imports like screennet
- `devices.py` — identical device-selection module (not shared; each subproject is standalone)

### Shared utilities

**`going_modular/going_modular/`** — reusable training modules, installed as the
`going_modular` package (via `[tool.setuptools] package-dir`) and imported by both
`main.py` files:
- `data_setup.py` — dataset download + DataLoader creation
- `engine.py` — `train_step`, `test_step`, `train` loop
- `model_builder.py` — TinyVGG model definition
- `utils.py` — `save_model`, `plot_loss_curves`
- `predictions.py` — `pred_and_plot_image`

**`helper_functions.py`** (root) — standalone plotting/training helpers used in notebooks.

**`tasks/`** — Invoke CLI tasks (`inv <namespace>.<task>`), configured via `invoke.yaml`. Namespaces: `local`, `ci`.

### Device handling

Both subprojects share the same `devices.py` pattern: prefer CUDA → MPS (Apple Silicon
via `torch.backends.mps.is_available()`) → CPU. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` in
your shell for ops not yet supported on MPS.

### Data

Datasets are stored in `scratch/datasets/` (gitignored). Labels are Pascal VOC format CSV files (`bounding_box_list.csv`, `screencropnet/labels_pascal_temp.csv`). Use `make download-dataset` / `make unzip-dataset` at the root to fetch the twitter/facebook/tiktok **classification** dataset.

For the screencropnet **localization** assets (dataset, checkpoints, sample
image) use the idempotent PEP 723 fetcher:

```bash
make fetch-assets                    # dataset + all checkpoints + sample
make download-localization-dataset   # dataset only
uv run contrib/fetch_screencropnet_assets.py --weights --sample
```

Asset URLs/destinations/layout are documented in
`ai_docs/screencropnet-assets.md` (provenance recovered from the local-only
`contrib/*.ipynb` reference notebooks, which are git-ignored).

### Code style

- Line length: 88
- Formatter + linter: **ruff** (replaces black / isort / pylint / autoflake)
- Type checker: **pyright** (basic mode), replaces mypy
- All config is centralized in the root `pyproject.toml`
- Tests: `pytest` + `pytest-mock`, in `tests/` (see `ai_docs/pytorch-testing-notes.md`)
