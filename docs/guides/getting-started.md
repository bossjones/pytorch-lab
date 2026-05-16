# Getting Started

This guide walks you through cloning the repository, installing the environment, and
verifying that your Mac (Apple Silicon) is ready to train models.

**What you will be able to do after this guide:**

- Clone the repo and create a working virtualenv with `uv`
- Confirm that MPS acceleration and matplotlib are available
- Understand the two-subproject layout
- Know where datasets live and how they are managed

**Prerequisites:**

- macOS on Apple Silicon (M1 / M2 / M3 / M4 family). Intel Macs are not supported;
  PyTorch wheels for this project target MPS + CPU only.
- Homebrew available (`brew --version` succeeds)
- Git available (`git --version` succeeds)
- Approximately 3 GB free disk space for datasets and model weights

**Time estimate:** 10-15 minutes for a fresh machine.

---

## 1. Install uv

`uv` is the package and virtualenv manager used throughout this project. Never use
`uv pip install` — all dependency management goes through `uv add` / `uv add --dev`
and the lockfile.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify the installation:

```bash
uv --version
# uv 0.x.y (...)
```

> **Note:** `uv` manages a project-local `.venv` automatically. You do not need to
> activate it manually — every `uv run` command uses the correct environment.

---

## 2. Clone the repository

```bash
git clone https://github.com/bossjones/pytorch-lab.git
cd pytorch-lab
```

---

## 3. Create the virtualenv and install dependencies

```bash
make setup
```

This runs `uv sync`, which reads `uv.lock` and installs all pinned dependencies
(PyTorch, torchvision, timm, albumentations, etc.) into `.venv/`. On a clean machine
this downloads roughly 1-2 GB of wheels.

> **Warning:** Do not run `pip install`, `uv pip install`, or `conda install`. All
> dependency changes must go through `uv add` so that `uv.lock` stays in sync.

To verify the lockfile is up to date at any time:

```bash
make lock
```

---

## 4. Verify the environment

```bash
make env-works
```

This runs two smoke-check scripts:

- `contrib/is-mps-available.py` — confirms `torch.backends.mps.is_available()` returns
  `True` on your Apple Silicon Mac.
- `contrib/does-matplotlib-work.py` — imports matplotlib and renders a minimal figure.

Expected output (truncated):

```
MPS available: True
matplotlib OK
```

> **Warning:** If `MPS available: False` appears, you may be running on an Intel Mac or
> a VM without GPU passthrough. Training will fall back to CPU, which is much slower.

---

## 5. Set the MPS fallback environment variable

Some PyTorch operations are not yet implemented in the MPS backend. Without this
variable, those operations raise a runtime error. Add it to your shell profile so it
persists across sessions:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

To make this permanent, add the line to `~/.zshrc` (or `~/.zprofile`) and reload:

```bash
echo 'export PYTORCH_ENABLE_MPS_FALLBACK=1' >> ~/.zshrc
source ~/.zshrc
```

---

## 6. Project layout orientation

```
pytorch-lab/
├── screennet/          # Image classification: twitter / facebook / tiktok
│   ├── main.py         # Training and inference entry point
│   ├── devices.py      # MPS -> CPU device selection
│   ├── models/         # Saved .pth checkpoints (created on first train)
│   └── Makefile
│
├── screencropnet/      # Screenshot bounding-box localization
│   ├── main.py         # Training and inference entry point
│   ├── arch.py         # ObjLocModel (EfficientNet-B0 backbone)
│   ├── data_set.py     # ObjLocDataset with albumentations
│   ├── helpers.py      # IoU, NMS, mAP utilities
│   ├── models/         # Saved .pth checkpoints (created on first train)
│   └── Makefile
│
├── going_modular/      # Shared training utilities (installed as a package)
│   └── going_modular/
│       ├── data_setup.py
│       ├── engine.py
│       ├── model_builder.py
│       ├── predictions.py
│       └── utils.py
│
├── scratch/            # Gitignored: datasets and large assets
│   └── datasets/
│
├── contrib/            # Standalone scripts (dataset fetchers, smoke checks)
├── tasks/              # Invoke CLI tasks (inv local.*)
├── tests/              # pytest test suite
├── ai_docs/            # Background notes and asset provenance
├── pyproject.toml      # Project metadata, ruff, pyright, pytest config
├── uv.lock             # Pinned dependency lockfile (committed)
└── Makefile            # Root-level convenience targets
```

Key facts:

- `going_modular` is installed as a real package by `uv sync` (via
  `[tool.setuptools.package-dir]` in `pyproject.toml`). There are no `sys.path` hacks.
- `screennet` and `screencropnet` are each standalone subprojects with their own
  `Makefile`. Run their targets from inside those directories.
- `scratch/datasets/` is gitignored. Datasets must be fetched separately — see the
  subproject guides.

---

## Next steps

| Goal | Guide |
|---|---|
| Train a classifier (twitter / facebook / tiktok) | [screennet-classification.md](screennet-classification.md) |
| Train a bounding-box regression model | [screencropnet-localization.md](screencropnet-localization.md) |
| Add dependencies, run tests, lint | [development-workflow.md](development-workflow.md) |
| Fix a broken environment or MPS error | [troubleshooting.md](troubleshooting.md) |
