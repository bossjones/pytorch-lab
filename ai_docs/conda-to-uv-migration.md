# conda → uv Migration Map — Reference

## Command cheat sheet

| Old (conda/make) | New (uv) |
|---|---|
| `make link-conda-env` / `-intel` | (gone — single Mac project) |
| `conda env update` / `conda activate pytorch-lab3` | `uv sync` |
| `conda env update --prune` | `uv sync` (lock is authoritative) |
| `make conda-lock-env` (`env.yml.lock`) | `uv lock` → `uv.lock` |
| `pip install X` / add to environment.yml | `uv add X` |
| add a dev/test tool | `uv add --dev X` |
| `make env-works` | `uv run python contrib/is-mps-available.py && uv run python contrib/does-matplotlib-work.py` |
| `python main.py …` | `uv run python main.py …` |
| `inv ci.pytest` | `uv run pytest` |
| `black . && isort .` | `uv run ruff format .` |
| `pylint … main.py` | `uv run ruff check .` |
| `mypy … main.py` | `uv run pyright` |

## Real runtime dependency set (kept)

Derived from actual imports in `screennet/`, `screencropnet/`, `going_modular/`,
`helper_functions.py`:

`torch torchvision torchmetrics timm albumentations fastai numpy pandas
scikit-learn scipy opencv-python pillow matplotlib mlxtend torchinfo tensorboard
rich icecream better_exceptions tqdm pyfiglet watermark requests bpython`

Dev: `pytest pytest-mock ruff pyright invoke`

## Cruft removed (`git rm`) — InvokeAI, not this project

The `environments-and-requirements/` tree was almost entirely a different project
(InvokeAI: Stable Diffusion). None of `screennet`/`screencropnet`/`going_modular`
import any of it. Removed:

- `environments-and-requirements/` entire dir, incl.
  `environment-lin-cuda.yml`, `environment-lin-amd.yml`,
  `environment-lin-aarch64.yml`, `environment-win-cuda.yml`,
  `environment-mac.yml`, `environment-mac-intel.yml`,
  `requirements-base.txt`, `requirements-*.txt`
  (flask, streamlit, diffusers, gfpgan, clipseg, k-diffusion,
  taming-transformers, realesrgan, kornia, omegaconf, pytorch-lightning, …)
- root `requirements.txt` (stale 2020-era pins: matplotlib 3.2.2, Pillow 7.1.2)
- `installed_conda.txt`, `installed_pip.txt`
- `environment.yml` symlink, any `env.yml.lock` / `spec-file.txt`
- `screennet/setup.cfg`, `screencropnet/setup.cfg`
  (flake8/isort/pytest config → consolidated into root `pyproject.toml`)

## Version jumps

torch 1.12.1 → 2.x · torchvision 0.13.1 → matching · torchmetrics 0.7 → 1.x ·
numpy 1.23 → 2.x · transformers (unused by core code; dropped) · matplotlib 3.6 →
latest · timm old → latest.

## Deprecated APIs fixed test-first (Phase 3)

- `torch.has_mps` → `torch.backends.mps.is_available()`
  (`screennet/devices.py:152`, `screencropnet/devices.py:152`)
- `torch.autocast("cuda")` → `torch.autocast(device_type=...)`
  (`devices.py:116`, both)
- `torch.load(...)` → add `weights_only=True` for state-dict loads
  (`screennet/main.py` 1458/1462/1937/1951; `screencropnet/main.py`
  1761/1765/2244/2258)
- `timm.create_model(..., pretrained=True, ...)` — verify against installed timm
  (`screencropnet/arch.py:30`)
- numpy 2.x scalar-alias removals (`np.float`/`np.int`/`np.bool`/`np.object`) if any
- delete `sys.path.append` blocks; use real package imports
