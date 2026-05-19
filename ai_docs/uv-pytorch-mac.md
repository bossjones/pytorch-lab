# PyTorch + uv on macOS (Apple Silicon) — Reference

Source: https://docs.astral.sh/uv/guides/integration/pytorch/

## Key fact

> PyPI hosts CPU-only wheels for Windows and macOS, and GPU-accelerated wheels on
> Linux. PyTorch does not publish CUDA builds for macOS.

On **Apple Silicon**, the standard macOS wheel from PyPI **includes the MPS (Metal)
backend**. MPS is selected at *runtime* (`torch.backends.mps.is_available()`), not via
packaging. So for a Mac-only project the simplest, correct setup is just:

```bash
uv init --python 3.12
uv add torch torchvision
```

No `[[tool.uv.index]]`, no `[tool.uv.sources]`, no extra index URL. Those are only
needed for CUDA/ROCm or to force CPU-only wheels across platforms — out of scope here.

## Verifying MPS after install

```bash
uv run python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

Expected on this box: a 2.x version and `True`.

The repo already ships `contrib/is-mps-available.py` and
`contrib/does-matplotlib-work.py` — run them through uv as the environment smoke
check (this replaces the old `make env-works`):

```bash
uv run python contrib/is-mps-available.py
uv run python contrib/does-matplotlib-work.py
```

## Runtime device-selection caveats (relevant to this codebase)

`screennet/devices.py` and `screencropnet/devices.py` use legacy MPS detection:

- `torch.has_mps` — removed/deprecated in modern torch. Use
  `torch.backends.mps.is_available()` (and `torch.backends.mps.is_built()`).
- `torch.autocast("cuda")` hardcodes CUDA. On Mac use
  `torch.autocast(device_type=<resolved device>)`; autocast support differs per
  backend, so guard it.

These are fixed test-first in Phase 3.

## Modern stack target

| Package | Old (conda) | New (uv, latest stable) |
|---|---|---|
| torch | 1.12.1 | 2.x |
| torchvision | 0.13.1 | matching 0.x for the torch 2.x |
| torchmetrics | 0.7.0 | 1.x |
| numpy | 1.23 | 2.x (watch `np.float`/`np.int`/`np.bool` removals) |
| timm | unpinned (old) | latest (note `pretrained=`/`weights=` API) |
| albumentations | 1.3.0 | latest |

uv's resolver picks compatible torch/torchvision pairs automatically; only pin if a
conflict appears.
