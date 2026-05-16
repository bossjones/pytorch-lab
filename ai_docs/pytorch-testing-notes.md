# PyTorch Testing Conventions — Reference

Source: https://github.com/pytorch/pytorch/wiki/Running-and-writing-tests
(fetched 2026-05-16)

## What PyTorch itself does

- Uses **both unittest and pytest**; tests in `pytorch/test/`, shared helpers in
  `torch/testing/_internal/`.
- Run a single test: `pytest test_file.py::TestClass::test_name`
  (or `python test_file.py TestClass.test_name`).
- Useful pytest flags: `-x` stop on first failure, `-s` show stdout,
  `--lf` rerun last-failed.
- Parametrizes tests across **devices** (cpu/cuda) and **dtypes**
  (`instantiate_device_type_tests`, `@dtypes(...)`). Generated names look like
  `TestFooCPU.test_bar_cpu_float32`.
- Prefer **`make_tensor`** over `torch.randn` for deterministic test tensors; compare
  with **`assertEqual`** (handles tensor tolerances).
- Skip/condition decorators: `@skipIfNoCUDA`, `@slowTest`
  (gated by `PYTORCH_TEST_WITH_SLOW=1`), device-only via
  `PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu`.

## How we apply this to pytorch-lab (scope: pure units + import smoke)

This repo doesn't need PyTorch's device-matrix machinery. We adopt the *spirit*:

1. **Deterministic tensors.** Seed every test
   (`torch.manual_seed`, or call the repo's `devices.seed_everything`). Build inputs
   with explicit values, not unseeded `randn`, so IoU/shape assertions are stable.
2. **Tolerance-aware comparisons.** Use
   `torch.testing.assert_close(actual, expected, rtol=…, atol=…)` for float tensors
   (the public equivalent of PyTorch's internal `assertEqual`). Exact `==` only for
   shapes/ints.
3. **CPU-only by default.** All tests run on `torch.device("cpu")`. Don't require
   MPS/CUDA in assertions. For device-selection logic in `devices.py`, **mock**
   `torch.backends.mps.is_available` / `torch.cuda.is_available` with `pytest-mock`
   rather than depending on the host.
4. **Skip the expensive path.** No dataset downloads, no training loops. Mock heavy
   I/O (`requests`, filesystem walks, `torch.load`, model downloads) via the
   `mocker` fixture.
5. **One behavior per test, AAA layout** (Arrange / Act / Assert), named
   `test_<unit>_<behavior>`.
6. **Run focused tests fast**: `uv run pytest tests/test_devices.py -q`,
   `uv run pytest -k iou -x`.

## pytest-mock patterns we'll use

```python
def test_get_optimal_device_prefers_cuda(mocker):
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocker.patch("torch.backends.mps.is_available", return_value=False)
    # ... assert resolved device.type == "cuda"

def test_walk_through_dir(mocker, tmp_path):
    # tmp_path real fs is fine; mock network with mocker
    mocker.patch("requests.get", ...)
```

`mocker.patch` auto-undoes after each test (no manual teardown). Patch where the name
is *looked up*, not where it's defined (e.g. patch `screennet.devices.torch...`
target only if the module imported the symbol directly).

## Config (root `pyproject.toml`)

```toml
[tool.pytest.ini_options]
addopts = "--capture=no --disable-warnings"   # migrated from old setup.cfg
testpaths = ["tests"]
```
