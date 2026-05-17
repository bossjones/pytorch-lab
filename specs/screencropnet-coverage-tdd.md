# screencropnet Coverage TDD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Canonical save location:** Per the `/plan` convention this spec should also live at `specs/screencropnet-coverage-tdd.md`. (Cannot be written there during plan mode — copy it there as the first execution action; see Task 0.)

**Goal:** Raise test coverage of `screencropnet/*.py` by mirroring the existing screennet Tier-2 characterization pattern, fixing 4 real source bugs via red-green TDD, and refactoring 3 standalone scripts so their pure logic becomes testable.

**Architecture:** Clone the proven two-file screennet pattern (`tests/test_screennet_main_units.py` + `tests/test_screennet_main_compute.py`, added in commit `33917a4`) into `screencropnet` equivalents, and add focused test files for the screencropnet-only modules (`helpers.py` NMS/mAP, `image_utils.py`, `data_set.py`). Genuine bugs and script-extraction work use strict red-green TDD; coverage of already-correct existing code uses characterization tests where the *coverage delta* is the "RED" proof the test exercises new lines.

**Tech Stack:** pytest 9 + pytest-mock + pytest-cov (branch coverage), torch (CPU-only), cv2, albumentations, pandas, PIL, matplotlib (Agg). No new dependencies — all required libs are already in `pyproject.toml`.

---

## Context

`screencropnet/` (the bounding-box localization subproject) is largely untested. A stale local `htmlcov` (pre-dates the screennet Tier-2 push, regenerate before trusting) suggests: `main.py` ~23% (896 stmts — by far the biggest gap), `helpers.py` ~37%, `data_set.py` ~33%, `image_utils.py` ~42%, plus five standalone scripts at 0%. `arch.py` and the IoU subset of `helpers.py` are already covered (`tests/test_arch.py`, `tests/test_helpers_iou.py`); `devices.py` is already covered by the parametrized `tests/test_devices.py` (runs against both `screencropnet.devices` and `screennet.devices`) — **do not re-test arch or devices**.

Commit `33917a4 "test(screennet): add Tier-2 characterization tests for coverage push"` established the exact pattern to replicate: two files splitting "pure deterministic units" from "small-model compute/plot helpers", CPU-only, no network, no training, no custom pytest markers. This plan applies that pattern to `screencropnet` and additionally (per user decisions) **fixes** the source bugs found during exploration and **extracts** pure logic from import-unsafe scripts.

User decisions driving this plan:
- **Scope:** Tier 1 + Tier 2 (pure functions + light-fixture tests). No GPU/network/training.
- **Scripts:** Extract + test pure pieces.
- **Source bugs:** Fix as part of this work (true red-green TDD).
- **Definition of done:** Match the screennet pattern (no hard % gate) — every Tier-1/Tier-2 testable unit has a test.

## Objective

Every Tier-1/Tier-2 testable unit in `screencropnet/*.py` has a passing test; 4 named source bugs are fixed with regression tests proving the fix; 3 scripts are import-safe with their pure logic covered; `make check` (ruff + pyright + pytest) and `make test-cov` are green.

## Problem Statement

The localization subproject has near-zero automated test coverage, contains at least 4 latent crash bugs (including one triggered by a function's own default argument), and ships scratch scripts that execute side effects at import time. There is no regression safety net for the bbox-encoding math, IoU/NMS/mAP eval layer, dataset loader, or image utilities.

## Solution Approach

Replicate the validated screennet Tier-2 template verbatim where logic is shared (`main.py` has ~1:1 mirrors of screennet helpers), add focused test files for screencropnet-only modules, and treat the three categories of work with the appropriate TDD rigor:

1. **Bug fixes (strict red-green):** write a test asserting *correct* behavior → run, watch it fail with the documented crash → fix source minimally → run, watch it pass.
2. **Script extraction (strict red-green):** write a test that imports the pure function → run, watch it fail (import-time side effect / `UnboundLocalError`) → add `if __name__ == "__main__":` guard / minimal refactor → run, watch it pass.
3. **Characterization (coverage-delta as RED):** for already-correct existing code, write the contract test → run with `--cov`; the previously-uncovered lines now showing as covered is the proof the test exercises real behavior. (Mirrors the existing repo `SUSPECTED BUG` / known-value convention.)

## Relevant Files

Use these files to complete the task:

- `tests/conftest.py` — REUSE: autouse `_deterministic` fixture (`torch.manual_seed(0)`) and `cpu` fixture (`torch.device("cpu")`). Do not add new conftest fixtures unless shared by 2+ files.
- `tests/test_screennet_main_units.py` — TEMPLATE to clone for `tests/test_screencropnet_main_units.py`.
- `tests/test_screennet_main_compute.py` — TEMPLATE to clone for `tests/test_screencropnet_main_compute.py` (note its `import matplotlib; matplotlib.use("Agg")` BEFORE importing the main module).
- `tests/test_devices.py` — PATTERN reference for parametrizing one test file over both twin modules (`@pytest.fixture(params=[...])` + `importlib.import_module`). Reuse this exact idiom for `test_errors.py`.
- `tests/test_helpers_iou.py` — existing IoU known-value tests; do NOT duplicate. New NMS/mAP tests go in a new file.
- `tests/fixtures/00006_twitter.PNG` — existing 1179×2556 RGBA PNG fixture (loads fine via `cv2.imread`). Use only for one "real-asset smoke test"; synthesize tiny images in `tmp_path` for everything else.
- `screencropnet/helpers.py` — `non_max_suppression`, `mean_average_precision` (uncovered); BUG FIX targets: `intersection_over_union` (UnboundLocalError on bad `box_format`), `mean_average_precision` (ZeroDivisionError when all classes empty).
- `screencropnet/image_utils.py` — `convert_image_numpy_array_to_tensor` (pure), `opencv_read_and_convert_image`, `safe_read_image`; BUG FIX target: `load_and_transform_image_for_prediction` (UnboundLocalError when `transform=None`, the default).
- `screencropnet/data_set.py` — `ObjLocDataset.__init__/__len__/__getitem__` (cv2-based).
- `screencropnet/main.py` — large module; pure helpers + compute helpers mirror screennet; BUG FIX target: `pred_and_store` (UnboundLocalError on empty `paths`).
- `screencropnet/pascal_to_csv.py` — calls `main()` at module level (import-time side effect); `xml_to_csv(path)` is the pure piece to expose.
- `screencropnet/try_predict.py` — `get_bbox(...)` pure piece; heavy import-time work to guard.
- `screencropnet/show_bbox_example.py` — duplicate `df_to_table`; top-level `pd.read_csv` to guard.
- `pyproject.toml` — `[tool.coverage.run]` (no `omit` today); add an `omit` for the two hyphenated, logic-free color scripts only.

### New Files
- `tests/fixtures/sample_voc.xml` — minimal Pascal VOC annotation XML for `xml_to_csv`.
- `tests/test_helpers_nms_map.py` — NMS + mAP tests and the two `helpers.py` bug-fix regression tests.
- `tests/test_image_utils.py` — `image_utils.py` tests + the `load_and_transform_image_for_prediction` bug-fix test.
- `tests/test_data_set.py` — `ObjLocDataset` tests.
- `tests/test_screencropnet_main_units.py` — clone of the screennet units template.
- `tests/test_screencropnet_main_compute.py` — clone of the screennet compute template + `pred_and_store` bug-fix test.
- `tests/test_screencropnet_scripts.py` — script-extraction tests (`xml_to_csv`, `get_bbox`, import-safety of `show_bbox_example`).

## Implementation Phases

### Phase 1: Foundation
Copy spec to `specs/`, capture a baseline coverage number, add the VOC fixture.

### Phase 2: Core Implementation
Per-module TDD: helpers NMS/mAP + 2 bug fixes → image_utils + 1 bug fix → data_set → errors twin parametrization → main units clone → main compute clone + 1 bug fix → script extraction (3 scripts + 1 omit decision).

### Phase 3: Integration & Polish
Full-suite + coverage run, ruff/pyright clean, regression check, final commit + handoff.

## Step by Step Tasks

IMPORTANT: Execute every step in order, top to bottom. Conventions for EVERY new test file: start with `from __future__ import annotations`; import the module under test as a short alias (`import screencropnet.main as m`); one behavior per test; AAA layout; `# ----` section banners naming the unit; `torch.testing.assert_close(actual, expected, rtol=0, atol=1e-4)` for float tensors; exact `==` only for ints/shapes; no custom pytest markers; footer `if __name__ == "__main__":\n    raise SystemExit(pytest.main([__file__, "-v"]))`. Commit after every task.

### 1. Foundation: spec copy, baseline, VOC fixture

- [ ] **Step 1.1: Copy this plan to the canonical spec location**

```bash
cp /Users/bossjones/.claude/plans/i-want-to-create-floating-island.md \
   /Users/bossjones/dev/bossjones/pytorch-lab/specs/screencropnet-coverage-tdd.md
```

- [ ] **Step 1.2: Capture baseline coverage (proves the RED for characterization work)**

Run: `cd /Users/bossjones/dev/bossjones/pytorch-lab && uv run pytest --cov=screencropnet --cov-report=term-missing 2>&1 | tee /tmp/screencropnet-cov-baseline.txt`
Expected: a term-missing table; record the starting `screencropnet/*.py` numbers. This file is the before/after evidence.

- [ ] **Step 1.3: Create the minimal Pascal VOC fixture**

Create `tests/fixtures/sample_voc.xml`:

```xml
<annotation>
  <folder>fixtures</folder>
  <filename>00006_twitter.PNG</filename>
  <path>00006_twitter.PNG</path>
  <size>
    <width>1179</width>
    <height>2556</height>
    <depth>3</depth>
  </size>
  <object>
    <name>screen</name>
    <bndbox>
      <xmin>10</xmin>
      <ymin>20</ymin>
      <xmax>110</xmax>
      <ymax>220</ymax>
    </bndbox>
  </object>
</annotation>
```

- [ ] **Step 1.4: Commit**

```bash
git add specs/screencropnet-coverage-tdd.md tests/fixtures/sample_voc.xml
git commit -m "test(screencropnet): add coverage spec + VOC fixture"
```

### 2. helpers.py — NMS/mAP characterization + 2 bug fixes (strict red-green)

**Files:**
- Create: `tests/test_helpers_nms_map.py`
- Modify: `screencropnet/helpers.py` (`intersection_over_union`, `mean_average_precision`)

- [ ] **Step 2.1: Write the failing test for the `intersection_over_union` bad-format bug**

Create `tests/test_helpers_nms_map.py`:

```python
from __future__ import annotations

import pytest
import torch

import screencropnet.helpers as h


# ---- intersection_over_union: invalid box_format ----
def test_iou_invalid_box_format_raises_valueerror() -> None:
    a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    b = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    with pytest.raises(ValueError):
        h.intersection_over_union(a, b, box_format="bogus")
```

- [ ] **Step 2.2: Run it, watch it fail for the wrong reason (the bug)**

Run: `uv run pytest tests/test_helpers_nms_map.py::test_iou_invalid_box_format_raises_valueerror -v`
Expected: FAIL — raises `UnboundLocalError` (not `ValueError`), confirming the bug.

- [ ] **Step 2.3: Fix `intersection_over_union` to raise `ValueError` on unknown format**

In `screencropnet/helpers.py`, in `intersection_over_union`, after the `if box_format == "midpoint":` / `if box_format == "corners":` branches and before `x1 = torch.max(...)`, add:

```python
    else:
        raise ValueError(f"Unknown box_format: {box_format!r}")
```

(Place the `else` on the second/`corners` `if` so both known formats are handled and anything else raises. Verify the surrounding indentation matches the existing function.)

- [ ] **Step 2.4: Run it, watch it pass**

Run: `uv run pytest tests/test_helpers_nms_map.py::test_iou_invalid_box_format_raises_valueerror -v`
Expected: PASS.

- [ ] **Step 2.5: Write the failing test for the `mean_average_precision` empty-classes bug**

Append to `tests/test_helpers_nms_map.py`:

```python
# ---- mean_average_precision: no true boxes for any class ----
def test_map_returns_zero_when_no_true_boxes() -> None:
    # pred/true box rows: [train_idx, class, prob, x1, y1, x2, y2]
    result = h.mean_average_precision(
        pred_boxes=[],
        true_boxes=[],
        iou_threshold=0.5,
        box_format="midpoint",
        num_classes=1,
    )
    assert float(result) == 0.0
```

- [ ] **Step 2.6: Run it, watch it fail**

Run: `uv run pytest tests/test_helpers_nms_map.py::test_map_returns_zero_when_no_true_boxes -v`
Expected: FAIL — `ZeroDivisionError` from `sum(average_precisions) / len(average_precisions)`.

- [ ] **Step 2.7: Fix `mean_average_precision`**

In `screencropnet/helpers.py`, change the final return of `mean_average_precision` from `return sum(average_precisions) / len(average_precisions)` to:

```python
    if len(average_precisions) == 0:
        return 0.0
    return sum(average_precisions) / len(average_precisions)
```

- [ ] **Step 2.8: Run it, watch it pass**

Run: `uv run pytest tests/test_helpers_nms_map.py::test_map_returns_zero_when_no_true_boxes -v`
Expected: PASS.

- [ ] **Step 2.9: Add NMS + mAP characterization tests**

Append to `tests/test_helpers_nms_map.py`:

```python
# ---- non_max_suppression: known-value contract ----
def test_nms_suppresses_overlapping_lower_score_box() -> None:
    # boxes: [class, prob, x1, y1, x2, y2], corners format
    boxes = [
        [0, 0.9, 0.0, 0.0, 1.0, 1.0],
        [0, 0.8, 0.0, 0.0, 0.9, 0.9],  # high IoU with the 0.9 box -> suppressed
        [0, 0.7, 5.0, 5.0, 6.0, 6.0],  # disjoint -> kept
    ]
    kept = h.non_max_suppression(
        boxes, iou_threshold=0.5, threshold=0.5, box_format="corners"
    )
    scores = sorted(b[1] for b in kept)
    assert scores == [0.7, 0.9]


def test_nms_drops_boxes_below_confidence_threshold() -> None:
    boxes = [
        [0, 0.9, 0.0, 0.0, 1.0, 1.0],
        [0, 0.2, 5.0, 5.0, 6.0, 6.0],  # below threshold -> dropped
    ]
    kept = h.non_max_suppression(
        boxes, iou_threshold=0.5, threshold=0.5, box_format="corners"
    )
    assert [b[1] for b in kept] == [0.9]


def test_nms_keeps_different_classes_even_when_overlapping() -> None:
    boxes = [
        [0, 0.9, 0.0, 0.0, 1.0, 1.0],
        [1, 0.85, 0.0, 0.0, 1.0, 1.0],  # same box, different class -> kept
    ]
    kept = h.non_max_suppression(
        boxes, iou_threshold=0.5, threshold=0.5, box_format="corners"
    )
    assert sorted(b[0] for b in kept) == [0, 1]


# ---- mean_average_precision: perfect prediction -> AP == 1.0 ----
def test_map_perfect_prediction_is_one() -> None:
    box = [0, 0, 0.9, 0.5, 0.5, 1.0, 1.0]  # [train_idx, class, prob, mid x,y,w,h]
    result = h.mean_average_precision(
        pred_boxes=[box],
        true_boxes=[box],
        iou_threshold=0.5,
        box_format="midpoint",
        num_classes=1,
    )
    torch.testing.assert_close(
        torch.as_tensor(float(result)), torch.tensor(1.0), rtol=0, atol=1e-4
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2.10: Run the file, watch all pass**

Run: `uv run pytest tests/test_helpers_nms_map.py -v`
Expected: all PASS, pristine output.

- [ ] **Step 2.11: Commit**

```bash
git add tests/test_helpers_nms_map.py screencropnet/helpers.py
git commit -m "fix(screencropnet): guard helpers iou/map crashes + cover NMS/mAP"
```

### 3. image_utils.py — pure conversion + I/O + 1 bug fix (strict red-green)

**Files:**
- Create: `tests/test_image_utils.py`
- Modify: `screencropnet/image_utils.py` (`load_and_transform_image_for_prediction`)

- [ ] **Step 3.1: Write the failing test for the `transform=None` default bug**

Create `tests/test_image_utils.py`:

```python
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

import screencropnet.image_utils as iu

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "00006_twitter.PNG"


def _write_tmp_image(tmp_path: Path) -> Path:
    arr = np.zeros((4, 6, 3), dtype=np.uint8)
    arr[:, :, 0] = 255  # blue channel in BGR
    out = tmp_path / "tiny.png"
    cv2.imwrite(str(out), arr)
    return out


# ---- load_and_transform_image_for_prediction: default transform=None ----
def test_load_and_transform_with_default_transform(tmp_path: Path) -> None:
    img_path = _write_tmp_image(tmp_path)
    img, img_transform = iu.load_and_transform_image_for_prediction(str(img_path))
    assert isinstance(img, torch.Tensor)
    assert img_transform is not None
```

- [ ] **Step 3.2: Run it, watch it fail**

Run: `uv run pytest tests/test_image_utils.py::test_load_and_transform_with_default_transform -v`
Expected: FAIL — `UnboundLocalError: ... 'img_transform'` (the default code path never assigns it).

- [ ] **Step 3.3: Fix `load_and_transform_image_for_prediction`**

In `screencropnet/image_utils.py`, give the function a real default transform instead of leaving `img_transform` unbound. At the top of the function body, before the `if transform is not None:` branch, add a default-construction branch so `img_transform` is always assigned. Concretely, replace the existing `if transform is not None:` block so the structure is:

```python
    if transform is None:
        img_transform = A.Compose(
            [A.Resize(img_size, img_size)],
        )
    else:
        img_transform = transform
```

(Keep the rest of the function — the image read and the application of `img_transform` — unchanged. `A` is the already-imported `albumentations` alias.)

- [ ] **Step 3.4: Run it, watch it pass**

Run: `uv run pytest tests/test_image_utils.py::test_load_and_transform_with_default_transform -v`
Expected: PASS.

- [ ] **Step 3.5: Add pure + I/O characterization tests**

Append to `tests/test_image_utils.py`:

```python
# ---- convert_image_numpy_array_to_tensor: pure ----
def test_convert_numpy_to_tensor_shape_and_scale() -> None:
    arr = np.full((4, 6, 3), 255, dtype=np.uint8)  # HWC
    t = iu.convert_image_numpy_array_to_tensor(arr)
    assert t.shape == (3, 4, 6)  # CHW
    torch.testing.assert_close(
        t.max(), torch.tensor(1.0), rtol=0, atol=1e-6
    )


# ---- opencv_read_and_convert_image: tmp file ----
def test_opencv_read_and_convert_returns_rgb_ndarray(tmp_path: Path) -> None:
    img_path = _write_tmp_image(tmp_path)
    out = iu.opencv_read_and_convert_image(str(img_path))
    assert isinstance(out, np.ndarray)
    assert out.shape == (4, 6, 3)
    # BGR(255,0,0) -> RGB: red channel (index 0) should be 0, blue (2) 255
    assert out[0, 0, 2] == 255


# ---- safe_read_image: tmp file -> tensor ----
def test_safe_read_image_returns_chw_tensor(tmp_path: Path) -> None:
    img_path = _write_tmp_image(tmp_path)
    t = iu.safe_read_image(str(img_path))
    assert isinstance(t, torch.Tensor)
    assert t.shape[0] == 3


# ---- real-asset smoke test ----
def test_opencv_read_real_fixture() -> None:
    assert FIXTURE.exists(), f"Fixture not found: {FIXTURE}"
    out = iu.opencv_read_and_convert_image(str(FIXTURE))
    assert out.shape[2] == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 3.6: Run the file, watch all pass**

Run: `uv run pytest tests/test_image_utils.py -v`
Expected: all PASS.

- [ ] **Step 3.7: Commit**

```bash
git add tests/test_image_utils.py screencropnet/image_utils.py
git commit -m "fix(screencropnet): default transform in image_utils + cover conversions"
```

### 4. data_set.py — ObjLocDataset

**Files:**
- Create: `tests/test_data_set.py`

- [ ] **Step 4.1: Write the tests (characterization — coverage delta is the RED)**

Create `tests/test_data_set.py`:

```python
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from screencropnet.data_set import ObjLocDataset


def _df_with_image(tmp_path: Path) -> pd.DataFrame:
    arr = np.zeros((20, 30, 3), dtype=np.uint8)
    img_path = tmp_path / "img0.png"
    cv2.imwrite(str(img_path), arr)
    return pd.DataFrame(
        [{"img_path": str(img_path), "xmin": 1, "ymin": 2, "xmax": 11, "ymax": 12}]
    )


# ---- __len__ / __init__: pure ----
def test_dataset_len_matches_dataframe(tmp_path: Path) -> None:
    df = _df_with_image(tmp_path)
    ds = ObjLocDataset(df=df, transform=None, root_dir="")
    assert len(ds) == 1


# ---- __getitem__: real tmp image ----
def test_dataset_getitem_returns_image_and_bbox(tmp_path: Path) -> None:
    df = _df_with_image(tmp_path)
    ds = ObjLocDataset(df=df, transform=None, root_dir="")
    img, bbox = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape[0] == 3            # CHW
    assert tuple(int(v) for v in bbox) == (1, 2, 11, 12)
    torch.testing.assert_close(
        img.max(), torch.tensor(0.0), rtol=0, atol=1e-6
    )  # all-zero image stays 0 after /255


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 4.2: Run, confirm pass, confirm coverage delta**

Run: `uv run pytest tests/test_data_set.py --cov=screencropnet.data_set --cov-report=term-missing -v`
Expected: all PASS; `screencropnet/data_set.py` coverage jumps well above the baseline recorded in Step 1.2.

- [ ] **Step 4.3: Commit**

```bash
git add tests/test_data_set.py
git commit -m "test(screencropnet): cover ObjLocDataset len/getitem"
```

### 5. errors.py — parametrize the existing test over both twin modules

**Files:**
- Modify: `tests/test_errors.py`

- [ ] **Step 5.1: Read both files to confirm the twin is identical**

Read `tests/test_errors.py` and `screencropnet/errors.py`. Confirm `screencropnet.errors.run` has the same signature/behavior as `screennet.errors.run` (it does — identical twin).

- [ ] **Step 5.2: Parametrize `test_errors.py` over both modules**

Modify `tests/test_errors.py` to use the `test_devices.py` idiom: add at top `import importlib` and a fixture

```python
import pytest

ERRORS_MODULES = ["screennet.errors", "screencropnet.errors"]


@pytest.fixture(params=ERRORS_MODULES)
def errors_mod(request):
    import importlib

    return importlib.import_module(request.param)
```

Then change every existing test that imports `screennet.errors` directly to take the `errors_mod` fixture and call `errors_mod.run(...)` instead. Keep assertions identical.

- [ ] **Step 5.3: Run, watch all pass for both params**

Run: `uv run pytest tests/test_errors.py -v`
Expected: each test now reports `[screennet.errors]` and `[screencropnet.errors]` variants, all PASS.

- [ ] **Step 5.4: Commit**

```bash
git add tests/test_errors.py
git commit -m "test: parametrize errors tests over screennet+screencropnet twins"
```

### 6. main.py — units file (clone screennet template)

**Files:**
- Create: `tests/test_screencropnet_main_units.py`

- [ ] **Step 6.1: Clone the screennet units template**

Copy `tests/test_screennet_main_units.py` → `tests/test_screencropnet_main_units.py`. In the new file, replace every `import screennet.main as m` / `from screennet.main import ...` with the `screencropnet.main` equivalent. Update the module docstring to say `screencropnet`.

- [ ] **Step 6.2: Run, triage failures, prune non-existent units**

Run: `uv run pytest tests/test_screencropnet_main_units.py -v`
Expected: most PASS (mirrored helpers); some FAIL/ERROR for screennet-only functions absent from `screencropnet.main`. For each failure, read `screencropnet/main.py` to confirm the function genuinely doesn't exist, then **delete that test** (it tests a function not in this module). Do not weaken assertions.

- [ ] **Step 6.3: Add screencropnet-only bbox-encoding round-trip tests**

Append to `tests/test_screencropnet_main_units.py`:

```python
# ---- screencropnet-only: bbox coordinate encodings ----
import torch  # noqa: E402  (already imported at top in the template; keep one)


def test_xy_to_cxcy_to_xy_round_trip() -> None:
    xy = torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 4.0, 6.0, 10.0]])
    round_trip = m.cxcy_to_xy(m.xy_to_cxcy(xy))
    torch.testing.assert_close(round_trip, xy, rtol=0, atol=1e-5)


def test_gcxgcy_round_trip_with_priors() -> None:
    cxcy = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
    priors = torch.tensor([[0.5, 0.5, 2.0, 2.0]])
    encoded = m.cxcy_to_gcxgcy(cxcy, priors)
    decoded = m.gcxgcy_to_cxcy(encoded, priors)
    torch.testing.assert_close(decoded, cxcy, rtol=0, atol=1e-5)


def test_dimensions_enum_values() -> None:
    assert int(m.Dimensions.HEIGHT) == 224
    assert int(m.Dimensions.WIDTH) == 224
```

(If the template already imports `torch` at module top, do not add the duplicate import — keep the file's existing single import.)

- [ ] **Step 6.4: Run the file, watch all pass**

Run: `uv run pytest tests/test_screencropnet_main_units.py -v`
Expected: all PASS, pristine output.

- [ ] **Step 6.5: Commit**

```bash
git add tests/test_screencropnet_main_units.py
git commit -m "test(screencropnet): Tier-2 unit tests for main.py pure helpers"
```

### 7. main.py — compute file (clone screennet template) + pred_and_store bug fix

**Files:**
- Create: `tests/test_screencropnet_main_compute.py`
- Modify: `screencropnet/main.py` (`pred_and_store`)

- [ ] **Step 7.1: Clone the screennet compute template**

Copy `tests/test_screennet_main_compute.py` → `tests/test_screencropnet_main_compute.py`. Preserve the `import matplotlib; matplotlib.use("Agg")` lines that appear BEFORE the `screennet.main` import; repoint all `screennet.main` references to `screencropnet.main`. Update the docstring.

- [ ] **Step 7.2: Run, triage, prune non-existent units (same procedure as 6.2)**

Run: `uv run pytest tests/test_screencropnet_main_compute.py -v`
For each failure caused by a screennet-only function missing from `screencropnet.main`, confirm by reading source then delete that test. Keep all tests for functions that DO exist.

- [ ] **Step 7.3: Write the failing test for the `pred_and_store` empty-paths bug**

Append to `tests/test_screencropnet_main_compute.py`:

```python
# ---- pred_and_store: empty paths must not crash ----
def test_pred_and_store_empty_paths_returns_empty() -> None:
    result = m.pred_and_store(
        paths=[],
        model=None,
        transform=None,
        class_names=[],
        device="cpu",
    )
    assert result == [] or result is None
```

(If `pred_and_store`'s real signature differs, read `screencropnet/main.py` and adjust the call to pass the minimal arguments with an empty `paths` list — the behavior under test is "empty input → no `UnboundLocalError`".)

- [ ] **Step 7.4: Run it, watch it fail**

Run: `uv run pytest tests/test_screencropnet_main_compute.py::test_pred_and_store_empty_paths_returns_empty -v`
Expected: FAIL — `UnboundLocalError` on `fullsize_bboxes` (referenced after a loop that never ran).

- [ ] **Step 7.5: Fix `pred_and_store`**

In `screencropnet/main.py`, in `pred_and_store`, initialize the accumulator(s) referenced after the loop *before* the loop (e.g. `fullsize_bboxes = []` and any sibling list referenced post-loop), so an empty `paths` yields an empty result instead of `UnboundLocalError`. Make the minimal change only.

- [ ] **Step 7.6: Run it, watch it pass**

Run: `uv run pytest tests/test_screencropnet_main_compute.py::test_pred_and_store_empty_paths_returns_empty -v`
Expected: PASS.

- [ ] **Step 7.7: Run the whole compute file**

Run: `uv run pytest tests/test_screencropnet_main_compute.py -v`
Expected: all PASS.

- [ ] **Step 7.8: Commit**

```bash
git add tests/test_screencropnet_main_compute.py screencropnet/main.py
git commit -m "fix(screencropnet): guard pred_and_store empty paths + Tier-2 compute tests"
```

### 8. Script extraction — make pure logic testable (strict red-green)

**Files:**
- Create: `tests/test_screencropnet_scripts.py`
- Modify: `screencropnet/pascal_to_csv.py`, `screencropnet/try_predict.py`, `screencropnet/show_bbox_example.py`
- Modify: `pyproject.toml` (coverage `omit` for the two logic-free hyphenated scripts)

- [ ] **Step 8.1: Write the failing import/extraction tests**

Create `tests/test_screencropnet_scripts.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VOC_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "sample_voc.xml"


# ---- pascal_to_csv.xml_to_csv: pure, needs import-safe module ----
def test_xml_to_csv_parses_voc(tmp_path: Path) -> None:
    import shutil

    voc_dir = tmp_path / "voc"
    voc_dir.mkdir()
    shutil.copy(VOC_FIXTURE, voc_dir / "sample_voc.xml")

    from screencropnet.pascal_to_csv import xml_to_csv

    df = xml_to_csv(str(voc_dir))
    assert len(df) == 1
    row = df.iloc[0]
    assert int(row["xmin"]) == 10
    assert int(row["ymax"]) == 220


# ---- try_predict.get_bbox: pure matplotlib patch ----
def test_get_bbox_returns_rectangle_patch() -> None:
    from matplotlib.patches import Rectangle

    from screencropnet.try_predict import get_bbox

    rect = get_bbox([10, 20, 110, 220], col="red", bbox_format="pascal_voc")
    assert isinstance(rect, Rectangle)


# ---- show_bbox_example imports without side effects ----
def test_show_bbox_example_import_is_safe() -> None:
    import importlib

    mod = importlib.import_module("screencropnet.show_bbox_example")
    assert hasattr(mod, "df_to_table")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 8.2: Run it, watch it fail**

Run: `uv run pytest tests/test_screencropnet_scripts.py -v`
Expected: FAIL — importing these modules triggers import-time `main()` / `pd.read_csv` of hardcoded paths / heavy work (not the assertions).

- [ ] **Step 8.3: Guard `pascal_to_csv.py`**

In `screencropnet/pascal_to_csv.py`, the module-level call to `main()` (around line 60) must be wrapped:

```python
if __name__ == "__main__":
    main()
```

Leave `xml_to_csv` and `main` definitions unchanged.

- [ ] **Step 8.4: Guard `try_predict.py`**

In `screencropnet/try_predict.py`, ensure nothing heavy runs at import: the prediction/`bpdb`/checkpoint-loading block must be inside `if __name__ == "__main__":`. Keep the version `assert`s and `get_bbox` at module scope (asserts pass in the synced env; `get_bbox` is the pure piece). If `torch.set_num_threads(1)` is module-level, leave it (harmless, fast).

- [ ] **Step 8.5: Guard `show_bbox_example.py` and dedup `df_to_table`**

In `screencropnet/show_bbox_example.py`, move the top-level `pd.read_csv(CSV_FILE)` + the `try/except` rendering block into `if __name__ == "__main__":`. Replace its duplicate `df_to_table` definition with a re-export: `from screencropnet.main import df_to_table` so the single canonical implementation (already covered by Task 6) is used and the symbol still exists for the import-safety test.

- [ ] **Step 8.6: Run it, watch it pass**

Run: `uv run pytest tests/test_screencropnet_scripts.py -v`
Expected: all PASS.

- [ ] **Step 8.7: Omit the two logic-free hyphenated scripts from coverage**

`screencropnet/extract-dominant-colors-of-an-image.py` and `screencropnet/finding-most-common-colors.py` have hyphenated filenames (not importable as modules) and contain no extractable pure logic (top-level imread + `plt.show()` only). In `pyproject.toml` `[tool.coverage.run]`, add:

```toml
omit = [
    "screencropnet/extract-dominant-colors-of-an-image.py",
    "screencropnet/finding-most-common-colors.py",
]
```

(Scoped exception to the "extract + test" decision: these two have nothing to extract. All three scripts with real pure logic are extracted and tested above.)

- [ ] **Step 8.8: Commit**

```bash
git add tests/test_screencropnet_scripts.py screencropnet/pascal_to_csv.py \
        screencropnet/try_predict.py screencropnet/show_bbox_example.py pyproject.toml
git commit -m "refactor(screencropnet): make scripts import-safe + cover pure logic"
```

### 9. Integration & Polish

- [ ] **Step 9.1: Full suite + coverage (what CI runs)**

Run: `make test-cov 2>&1 | tee /tmp/screencropnet-cov-after.txt`
Expected: 0 failures, 0 errors. Compare `screencropnet/*.py` numbers against `/tmp/screencropnet-cov-baseline.txt` — every targeted module materially higher; `arch.py`/`devices.py` unchanged (already covered).

- [ ] **Step 9.2: Lint + typecheck + test gate**

Run: `make check`
Expected: ruff clean, pyright clean (note: `make check` runs `pyright` repo-wide), pytest green. Fix any ruff/pyright issues introduced (unused imports in cloned files are the most likely — remove them).

- [ ] **Step 9.3: Regression sanity — pre-existing screennet tests still green**

Run: `uv run pytest tests/test_screennet_main_units.py tests/test_screennet_main_compute.py tests/test_devices.py tests/test_errors.py -v`
Expected: all PASS (the `test_errors.py` parametrization and shared `helpers`/`main` edits must not regress screennet).

- [ ] **Step 9.4: Final commit**

```bash
git add -A
git commit -m "test(screencropnet): finalize Tier-2 coverage push"
```

## Testing Strategy

- **CPU-only, deterministic:** rely on the autouse `_deterministic` fixture (`torch.manual_seed(0)`); pass the `cpu` fixture into any device-taking helper. No CUDA/MPS assertions without mocking.
- **No network, no training, no real datasets:** synthesize tiny images with `cv2.imwrite`/`np.zeros` into `tmp_path`; the 2.7 MB real fixture is used only for one smoke test per relevant module.
- **Three TDD modes** (see Solution Approach): strict red-green for the 4 bug fixes and 3 script extractions (test must be observed failing for the documented reason before the source change); coverage-delta-as-RED for characterization of already-correct code.
- **Edge cases covered:** invalid `box_format`; all-empty mAP classes; `transform=None` default; empty `paths`; disjoint/overlapping/cross-class NMS; bbox-encoding round-trip identity; image normalization scale (`/255`).
- **Assertion discipline:** `torch.testing.assert_close(..., rtol=0, atol=1e-4)` for float tensors, exact `==` for shapes/ints/enums, existence/size (not pixels) for any plot output, per `.claude/rules/visual-testing.md`.
- **No new pytest markers** (repo has none registered; adding unregistered markers warns).

## Acceptance Criteria

- `make test-cov` runs with **0 failures and 0 errors**; `screencropnet/helpers.py`, `image_utils.py`, `data_set.py`, `main.py` line+branch coverage is materially higher than the Step 1.2 baseline.
- The 4 source bugs each have a regression test that was observed failing before the fix and passing after: `intersection_over_union` invalid format → `ValueError`; `mean_average_precision` all-empty → `0.0`; `load_and_transform_image_for_prediction` default `transform=None` → works; `pred_and_store` empty `paths` → no crash.
- `screencropnet/pascal_to_csv.py`, `try_predict.py`, `show_bbox_example.py` are import-safe (no side effects at import) and their pure logic (`xml_to_csv`, `get_bbox`, `df_to_table`) is tested.
- `tests/test_errors.py` is parametrized over `screennet.errors` + `screencropnet.errors`.
- `make check` is green (ruff + pyright + pytest); pre-existing screennet tests still pass.
- Every Tier-1/Tier-2 testable unit in `screencropnet/*.py` has a test (definition of done = parity with the screennet Tier-2 pattern).

## Validation Commands

Execute these to validate completion:

- `cd /Users/bossjones/dev/bossjones/pytorch-lab && make test-cov` — full suite + branch coverage; 0 failures, screencropnet modules up vs. baseline.
- `uv run pytest tests/test_helpers_nms_map.py tests/test_image_utils.py tests/test_data_set.py tests/test_screencropnet_main_units.py tests/test_screencropnet_main_compute.py tests/test_screencropnet_scripts.py tests/test_errors.py -v` — all new/changed tests pass.
- `make check` — ruff + pyright + pytest gate green.
- `uv run python -c "import screencropnet.pascal_to_csv, screencropnet.try_predict, screencropnet.show_bbox_example; print('import-safe')"` — confirms no import-time side effects (prints `import-safe` with no script output/crash).
- `git log --oneline -12` — confirms one focused commit per task.

## Notes

- **No new dependencies.** pytest / pytest-mock / pytest-cov / cv2 / albumentations / pandas / PIL / matplotlib are all already present. Do **not** run `uv add`.
- **Out of scope (deliberate, flag for a separate ticket):** `devices.extract_device_id` IndexError when the flag is the last arg is *intentionally pinned* by the existing `tests/test_devices.py` "SUSPECTED BUG" test, which runs against BOTH twin modules — fixing it requires a coordinated change to a shared test and both `devices.py` twins, a different concern from this screencropnet coverage push. Same for `main.fix_path`'s 2-arg `exit(...)` call and the `devices.device is None` global. Recommend a follow-up `fix(devices): ...` ticket.
- **Templates are load-bearing:** Tasks 6 & 7 depend on the exact structure of `tests/test_screennet_main_units.py` / `tests/test_screennet_main_compute.py`. Read them fully before cloning; the `matplotlib.use("Agg")`-before-import ordering in the compute file is mandatory for headless determinism.
- **Characterization vs. TDD honesty:** this plan does not pretend every test is clean red-green. Bug fixes and script extraction are strict red-green (observe the failure first). Characterization tests of already-correct code use the coverage delta (baseline captured in Step 1.2) as the evidence the test exercises previously-unexecuted lines — consistent with the repo's existing `SUSPECTED BUG` / known-value convention and the TDD skill's "existing code" guidance.
- **Commit cadence:** one commit per task (9 commits) keeps the bug fixes bisectable and separable from the bulk characterization work.
