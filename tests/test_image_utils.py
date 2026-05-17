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
