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
    # bbox is shape (1, 4) because __getitem__ builds [[xmin, ymin, xmax, ymax]]
    # and passes the list-of-lists directly to torch.Tensor()
    bbox_flat = bbox.flatten()
    assert tuple(int(v) for v in bbox_flat) == (1, 2, 11, 12)
    torch.testing.assert_close(
        img.max(), torch.tensor(0.0), rtol=0, atol=1e-6
    )  # all-zero image stays 0 after /255


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
