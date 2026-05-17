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
