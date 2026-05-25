"""Tests for contrib/preprocessing_data_loader.py — pure helpers and CLI wiring.

The interactive matplotlib RectangleSelector loop is not unit-tested; only the
extracted pure functions and the argparse wiring are covered here.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "contrib" / "preprocessing_data_loader.py"


def _load_module():
    """Load the PEP 723 contrib script as a module for in-process testing."""
    spec = importlib.util.spec_from_file_location("preprocessing_data_loader", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["preprocessing_data_loader"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_help_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_discover_images_finds_images_recursively(tmp_path) -> None:
    mod = _load_module()
    (tmp_path / "twitter").mkdir()
    (tmp_path / "tiktok").mkdir()
    a = tmp_path / "twitter" / "a.PNG"
    b = tmp_path / "tiktok" / "b.jpg"
    a.write_bytes(b"")
    b.write_bytes(b"")
    assert mod.discover_images(tmp_path) == [b, a]


def test_discover_images_excludes_non_image_files(tmp_path) -> None:
    mod = _load_module()
    img = tmp_path / "a.png"
    img.write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    assert mod.discover_images(tmp_path) == [img]


def test_discover_images_returns_empty_for_no_images(tmp_path) -> None:
    mod = _load_module()
    assert mod.discover_images(tmp_path) == []


def test_class_num_from_path_uses_parent_folder_name() -> None:
    mod = _load_module()
    p = Path("data/test/tiktok/IMG_7122.PNG")
    assert mod.class_num_from_path(p, {"twitter": 2, "tiktok": 1}) == 1


def test_class_num_from_path_raises_for_unknown_label() -> None:
    mod = _load_module()
    with pytest.raises(KeyError):
        mod.class_num_from_path(Path("data/unknown/x.png"), {"twitter": 0})


def test_extents_to_bbox_reorders_and_truncates() -> None:
    mod = _load_module()
    # RectangleSelector.extents is (xmin, xmax, ymin, ymax)
    assert mod.extents_to_bbox((4.6, 1159.5, 503.6, 1734.6)) == (4, 503, 1159, 1734)


def test_write_annotations_writes_header_and_rows(tmp_path) -> None:
    mod = _load_module()
    out = tmp_path / "annotations.csv"
    rows = [
        ("a.png", 1179, 2556, 2, 30, 391, 1161, 752),
        ("b.png", 800, 600, 0, 1, 2, 3, 4),
    ]
    mod.write_annotations(rows, out)
    lines = out.read_text().splitlines()
    assert lines[0] == "filename,width,height,class_num,xmin,ymin,xmax,ymax"
    assert lines[1] == "a.png,1179,2556,2,30,391,1161,752"
    assert lines[2] == "b.png,800,600,0,1,2,3,4"


def test_write_annotations_empty_rows_writes_only_header(tmp_path) -> None:
    mod = _load_module()
    out = tmp_path / "annotations.csv"
    mod.write_annotations([], out)
    assert out.read_text().splitlines() == [
        "filename,width,height,class_num,xmin,ymin,xmax,ymax"
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
