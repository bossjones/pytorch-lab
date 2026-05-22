"""Tests for contrib/label_pointer.py — pure helpers and CLI wiring.

The interactive matplotlib click-to-label loop is not unit-tested; only the
extracted pure functions and the argparse wiring are covered here.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "contrib" / "label_pointer.py"


def _load_module():
    """Load the PEP 723 contrib script as a module for in-process testing."""
    spec = importlib.util.spec_from_file_location("label_pointer", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["label_pointer"] = mod
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


def test_read_labeled_paths_returns_empty_set_when_file_absent(tmp_path) -> None:
    mod = _load_module()
    assert mod.read_labeled_paths(tmp_path / "missing.txt") == set()


def test_read_labeled_paths_extracts_first_csv_column(tmp_path) -> None:
    mod = _load_module()
    ann = tmp_path / "annotation.txt"
    ann.write_text("a.png,10,20,30,40,twitter\nb.png,1,2,3,4,facebook\n")
    assert mod.read_labeled_paths(ann) == {"a.png", "b.png"}


def test_read_labeled_paths_skips_blank_lines(tmp_path) -> None:
    mod = _load_module()
    ann = tmp_path / "annotation.txt"
    ann.write_text("a.png,10,20,30,40,twitter\n\n   \n")
    assert mod.read_labeled_paths(ann) == {"a.png"}


def test_discover_class_images_groups_by_subfolder(tmp_path) -> None:
    mod = _load_module()
    (tmp_path / "twitter").mkdir()
    (tmp_path / "facebook").mkdir()
    (tmp_path / "twitter" / "a.png").write_bytes(b"")
    (tmp_path / "twitter" / "b.jpg").write_bytes(b"")
    (tmp_path / "facebook" / "c.PNG").write_bytes(b"")
    assert mod.discover_class_images(tmp_path) == {
        "facebook": [tmp_path / "facebook" / "c.PNG"],
        "twitter": [tmp_path / "twitter" / "a.png", tmp_path / "twitter" / "b.jpg"],
    }


def test_discover_class_images_excludes_non_image_files(tmp_path) -> None:
    mod = _load_module()
    (tmp_path / "twitter").mkdir()
    (tmp_path / "twitter" / "a.png").write_bytes(b"")
    (tmp_path / "twitter" / "notes.txt").write_bytes(b"")
    assert mod.discover_class_images(tmp_path) == {
        "twitter": [tmp_path / "twitter" / "a.png"]
    }


def test_discover_class_images_includes_empty_subfolders(tmp_path) -> None:
    mod = _load_module()
    (tmp_path / "tiktok").mkdir()
    assert mod.discover_class_images(tmp_path) == {"tiktok": []}


def test_format_annotation_line_truncates_float_coords() -> None:
    mod = _load_module()
    line = mod.format_annotation_line("a.png", (10.7, 20.2), (110.9, 220.1), "twitter")
    assert line == "a.png,10,20,110,220,twitter"


def test_format_annotation_line_accepts_path_object() -> None:
    mod = _load_module()
    line = mod.format_annotation_line(Path("d/a.png"), (1, 2), (3, 4), "tiktok")
    assert line == "d/a.png,1,2,3,4,tiktok"


def test_summarize_progress_sorts_classes_by_labeled_count_desc() -> None:
    mod = _load_module()
    summary = mod.summarize_progress(
        {"twitter": 3, "facebook": 1}, {"twitter": 10, "facebook": 5}
    )
    assert summary == "twitter : 3/10\nfacebook : 1/5\nTotal 4/15 (27%)"


def test_summarize_progress_handles_zero_total() -> None:
    mod = _load_module()
    summary = mod.summarize_progress({"twitter": 0}, {"twitter": 0})
    assert summary == "twitter : 0/0\nTotal 0/0 (0%)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
