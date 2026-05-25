"""Tests for contrib/docstring_coverage.py — the docstring-audit generator.

Covers the pure scanning/bucketing helpers and the CLI wiring. The full report
rendering is verified end-to-end by regenerating docs/coverage_docs/report.md.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "contrib" / "docstring_coverage.py"


def _load_module():
    """Load the PEP 723 contrib script as a module for in-process testing."""
    spec = importlib.util.spec_from_file_location("docstring_coverage", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["docstring_coverage"] = mod
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


def _scan(mod, tmp_path: Path, source: str):
    """Write `source` to a temp .py file and return its scan_file() result."""
    f = tmp_path / "sample.py"
    f.write_text(source)
    return mod.scan_file(f)


def test_scan_file_counts_documented_function(tmp_path) -> None:
    mod = _load_module()
    stats = _scan(mod, tmp_path, '"""Mod."""\ndef foo():\n    """Doc."""\n')
    assert (stats.pub_funcs, stats.doc_funcs) == (1, 1)
    assert stats.gaps == []


def test_scan_file_records_undocumented_function_as_gap(tmp_path) -> None:
    mod = _load_module()
    stats = _scan(mod, tmp_path, "def foo():\n    return 1\n")
    assert (stats.pub_funcs, stats.doc_funcs) == (1, 0)
    assert len(stats.gaps) == 1
    gap = stats.gaps[0]
    assert (gap.qualname, gap.lineno, gap.kind) == ("foo", 1, "function")


def test_scan_file_excludes_private_functions(tmp_path) -> None:
    mod = _load_module()
    stats = _scan(mod, tmp_path, "def _hidden():\n    return 1\n")
    assert stats.pub_funcs == 0
    assert stats.gaps == []


def test_scan_file_counts_dunder_init_as_public(tmp_path) -> None:
    mod = _load_module()
    source = 'class C:\n    """Doc."""\n    def __init__(self):\n        pass\n'
    stats = _scan(mod, tmp_path, source)
    assert stats.pub_funcs == 1
    assert stats.gaps[0].qualname == "C.__init__"


def test_scan_file_counts_class_and_method(tmp_path) -> None:
    mod = _load_module()
    source = (
        'class C:\n'
        '    """Class doc."""\n'
        '    def method(self):\n'
        '        """Method doc."""\n'
    )
    stats = _scan(mod, tmp_path, source)
    assert (stats.pub_classes, stats.doc_classes) == (1, 1)
    assert (stats.pub_funcs, stats.doc_funcs) == (1, 1)


def test_scan_file_ignores_defs_nested_in_function_bodies(tmp_path) -> None:
    mod = _load_module()
    source = 'def outer():\n    """Doc."""\n    def inner():\n        pass\n'
    stats = _scan(mod, tmp_path, source)
    assert stats.pub_funcs == 1


def test_scan_file_detects_module_docstring(tmp_path) -> None:
    mod = _load_module()
    assert _scan(mod, tmp_path, '"""Module doc."""\n').has_module_docstring is True


def test_scan_file_detects_missing_module_docstring(tmp_path) -> None:
    mod = _load_module()
    assert _scan(mod, tmp_path, "x = 1\n").has_module_docstring is False


def test_directory_bucket_root_file_is_root() -> None:
    mod = _load_module()
    assert mod.directory_bucket("helper_functions.py") == "root"


def test_directory_bucket_nested_file_uses_top_dir() -> None:
    mod = _load_module()
    assert mod.directory_bucket("screencropnet/main.py") == "screencropnet"
    assert (
        mod.directory_bucket("going_modular/going_modular/engine.py")
        == "going_modular"
    )


def test_iter_scope_files_excludes_tests_and_notebooks(tmp_path) -> None:
    mod = _load_module()
    (tmp_path / "contrib").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "contrib" / "a.py").write_text("")
    (tmp_path / "contrib" / "nb.ipynb").write_text("")
    (tmp_path / "tests" / "test_x.py").write_text("")
    rels = {p.relative_to(tmp_path).as_posix() for p in mod.iter_scope_files(tmp_path)}
    assert "contrib/a.py" in rels
    assert "contrib/nb.ipynb" not in rels
    assert "tests/test_x.py" not in rels


def test_iter_scope_files_includes_scope_file_excludes_pycache(tmp_path) -> None:
    mod = _load_module()
    (tmp_path / "helper_functions.py").write_text("")
    (tmp_path / "contrib" / "__pycache__").mkdir(parents=True)
    (tmp_path / "contrib" / "__pycache__" / "cached.py").write_text("")
    rels = {p.relative_to(tmp_path).as_posix() for p in mod.iter_scope_files(tmp_path)}
    assert "helper_functions.py" in rels
    assert "contrib/__pycache__/cached.py" not in rels


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
