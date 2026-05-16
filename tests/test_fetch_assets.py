"""Guards for contrib/fetch_screencropnet_assets.py.

No network in CI: requests is mocked. Verifies the CLI runs, the asset
map stays in lock-step with ai_docs/screencropnet-assets.md, and that the
fetcher is idempotent (skips an asset whose target already exists).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "contrib" / "fetch_screencropnet_assets.py"
DOC = REPO_ROOT / "ai_docs" / "screencropnet-assets.md"


def _load_module():
    spec = importlib.util.spec_from_file_location("fetch_assets", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fetch_assets"] = mod  # dataclasses needs this during exec
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


def test_no_selection_exits_nonzero() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Nothing selected" in result.stdout


def test_asset_map_matches_provenance_doc() -> None:
    mod = _load_module()
    doc_text = DOC.read_text()
    for asset in mod.ASSETS.values():
        assert asset.url in doc_text, f"{asset.key} url missing from {DOC.name}"
        assert asset.dest in doc_text, f"{asset.key} dest missing from {DOC.name}"


def test_fetch_is_idempotent_when_target_exists(tmp_path, mocker) -> None:
    mod = _load_module()
    get = mocker.patch.object(mod.requests, "get")
    mocker.patch.object(mod, "REPO_ROOT", tmp_path)

    asset = mod.ASSETS["sample"]
    target = tmp_path / asset.dest
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"already here")

    resolved = mod.fetch(asset)

    assert resolved == target
    get.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
