"""CLI smoke tests — `--help` must exit 0 and print usage.

Cheap end-to-end guard that the argparse setup and the import-safe
``main()`` entrypoints stay wired correctly. No model load, no training.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script",
    ["screennet/main.py", "screencropnet/main.py"],
)
def test_cli_help_exits_zero(script: str) -> None:
    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
