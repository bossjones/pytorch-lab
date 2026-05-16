"""Entrypoint import-safety guards.

Importing a module must not mutate global interpreter state. The two CLI
``main.py`` files install exception hooks and tune torch threads at module
scope, and ``going_modular.train`` trains a model at module scope. These
tests pin the post-hardening contract: importing them runs no such side
effects (they belong inside ``main()``).

Note: ``BETTER_EXCEPTIONS=1`` pre-installs the hook at interpreter startup,
so a ``sys.excepthook`` identity check cannot observe a *re-install*. We
instead spy on the installer functions in a fresh subprocess and assert the
import never calls them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _import_side_effect_check(
    module: str, *, expect_rich: bool
) -> subprocess.CompletedProcess[str]:
    """Import ``module`` in a fresh interpreter with the side-effect installers
    spied; exit non-zero iff the import triggered any of them."""
    rich_spy = (
        "import rich.traceback\n"
        "rich.traceback.install = _rec('rich.traceback.install')\n"
        if expect_rich
        else ""
    )
    code = (
        "import better_exceptions\n"
        "import torch\n"
        "fired = []\n"
        "def _rec(name):\n"
        "    return lambda *a, **k: fired.append(name)\n"
        "better_exceptions.hook = _rec('better_exceptions.hook')\n"
        "torch.set_num_threads = _rec('torch.set_num_threads')\n"
        f"{rich_spy}"
        f"import {module}\n"
        "assert not fired, "
        f"{module!r} + ' ran module-scope side effects: ' + ', '.join(fired)\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_importing_screennet_main_has_no_side_effects() -> None:
    result = _import_side_effect_check("screennet.main", expect_rich=False)
    assert result.returncode == 0, result.stderr


def test_importing_screencropnet_main_has_no_side_effects() -> None:
    # Also covers rich.traceback.install(), active at module scope here.
    result = _import_side_effect_check("screencropnet.main", expect_rich=True)
    assert result.returncode == 0, result.stderr


def test_importing_going_modular_train_does_not_train(mocker) -> None:
    train_spy = mocker.patch("going_modular.engine.train")
    save_spy = mocker.patch("going_modular.utils.save_model")
    mocker.patch(
        "going_modular.data_setup.create_dataloaders",
        return_value=(mocker.MagicMock(), mocker.MagicMock(), ["a", "b", "c"]),
    )

    sys.modules.pop("going_modular.train", None)
    try:
        import going_modular.train  # noqa: F401
    finally:
        sys.modules.pop("going_modular.train", None)

    train_spy.assert_not_called()
    save_spy.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
