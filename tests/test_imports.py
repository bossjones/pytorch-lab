"""Import smoke test — guards the conda→uv package restructure (no more
sys.path hacks) and the torch 2.x / numpy 2.x modernization.

Excludes modules that execute work at import time (the two main.py CLIs and
going_modular.train, which loads a dataset on import).
"""

from __future__ import annotations

import importlib

import pytest

IMPORTABLE = [
    "helper_functions",
    "going_modular.data_setup",
    "going_modular.engine",
    "going_modular.model_builder",
    "going_modular.utils",
    "going_modular.predictions",
    "screennet.devices",
    "screennet.errors",
    "screencropnet.devices",
    "screencropnet.errors",
    "screencropnet.arch",
    "screencropnet.helpers",
    "screencropnet.ml_types",
    "screencropnet.data_set",
    "screencropnet.image_utils",
]


@pytest.mark.parametrize("module", IMPORTABLE)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)
