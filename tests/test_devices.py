"""devices.py must detect MPS without touching the deprecated
`torch.has_mps` (removed in a future torch; emits UserWarning today)."""

from __future__ import annotations

import importlib
import warnings

import pytest
import torch

DEVICE_MODULES = ["screencropnet.devices", "screennet.devices"]


@pytest.fixture(params=DEVICE_MODULES)
def devices_mod(request: pytest.FixtureRequest):
    return importlib.import_module(request.param)


def test_has_mps_emits_no_deprecation_warning(devices_mod) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        devices_mod.has_mps()
    offending = [w for w in caught if "has_mps" in str(w.message)]
    assert not offending, (
        f"deprecated torch.has_mps was used: {[str(w.message) for w in offending]}"
    )


def test_has_mps_matches_backend_truth(devices_mod) -> None:
    assert devices_mod.has_mps() == torch.backends.mps.is_available()


def test_autocast_no_cuda_warning_without_cuda(devices_mod, mocker) -> None:
    # On a CUDA-less machine, hardcoding torch.autocast("cuda") emits
    # 'CUDA is not available ... Disabling autocast'. autocast() should be
    # device-aware and quiet.
    mocker.patch("torch.cuda.is_available", return_value=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with devices_mod.autocast():
            pass
    offending = [w for w in caught if "autocast" in str(w.message).lower()]
    assert not offending, [str(w.message) for w in offending]


def test_autocast_disable_returns_nullcontext(devices_mod) -> None:
    import contextlib

    assert isinstance(devices_mod.autocast(disable=True), contextlib.nullcontext)
