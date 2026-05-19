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


def test_extract_device_id_returns_value_after_flag(devices_mod) -> None:
    assert devices_mod.extract_device_id(["--device", "2"], "device") == "2"


def test_extract_device_id_returns_none_when_absent(devices_mod) -> None:
    assert devices_mod.extract_device_id(["--foo", "bar"], "device") is None


def test_extract_device_id_raises_indexerror_when_flag_is_last(devices_mod) -> None:
    # SUSPECTED BUG: extract_device_id does args[x+1] without bounds check,
    # so passing the named flag as the final arg causes IndexError.
    with pytest.raises(IndexError):
        devices_mod.extract_device_id(["--device"], "device")


def test_get_optimal_device_returns_cuda_when_available(devices_mod, mocker) -> None:
    import argparse

    mocker.patch("torch.cuda.is_available", return_value=True)
    args = argparse.Namespace(gpu=None)
    d = devices_mod.get_optimal_device(args)
    assert d.type == "cuda"


def test_get_optimal_device_uses_gpu_id_when_set(devices_mod, mocker) -> None:
    import argparse

    mocker.patch("torch.cuda.is_available", return_value=True)
    args = argparse.Namespace(gpu=0)
    d = devices_mod.get_optimal_device(args)
    assert str(d) == "cuda:0"


def test_get_optimal_device_falls_back_to_mps(devices_mod, mocker) -> None:
    import argparse

    mocker.patch("torch.cuda.is_available", return_value=False)
    mocker.patch.object(devices_mod, "has_mps", return_value=True)
    args = argparse.Namespace(gpu=None)
    d = devices_mod.get_optimal_device(args)
    assert d.type == "mps"


def test_get_optimal_device_falls_back_to_cpu(devices_mod, mocker) -> None:
    import argparse

    mocker.patch("torch.cuda.is_available", return_value=False)
    mocker.patch.object(devices_mod, "has_mps", return_value=False)
    args = argparse.Namespace(gpu=None)
    d = devices_mod.get_optimal_device(args)
    assert d.type == "cpu"


def test_torch_gc_is_noop_without_cuda(devices_mod, mocker) -> None:
    mocker.patch("torch.cuda.is_available", return_value=False)
    devices_mod.torch_gc()  # must not raise


def test_enable_tf32_is_noop_without_cuda(devices_mod, mocker) -> None:
    mocker.patch("torch.cuda.is_available", return_value=False)
    devices_mod.enable_tf32()  # must not raise


def test_mps_contiguous_returns_same_object_for_cpu_device(devices_mod, cpu) -> None:
    t = torch.ones(2, 3)
    result = devices_mod.mps_contiguous(t, cpu)
    assert result is t


def test_mps_contiguous_to_returns_tensor_on_cpu(devices_mod, cpu) -> None:
    t = torch.ones(2, 3)
    result = devices_mod.mps_contiguous_to(t, cpu)
    assert result.device.type == "cpu"
    torch.testing.assert_close(result, t)


def test_randn_is_seeded_and_reproducible(devices_mod, mocker) -> None:
    mocker.patch.object(devices_mod, "device", torch.device("cpu"))
    r1 = devices_mod.randn(42, (2, 3))
    mocker.patch.object(devices_mod, "device", torch.device("cpu"))
    r2 = devices_mod.randn(42, (2, 3))
    assert r1.shape == (2, 3)
    torch.testing.assert_close(r1, r2)


def test_randn_without_seed_returns_correct_shape(devices_mod, mocker) -> None:
    mocker.patch.object(devices_mod, "device", torch.device("cpu"))
    r = devices_mod.randn_without_seed((2, 3))
    assert r.shape == (2, 3)


def test_autocast_full_precision_returns_nullcontext(devices_mod) -> None:
    import contextlib

    cm = devices_mod.autocast(precision="full")
    assert isinstance(cm, contextlib.nullcontext)


def test_seed_everything_sets_pythonhashseed(devices_mod) -> None:
    import os

    devices_mod.seed_everything(99)
    assert os.environ["PYTHONHASHSEED"] == "99"


def test_seed_everything_makes_random_reproducible(devices_mod) -> None:
    import random

    devices_mod.seed_everything(7)
    v1 = random.random()
    devices_mod.seed_everything(7)
    v2 = random.random()
    assert v1 == v2


def test_mps_check_prints_message_when_mps_not_built(
    devices_mod, mocker, capsys
) -> None:
    mocker.patch("torch.backends.mps.is_available", return_value=False)
    mocker.patch("torch.backends.mps.is_built", return_value=False)
    devices_mod.mps_check()
    captured = capsys.readouterr()
    assert "MPS not available" in captured.out


def test_mps_check_prints_message_when_mps_built_but_unavailable(
    devices_mod, mocker, capsys
) -> None:
    mocker.patch("torch.backends.mps.is_available", return_value=False)
    mocker.patch("torch.backends.mps.is_built", return_value=True)
    devices_mod.mps_check()
    captured = capsys.readouterr()
    assert "MPS not available" in captured.out


def test_has_mps_returns_false_when_backend_unavailable(devices_mod, mocker) -> None:
    mocker.patch("torch.backends.mps.is_available", return_value=False)
    assert devices_mod.has_mps() is False


def test_has_mps_returns_false_when_tensor_move_raises(devices_mod, mocker) -> None:
    # Backend reports available but the trial allocation fails: the bare
    # `except Exception` must swallow it and report MPS as unusable.
    mocker.patch("torch.backends.mps.is_available", return_value=True)
    mocker.patch("torch.zeros", side_effect=RuntimeError("mps allocation failed"))
    assert devices_mod.has_mps() is False


def test_torch_gc_invokes_cuda_cleanup_when_cuda_available(devices_mod, mocker) -> None:
    mocker.patch("torch.cuda.is_available", return_value=True)
    empty = mocker.patch("torch.cuda.empty_cache")
    ipc = mocker.patch("torch.cuda.ipc_collect")
    devices_mod.torch_gc()
    empty.assert_called_once()
    ipc.assert_called_once()


def test_enable_tf32_sets_backend_flags_when_cuda_available(
    devices_mod, mocker
) -> None:
    # The torch.backends.* flag objects can't be restored by mocker
    # (they aren't plain attributes), so snapshot/restore them by hand.
    mocker.patch("torch.cuda.is_available", return_value=True)
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        devices_mod.enable_tf32()  # must not raise
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn


def test_autocast_returns_torch_autocast_when_cuda_available(
    devices_mod, mocker
) -> None:
    import contextlib

    mocker.patch("torch.cuda.is_available", return_value=True)
    cm = devices_mod.autocast()
    # CUDA path returns torch.autocast (an autocast context manager), not the
    # nullcontext fallback used on MPS/CPU.
    assert not isinstance(cm, contextlib.nullcontext)
    assert isinstance(cm, torch.autocast)


def test_randn_mps_branch_uses_cpu_generator(devices_mod, mocker) -> None:
    # When the active device is MPS, randn() must sample on the CPU with a
    # seeded generator and then move to the device (PyTorch MPS seeding bug
    # workaround). Requires a real MPS device on this host.
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this host")
    mocker.patch.object(devices_mod, "device", torch.device("mps"))
    r1 = devices_mod.randn(42, (2, 3))
    r2 = devices_mod.randn(42, (2, 3))
    assert r1.device.type == "mps"
    assert r1.shape == (2, 3)
    torch.testing.assert_close(r1.cpu(), r2.cpu())


def test_randn_without_seed_mps_branch_moves_to_device(devices_mod, mocker) -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this host")
    mocker.patch.object(devices_mod, "device", torch.device("mps"))
    r = devices_mod.randn_without_seed((2, 3))
    assert r.device.type == "mps"
    assert r.shape == (2, 3)


def test_mps_check_exercises_device_when_mps_available(
    devices_mod, mocker, capsys
) -> None:
    # MPS available branch: ic() prints the availability, a tensor is created
    # on the mps device and printed. Requires a real MPS device.
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this host")
    devices_mod.mps_check()
    captured = capsys.readouterr()
    # The single-element ones tensor allocated on mps is printed to stdout.
    assert "device='mps" in captured.out
