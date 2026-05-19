"""load_checkpoint() helper — full-checkpoint resume contract.

A real resume checkpoint carries non-tensor objects (e.g. an
argparse.Namespace of training args), so it must be loaded with
weights_only=False. These tests pin that contract and the "gpu set but no
CUDA" fallback, which covers a latent undefined-variable bug in the old
inline main_worker() resume block.
"""

from __future__ import annotations

import argparse
import importlib
import pickle

import pytest
import torch

from going_modular import model_builder

MODULES = ["screennet.main", "screencropnet.main"]


def _write_checkpoint(tmp_path):
    state_dict = model_builder.TinyVGG(
        input_shape=3, hidden_units=2, output_shape=3
    ).state_dict()
    ckpt = {
        "epoch": 7,
        "best_acc1": torch.tensor(91.2),
        "state_dict": state_dict,
        "optimizer": {"lr": 1e-3},
        # Real checkpoints from these CLIs persist training args; the safe
        # unpickler rejects this, which is exactly why resume needs
        # weights_only=False.
        "args": argparse.Namespace(gpu=None, resume="x"),
    }
    path = tmp_path / "ckpt.pth"
    torch.save(ckpt, path)
    return path, state_dict


@pytest.mark.parametrize("module", MODULES)
def test_load_checkpoint_returns_full_dict(module: str, tmp_path) -> None:
    mod = importlib.import_module(module)
    path, state_dict = _write_checkpoint(tmp_path)

    out = mod.load_checkpoint(str(path))

    assert isinstance(out, dict)
    assert out["epoch"] == 7
    assert set(out["state_dict"].keys()) == set(state_dict.keys())


@pytest.mark.parametrize("module", MODULES)
def test_resume_checkpoint_requires_weights_only_false(module: str, tmp_path) -> None:
    # Characterization guard: documents the torch constraint load_checkpoint
    # works around. If this ever stops raising, revisit weights_only=False.
    path, _ = _write_checkpoint(tmp_path)
    with pytest.raises(pickle.UnpicklingError):
        torch.load(path, weights_only=True)


@pytest.mark.parametrize("module", MODULES)
def test_load_checkpoint_gpu_set_but_no_cuda(module: str, tmp_path, mocker) -> None:
    mod = importlib.import_module(module)
    mocker.patch("torch.cuda.is_available", return_value=False)
    path, _ = _write_checkpoint(tmp_path)

    # gpu is set but CUDA is unavailable: must still load via plain torch.load
    # (the old inline block left `checkpoint` undefined in this branch).
    out = mod.load_checkpoint(str(path), gpu=0)

    assert out["epoch"] == 7


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
