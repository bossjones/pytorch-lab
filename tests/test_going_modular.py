"""going_modular contract/regression guards under the modern torch stack.

torch >= 2.6 defaults torch.load(weights_only=True); these lock in that a
state_dict saved by utils.save_model reloads safely under that default.
"""

from __future__ import annotations

import torch

from going_modular import model_builder, utils


def _tiny_model() -> torch.nn.Module:
    return model_builder.TinyVGG(input_shape=3, hidden_units=2, output_shape=3)


def test_tinyvgg_forward_output_shape() -> None:
    model = _tiny_model().eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 64, 64))
    assert out.shape == (1, 3)


def test_save_model_writes_pth(tmp_path) -> None:
    utils.save_model(_tiny_model(), str(tmp_path), "m.pth")
    assert (tmp_path / "m.pth").is_file()


def test_save_model_roundtrips_under_weights_only_true(tmp_path) -> None:
    model = _tiny_model()
    utils.save_model(model, str(tmp_path), "m.pth")

    # The modern default. State dicts are pure tensors, so this must work.
    state = torch.load(tmp_path / "m.pth", weights_only=True)
    reloaded = _tiny_model()
    reloaded.load_state_dict(state)

    for a, b in zip(
        model.state_dict().values(), reloaded.state_dict().values(), strict=True
    ):
        assert torch.equal(a, b)


def test_save_model_rejects_bad_extension(tmp_path) -> None:
    import pytest

    with pytest.raises(AssertionError):
        utils.save_model(_tiny_model(), str(tmp_path), "model.bin")
