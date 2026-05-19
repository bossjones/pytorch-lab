"""Contract/regression guards for root helper_functions.py under the modern
stack. Filesystem is exercised via tmp_path; no network."""

from __future__ import annotations

import torch

import helper_functions as hf


def test_accuracy_fn_known_value() -> None:
    y_true = torch.tensor([1, 0, 1, 1])
    y_pred = torch.tensor([1, 0, 0, 1])  # 3/4 correct
    assert hf.accuracy_fn(y_true, y_pred) == 75.0


def test_print_train_time_returns_elapsed(capsys) -> None:
    assert hf.print_train_time(1.0, 3.5, device="cpu") == 2.5
    assert "2.500 seconds" in capsys.readouterr().out


def test_set_seeds_makes_torch_deterministic() -> None:
    hf.set_seeds(123)
    a = torch.rand(4)
    hf.set_seeds(123)
    b = torch.rand(4)
    torch.testing.assert_close(a, b)


def test_walk_through_dir_reports_counts(tmp_path, capsys) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "img.png").write_bytes(b"x")
    hf.walk_through_dir(str(tmp_path))
    out = capsys.readouterr().out
    assert "1 directories" in out  # the top dir contains 'sub'
    assert "1 images" in out  # 'sub' contains img.png
