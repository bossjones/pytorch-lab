"""Tier-2 unit tests for screencropnet.main: small models + tiny DataLoaders,
file-system helpers, the seed validator, and plotting helpers — still no
network, no real training, no GPU required (CPU + Agg backend only).

These exercise the larger uncovered blocks (validate_seed, info, the
compute_* eval helpers, create_writer, confusion-matrix + plot helpers,
write_training_results_to_csv, dir-cleanup utilities).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")  # headless: no display, deterministic across machines

import screencropnet.main as m  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


class _TwoClassNet(nn.Module):
    """Deterministic 4->2 linear classifier (no randomness, no training)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(2, 4))
            self.fc.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _tiny_loader() -> DataLoader:
    # class 0 → strong logit[0], class 1 → strong logit[1] (always correct).
    x = torch.tensor(
        [
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0, 0.0],
            [0.0, 6.0, 0.0, 0.0],
        ]
    )
    y = torch.tensor([0, 1, 0, 1])
    return DataLoader(TensorDataset(x, y), batch_size=2)


# ---------------------------------------------------------------------------
# compute_accuracy / compute_epoch_loss / compute_confusion_matrix
# ---------------------------------------------------------------------------


def test_compute_accuracy_perfect_classifier_is_100(cpu) -> None:
    acc = m.compute_accuracy(_TwoClassNet(), _tiny_loader(), device=cpu)
    torch.testing.assert_close(acc, torch.tensor(100.0), rtol=0, atol=1e-4)


def test_compute_epoch_loss_returns_finite_scalar(cpu) -> None:
    loss = m.compute_epoch_loss(_TwoClassNet(), _tiny_loader(), device=cpu)
    assert loss.numel() == 1
    assert torch.isfinite(loss).all()
    assert loss.item() >= 0.0


def test_compute_confusion_matrix_perfect_classifier_is_diagonal(cpu) -> None:
    mat = m.compute_confusion_matrix(_TwoClassNet(), _tiny_loader(), device=cpu)
    # 2 of each class, all correct → [[2,0],[0,2]]
    np.testing.assert_array_equal(mat, np.array([[2, 0], [0, 2]]))


# ---------------------------------------------------------------------------
# create_writer
# ---------------------------------------------------------------------------


def test_create_writer_builds_dated_logdir(tmp_path, monkeypatch) -> None:
    from datetime import datetime

    monkeypatch.chdir(tmp_path)
    writer = m.create_writer("exp", "effnetb0", extra="5_epochs")
    try:
        stamp = datetime.now().strftime("%Y-%m-%d")
        expected = Path("runs") / stamp / "exp" / "effnetb0" / "5_epochs"
        assert Path(writer.log_dir) == expected
    finally:
        writer.close()


def test_create_writer_without_extra_omits_trailing_segment(
    tmp_path, monkeypatch
) -> None:
    from datetime import datetime

    monkeypatch.chdir(tmp_path)
    writer = m.create_writer("exp", "effnetb0")
    try:
        stamp = datetime.now().strftime("%Y-%m-%d")
        expected = Path("runs") / stamp / "exp" / "effnetb0"
        assert Path(writer.log_dir) == expected
    finally:
        writer.close()


# ---------------------------------------------------------------------------
# validate_seed  (covers devices.seed_everything + CPU/MPS tensor branch)
# ---------------------------------------------------------------------------


def test_validate_seed_is_idempotent_and_seeds_rng() -> None:
    import random

    m.validate_seed(123)
    a = random.random()
    m.validate_seed(123)
    b = random.random()
    assert a == b


# ---------------------------------------------------------------------------
# info  (must print a watermark and sys.exit(0))
# ---------------------------------------------------------------------------


def test_info_exits_zero_after_reporting(tmp_path, mocker) -> None:
    import argparse

    mocker.patch.object(m, "watermark", return_value="WATERMARK")
    mocker.patch.object(m.devices, "mps_check")
    mocker.patch.object(m, "validate_seed")
    mocker.patch.object(m, "walk_through_dir")
    args = argparse.Namespace(seed=7)

    with pytest.raises(SystemExit) as excinfo:
        m.info(args, dataset_root_dir=str(tmp_path))

    assert excinfo.value.code == 0


# ---------------------------------------------------------------------------
# write_training_results_to_csv
# ---------------------------------------------------------------------------


def test_write_training_results_to_csv_writes_expected_row(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    model = _TwoClassNet()
    train_data = list(range(8))
    test_data = list(range(2))

    m.write_training_results_to_csv(
        "Test Machine",
        "cpu",
        dataset_name="ds",
        num_epochs=2,
        batch_size=4,
        image_size=(224, 224),
        train_data=train_data,
        test_data=test_data,
        total_train_time=10.0,
        model=model,
    )

    out = tmp_path / "results" / "test_machine_cpu_ds_image_size.csv"
    assert out.exists()
    df = pd.read_csv(str(out))
    assert df.loc[0, "num_train_samples"] == 8
    assert df.loc[0, "num_test_samples"] == 2
    assert df.loc[0, "time_per_epoch"] == 5.0
    assert df.loc[0, "model"] == "_TwoClassNet"


# ---------------------------------------------------------------------------
# clean_dir_images / clean_dirs_in_dir
# ---------------------------------------------------------------------------


def test_clean_dir_images_removes_only_jpgs(tmp_path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.jpg").write_bytes(b"x")
    keep = tmp_path / "c.png"
    keep.write_bytes(b"x")

    m.clean_dir_images(str(tmp_path))

    assert not (tmp_path / "a.jpg").exists()
    assert not (tmp_path / "b.jpg").exists()
    assert keep.exists()  # non-jpg untouched


def test_clean_dirs_in_dir_removes_tree(tmp_path) -> None:
    target = tmp_path / "victim"
    target.mkdir()
    (target / "nested.txt").write_text("data")

    m.clean_dirs_in_dir(str(target))

    assert not target.exists()


def test_clean_dirs_in_dir_handles_missing_path_gracefully(tmp_path, capsys) -> None:
    missing = tmp_path / "does_not_exist"
    m.clean_dirs_in_dir(str(missing))  # OSError is caught and printed
    captured = capsys.readouterr()
    assert "Error:" in captured.out


# ---------------------------------------------------------------------------
# show_confusion_matrix_helper / run_confusion_matrix
# ---------------------------------------------------------------------------


def test_show_confusion_matrix_helper_writes_file(tmp_path) -> None:
    cmat = np.array([[5, 1], [2, 4]])
    out = tmp_path / "cm.png"

    m.show_confusion_matrix_helper(
        cmat, class_names=["a", "b"], to_disk=True, fname=str(out)
    )

    assert out.exists()
    assert out.stat().st_size > 0


def test_run_confusion_matrix_writes_plot(tmp_path, monkeypatch, cpu) -> None:
    # End-to-end of the small eval path: tiny model -> confusion matrix ->
    # plot saved. show_confusion_matrix_helper defaults fname="plot.png" and
    # writes relative to cwd, so chdir into tmp_path.
    monkeypatch.chdir(tmp_path)
    m.run_confusion_matrix(
        _TwoClassNet(), _tiny_loader(), device=cpu, class_names=["a", "b"]
    )
    assert (tmp_path / "plot.png").exists()


# ---------------------------------------------------------------------------
# fix_path
# ---------------------------------------------------------------------------


def test_fix_path_passthrough_for_non_str_non_list() -> None:
    assert m.fix_path(42) == 42
    assert m.fix_path(3.5) == 3.5


def test_fix_path_returns_unknown_relative_path_unchanged() -> None:
    # Not a ~, not under home, not a file/dir → returned as-is.
    assert m.fix_path("definitely/not/a/real/relative/path") == (
        "definitely/not/a/real/relative/path"
    )


def test_fix_path_resolves_tilde_to_existing_file(tmp_path, monkeypatch) -> None:
    # Point HOME at tmp_path so "~/probe.txt" resolves to a real file.
    monkeypatch.setenv("HOME", str(tmp_path))
    probe = tmp_path / "probe.txt"
    probe.write_text("hi")
    assert m.fix_path("~/probe.txt") == str(probe)


def test_fix_path_maps_list_elementwise() -> None:
    out = m.fix_path(["nope/relative/a", 7])
    assert out[0] == "nope/relative/a"
    assert out[1] == 7  # non-str element passes through


# ---------------------------------------------------------------------------
# console_print_table / inspect_csv_results
# ---------------------------------------------------------------------------


def test_console_print_table_renders_without_error(capsys) -> None:
    df = pd.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})
    m.console_print_table(df)
    captured = capsys.readouterr()
    assert "col_a" in captured.out
    assert "col_b" in captured.out


def test_inspect_csv_results_concats_and_prints(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    (results / "r1.csv").write_text("a,b\n1,2\n")
    (results / "r2.csv").write_text("a,b\n3,4\n")

    df = m.inspect_csv_results()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert set(df["a"]) == {1, 3}


# ---------------------------------------------------------------------------
# download_and_predict  (already-exists branch: no network)
# ---------------------------------------------------------------------------


def test_download_and_predict_skips_download_when_file_exists(
    tmp_path, mocker, capsys
) -> None:
    data_path = tmp_path
    existing = data_path / "image.png"
    existing.write_bytes(b"fake")

    pred = mocker.patch.object(m, "pred_and_plot_image")

    m.download_and_predict(
        "https://example.com/path/image.png",
        model=object(),
        data_path=data_path,
        class_names=["a"],
        device=None,
    )

    captured = capsys.readouterr()
    # NOTE: the long tmp path makes rich wrap the print() output, so the
    # phrase spans a newline ("skipping \ndownload."). Normalize whitespace
    # before asserting on the message.
    assert "already exists" in captured.out
    assert "skipping" in " ".join(captured.out.split())
    pred.assert_called_once()


# ---------------------------------------------------------------------------
# setup_workspace  (directory-exists branch: no network)
# ---------------------------------------------------------------------------


def test_setup_workspace_noops_when_image_dir_exists(tmp_path, capsys) -> None:
    image_path = tmp_path / "twitter_screenshots_localization_dataset"
    image_path.mkdir()

    m.setup_workspace(data_path=tmp_path, image_path=image_path)

    captured = capsys.readouterr()
    assert "directory exists" in captured.out


# ---------------------------------------------------------------------------
# get_model_named_params
# ---------------------------------------------------------------------------


def test_get_model_named_params_prints_requires_grad(capsys) -> None:
    model = _TwoClassNet()
    m.get_model_named_params(model)
    captured = capsys.readouterr()
    assert "fc.weight" in captured.out
    assert "fc.bias" in captured.out


# ---- pred_and_store: empty paths must not crash ----


def test_pred_and_store_empty_paths_returns_empty() -> None:
    result = m.pred_and_store(
        paths=[],
        model=None,
        device="cpu",
    )
    assert result == []  # Before fix: UnboundLocalError at print(fullsize_bboxes) when paths=[]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
