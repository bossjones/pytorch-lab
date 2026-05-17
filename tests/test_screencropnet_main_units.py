"""Unit tests for deterministic pure functions and lightweight I/O helpers
in screencropnet.main — no GPU, no network, no training loops.

Coverage targets (Tier 2):
  pure helpers, PIL<->tensor conversions, channel detection,
  write/read CSV, AverageMeter, ProgressMeter, accuracy, save/load model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image
from rich.table import Table

import screencropnet.main as m

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "00006_twitter.PNG"

# ---------------------------------------------------------------------------
# tilda
# ---------------------------------------------------------------------------


def test_tilda_expands_tilde_in_str() -> None:
    result = m.tilda("~/testdir")
    assert result == str(Path("~/testdir").expanduser())
    assert not result.startswith("~")


def test_tilda_expands_str_members_in_list_and_passes_through_ints() -> None:
    result = m.tilda(["~/a", 42])
    assert result[0] == str(Path("~/a").expanduser())
    assert result[1] == 42  # non-str passes through


def test_tilda_passthrough_for_non_str_non_list() -> None:
    assert m.tilda(99) == 99
    assert m.tilda(3.14) == 3.14


# ---------------------------------------------------------------------------
# is_file / is_directory
# ---------------------------------------------------------------------------


def test_is_file_returns_true_for_existing_file(tmp_path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("hi")
    assert m.is_file(str(f)) is True


def test_is_file_returns_false_for_directory(tmp_path) -> None:
    assert m.is_file(str(tmp_path)) is False


def test_is_directory_returns_true_for_existing_dir(tmp_path) -> None:
    assert m.is_directory(str(tmp_path)) is True


def test_is_directory_returns_false_for_file(tmp_path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("hi")
    assert m.is_directory(str(f)) is False


# ---------------------------------------------------------------------------
# csv_to_df / write_predict_results_to_csv
# ---------------------------------------------------------------------------


def test_csv_to_df_reads_csv_and_returns_dataframe(tmp_path) -> None:
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n3,4\n")
    df = m.csv_to_df(str(csv))
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_write_predict_results_to_csv_creates_file_and_appends(tmp_path) -> None:
    import argparse

    results_path = tmp_path / "results.csv"
    args = argparse.Namespace(results=str(results_path))

    pred_dicts = [
        {
            "image_path": "a.png",
            "pred_prob": 0.9,
            "pred_class": "twitter",
            "class_name": "twitter",
            "correct": True,
        }
    ]

    # First call: creates file with header
    m.write_predict_results_to_csv(pred_dicts, args)
    assert results_path.exists()
    df1 = pd.read_csv(str(results_path))
    assert "class_name" not in df1.columns
    assert "correct" not in df1.columns
    assert len(df1) == 1

    # Second call: appends
    m.write_predict_results_to_csv(pred_dicts, args)
    df2 = pd.read_csv(str(results_path))
    assert len(df2) == 2


# ---------------------------------------------------------------------------
# df_to_table
# ---------------------------------------------------------------------------


def test_df_to_table_with_index_adds_extra_column() -> None:
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    table = Table()
    result = m.df_to_table(df, table, show_index=True)
    # 2 data columns + 1 index column
    assert len(result.columns) == 3
    assert result.row_count == 2


def test_df_to_table_without_index_matches_df_columns() -> None:
    df = pd.DataFrame({"a": [10], "b": [20], "c": [30]})
    table = Table()
    result = m.df_to_table(df, table, show_index=False)
    assert len(result.columns) == 3
    assert result.row_count == 1


def test_df_to_table_with_named_index() -> None:
    df = pd.DataFrame({"v": [1]})
    table = Table()
    result = m.df_to_table(df, table, show_index=True, index_name="idx")
    assert result.columns[0].header == "idx"


# ---------------------------------------------------------------------------
# print_train_time
# ---------------------------------------------------------------------------


def test_print_train_time_returns_rounded_elapsed(capsys) -> None:
    result = m.print_train_time(1.0, 2.567)
    assert result == 1.567
    captured = capsys.readouterr()
    assert "1.567" in captured.out


def test_print_train_time_with_device_includes_device_in_output(capsys) -> None:
    m.print_train_time(0.0, 1.0, device="cpu", machine="testmachine")
    captured = capsys.readouterr()
    assert "cpu" in captured.out
    assert "testmachine" in captured.out


def test_print_train_time_without_device_uses_simple_format(capsys) -> None:
    m.print_train_time(0.0, 1.0)
    captured = capsys.readouterr()
    assert "Train time:" in captured.out


# ---------------------------------------------------------------------------
# Summary enum
# ---------------------------------------------------------------------------


def test_summary_enum_values() -> None:
    from screencropnet.main import Summary

    assert Summary.NONE.value == 0
    assert Summary.AVERAGE.value == 1
    assert Summary.SUM.value == 2
    assert Summary.COUNT.value == 3


# ---------------------------------------------------------------------------
# AverageMeter
# ---------------------------------------------------------------------------


def test_average_meter_running_average() -> None:
    from screencropnet.main import AverageMeter

    meter = AverageMeter("loss")
    meter.update(2.0, n=1)
    meter.update(4.0, n=1)
    assert meter.val == 4.0
    assert meter.sum == 6.0
    assert meter.count == 2
    assert meter.avg == 3.0


def test_average_meter_reset_zeros_all_fields() -> None:
    from screencropnet.main import AverageMeter

    meter = AverageMeter("acc")
    meter.update(1.0)
    meter.reset()
    assert meter.val == 0
    assert meter.sum == 0
    assert meter.count == 0
    assert meter.avg == 0


def test_average_meter_str_uses_format_string() -> None:
    from screencropnet.main import AverageMeter

    meter = AverageMeter("acc", fmt=":.2f")
    meter.update(0.75)
    s = str(meter)
    assert "acc" in s
    assert "0.75" in s


def test_average_meter_summary_average() -> None:
    from screencropnet.main import AverageMeter, Summary

    meter = AverageMeter("loss", summary_type=Summary.AVERAGE)
    meter.update(2.0)
    meter.update(4.0)
    s = meter.summary()
    assert "3.000" in s


def test_average_meter_summary_sum() -> None:
    from screencropnet.main import AverageMeter, Summary

    meter = AverageMeter("loss", summary_type=Summary.SUM)
    meter.update(2.0)
    meter.update(4.0)
    s = meter.summary()
    assert "6.000" in s


def test_average_meter_summary_count() -> None:
    from screencropnet.main import AverageMeter, Summary

    meter = AverageMeter("loss", summary_type=Summary.COUNT)
    meter.update(1.0)
    meter.update(1.0)
    s = meter.summary()
    assert "2.000" in s


def test_average_meter_summary_none_returns_empty() -> None:
    from screencropnet.main import AverageMeter, Summary

    meter = AverageMeter("loss", summary_type=Summary.NONE)
    assert meter.summary() == ""


def test_average_meter_summary_invalid_type_raises_valueerror() -> None:
    from screencropnet.main import AverageMeter

    meter = AverageMeter("x")
    meter.summary_type = "not_a_summary"  # force invalid
    with pytest.raises(ValueError):
        meter.summary()


# ---------------------------------------------------------------------------
# ProgressMeter
# ---------------------------------------------------------------------------


def test_progress_meter_batch_fmtstr_pads_correctly() -> None:
    from screencropnet.main import ProgressMeter

    pm = ProgressMeter(100, [])
    # 100 → 3 digits → format is [{:3d}/100]
    assert pm.batch_fmtstr == "[{:3d}/100]"


def test_progress_meter_batch_fmtstr_single_digit_batches() -> None:
    from screencropnet.main import ProgressMeter

    pm = ProgressMeter(5, [])
    assert pm.batch_fmtstr == "[{:1d}/5]"


def test_progress_meter_display_prints_entries(capsys) -> None:
    from screencropnet.main import AverageMeter, ProgressMeter

    meter = AverageMeter("loss")
    meter.update(0.5)
    pm = ProgressMeter(10, [meter], prefix="Epoch: ")
    pm.display(3)
    captured = capsys.readouterr()
    assert "Epoch:" in captured.out
    assert "loss" in captured.out


def test_progress_meter_display_summary_prints_star(capsys) -> None:
    from screencropnet.main import AverageMeter, ProgressMeter, Summary

    meter = AverageMeter("acc", summary_type=Summary.AVERAGE)
    meter.update(0.9)
    pm = ProgressMeter(10, [meter])
    pm.display_summary()
    captured = capsys.readouterr()
    assert " *" in captured.out


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------


def test_accuracy_top1_perfect_score(cpu) -> None:
    # All predictions correct: top-1 should be 100.0
    output = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    target = torch.tensor([0, 1])
    res = m.accuracy(output, target, topk=(1,))
    torch.testing.assert_close(res[0], torch.tensor([100.0]), rtol=0, atol=1e-4)


def test_accuracy_top1_zero_score(cpu) -> None:
    # All predictions wrong: top-1 should be 0.0
    output = torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    target = torch.tensor([1, 2])
    res = m.accuracy(output, target, topk=(1,))
    torch.testing.assert_close(res[0], torch.tensor([0.0]), rtol=0, atol=1e-4)


def test_accuracy_topk_top2_geq_top1(cpu) -> None:
    output = torch.randn(4, 3)
    target = torch.randint(0, 3, (4,))
    res = m.accuracy(output, target, topk=(1, 2))
    assert res[1].item() >= res[0].item()


# ---------------------------------------------------------------------------
# walk_through_dir
# ---------------------------------------------------------------------------


def test_walk_through_dir_prints_correct_counts(tmp_path, capsys) -> None:
    subdir = tmp_path / "class_a"
    subdir.mkdir()
    (subdir / "img1.png").write_bytes(b"x")
    (subdir / "img2.png").write_bytes(b"x")

    m.walk_through_dir(tmp_path)
    captured = capsys.readouterr()
    # Root: 1 dir, 0 files
    assert "1 directories" in captured.out
    # Subdir: 0 dirs, 2 files
    assert "2 images" in captured.out


# ---------------------------------------------------------------------------
# PIL <-> tensor conversions  (no GPU, no network)
# ---------------------------------------------------------------------------


def test_get_pil_image_channels_rgb_synthetic(tmp_path) -> None:
    img = Image.new("RGB", (8, 8), color=(255, 0, 0))
    path = tmp_path / "rgb.png"
    img.save(str(path))
    channels = m.get_pil_image_channels(str(path))
    assert channels == 3


def test_convert_pil_image_to_rgb_channels_converts_rgb(tmp_path) -> None:
    img = Image.new("RGB", (4, 4))
    path = tmp_path / "rgb.png"
    img.save(str(path))
    result = m.convert_pil_image_to_rgb_channels(str(path))
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"


def test_convert_pil_image_to_rgb_channels_rgba_passthrough(tmp_path) -> None:
    img = Image.new("RGBA", (4, 4))
    path = tmp_path / "rgba.png"
    img.save(str(path))
    result = m.convert_pil_image_to_rgb_channels(str(path))
    assert isinstance(result, Image.Image)
    # 4-channel branch returns image as-is (mode preserved)


def test_convert_pil_image_to_rgb_channels_handles_real_fixture() -> None:
    # Real-asset smoke test: 16-bit RGBA PNG.
    assert FIXTURE.exists(), f"Fixture not found: {FIXTURE}"
    result = m.convert_pil_image_to_rgb_channels(str(FIXTURE))
    assert isinstance(result, Image.Image)


def test_convert_pil_image_to_torch_tensor_shape_and_range() -> None:
    img = Image.new("RGB", (16, 8))
    tensor = m.convert_pil_image_to_torch_tensor(img)
    assert tensor.shape == (3, 8, 16)  # (C, H, W)
    assert tensor.dtype == torch.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_convert_pil_tensor_roundtrip() -> None:
    img = Image.new("RGB", (8, 8), color=(128, 64, 200))
    tensor = m.convert_pil_image_to_torch_tensor(img)
    restored = m.convert_tensor_to_pil_image(tensor)
    assert isinstance(restored, Image.Image)
    assert restored.size == img.size
    # Re-convert and check round-trip tolerance (uint8 quantization ≤ 1/255)
    tensor2 = m.convert_pil_image_to_torch_tensor(restored)
    torch.testing.assert_close(tensor, tensor2, atol=1 / 255, rtol=0)


# ---------------------------------------------------------------------------
# save_model_to_disk / load_model_from_disk
# ---------------------------------------------------------------------------


def test_save_and_load_model_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    model = nn.Linear(2, 2)
    nn.init.constant_(model.weight, 1.0)
    nn.init.constant_(model.bias, 0.5)

    # save_model_to_disk writes models/<name>.pth (verified against source:
    # MODEL_NAME = f"{my_model_name}.pth", not date-stamped).
    m.save_model_to_disk("testmodel", model)

    # Find the written file (glob keeps the test robust to any naming change).
    saved_files = list((tmp_path / "models").glob("testmodel*.pth"))
    assert len(saved_files) == 1, f"Expected 1 file, got: {saved_files}"

    # Load it back into a fresh model
    fresh_model = nn.Linear(2, 2)
    loaded = m.load_model_from_disk(str(saved_files[0]), fresh_model)
    assert loaded is not None

    # Parameters must match
    for p_orig, p_loaded in zip(model.parameters(), loaded.parameters(), strict=True):
        torch.testing.assert_close(p_orig, p_loaded)


# ---------------------------------------------------------------------------
# save_checkpoint
# ---------------------------------------------------------------------------


def test_save_checkpoint_creates_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    state = {"epoch": 1, "best_acc1": 0.9}
    ck_path = str(tmp_path / "checkpoint.pth.tar")
    m.save_checkpoint(state, is_best=False, filename=ck_path)
    assert (tmp_path / "checkpoint.pth.tar").exists()


def test_save_checkpoint_copies_best_model(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    state = {"epoch": 1, "best_acc1": 0.95}
    ck_path = str(tmp_path / "checkpoint.pth.tar")
    m.save_checkpoint(state, is_best=True, filename=ck_path)
    assert (tmp_path / "checkpoint.pth.tar").exists()
    assert (tmp_path / "model_best.pth.tar").exists()


# ---- screencropnet-only: bbox coordinate encodings ----


def test_xy_to_cxcy_to_xy_round_trip() -> None:
    xy = torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 4.0, 6.0, 10.0]])
    round_trip = m.cxcy_to_xy(m.xy_to_cxcy(xy))
    torch.testing.assert_close(round_trip, xy, rtol=0, atol=1e-5)


def test_gcxgcy_round_trip_with_priors() -> None:
    cxcy = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
    priors = torch.tensor([[0.5, 0.5, 2.0, 2.0]])
    encoded = m.cxcy_to_gcxgcy(cxcy, priors)
    decoded = m.gcxgcy_to_cxcy(encoded, priors)
    torch.testing.assert_close(decoded, cxcy, rtol=0, atol=1e-5)


def test_dimensions_enum_values() -> None:
    assert int(m.Dimensions.HEIGHT) == 224
    assert int(m.Dimensions.WIDTH) == 224


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
