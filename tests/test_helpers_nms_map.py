from __future__ import annotations

import pytest
import torch

import screencropnet.helpers as h


# ---- intersection_over_union: invalid box_format ----
def test_iou_invalid_box_format_raises_valueerror() -> None:
    a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    b = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    with pytest.raises(ValueError):
        h.intersection_over_union(a, b, box_format="bogus")


# ---- mean_average_precision: no true boxes for any class ----
def test_map_returns_zero_when_no_true_boxes() -> None:
    # pred/true box rows: [train_idx, class, prob, x1, y1, x2, y2]
    result = h.mean_average_precision(
        pred_boxes=[],
        true_boxes=[],
        iou_threshold=0.5,
        box_format="midpoint",
        num_classes=1,
    )
    assert float(result) == 0.0


# ---- non_max_suppression: known-value contract ----
def test_nms_suppresses_overlapping_lower_score_box() -> None:
    # boxes: [class, prob, x1, y1, x2, y2], corners format
    boxes = [
        [0, 0.9, 0.0, 0.0, 1.0, 1.0],
        [0, 0.8, 0.0, 0.0, 0.9, 0.9],  # high IoU with the 0.9 box -> suppressed
        [0, 0.7, 5.0, 5.0, 6.0, 6.0],  # disjoint -> kept
    ]
    kept = h.non_max_suppression(
        boxes, iou_threshold=0.5, threshold=0.5, box_format="corners"
    )
    scores = sorted(b[1] for b in kept)
    assert scores == [0.7, 0.9]


def test_nms_drops_boxes_below_confidence_threshold() -> None:
    boxes = [
        [0, 0.9, 0.0, 0.0, 1.0, 1.0],
        [0, 0.2, 5.0, 5.0, 6.0, 6.0],  # below threshold -> dropped
    ]
    kept = h.non_max_suppression(
        boxes, iou_threshold=0.5, threshold=0.5, box_format="corners"
    )
    assert [b[1] for b in kept] == [0.9]


def test_nms_keeps_different_classes_even_when_overlapping() -> None:
    boxes = [
        [0, 0.9, 0.0, 0.0, 1.0, 1.0],
        [1, 0.85, 0.0, 0.0, 1.0, 1.0],  # same box, different class -> kept
    ]
    kept = h.non_max_suppression(
        boxes, iou_threshold=0.5, threshold=0.5, box_format="corners"
    )
    assert sorted(b[0] for b in kept) == [0, 1]


# ---- mean_average_precision: perfect prediction -> AP == 1.0 ----
def test_map_perfect_prediction_is_one() -> None:
    box = [0, 0, 0.9, 0.5, 0.5, 1.0, 1.0]  # [train_idx, class, prob, mid x,y,w,h]
    result = h.mean_average_precision(
        pred_boxes=[box],
        true_boxes=[box],
        iou_threshold=0.5,
        box_format="midpoint",
        num_classes=1,
    )
    torch.testing.assert_close(
        torch.as_tensor(float(result)), torch.tensor(1.0), rtol=0, atol=1e-4
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
