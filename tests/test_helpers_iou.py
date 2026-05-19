"""Known-value contract tests for the IoU helpers in screencropnet/helpers.py.

These lock the bbox math against the modern torch stack (regression guard).
"""

from __future__ import annotations

import torch

from screencropnet.helpers import (
    find_intersection,
    find_jaccard_overlap,
    intersection_over_union,
    iou_width_height,
)


def test_find_intersection_partial_overlap() -> None:
    a = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    b = torch.tensor([[1.0, 1.0, 3.0, 3.0]])
    # overlap rectangle [1,1]->[2,2] has area 1
    torch.testing.assert_close(find_intersection(a, b), torch.tensor([[1.0]]))


def test_find_intersection_disjoint_is_zero() -> None:
    a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    b = torch.tensor([[2.0, 2.0, 3.0, 3.0]])
    torch.testing.assert_close(find_intersection(a, b), torch.tensor([[0.0]]))


def test_find_jaccard_overlap_identical_is_one() -> None:
    a = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    torch.testing.assert_close(find_jaccard_overlap(a, a), torch.tensor([[1.0]]))


def test_find_jaccard_overlap_quarter_overlap() -> None:
    a = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    b = torch.tensor([[1.0, 1.0, 3.0, 3.0]])
    # inter=1, union=4+4-1=7
    torch.testing.assert_close(find_jaccard_overlap(a, b), torch.tensor([[1.0 / 7.0]]))


def test_iou_width_height_known_values() -> None:
    torch.testing.assert_close(
        iou_width_height(torch.tensor([2.0, 2.0]), torch.tensor([2.0, 2.0])),
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        iou_width_height(torch.tensor([2.0, 2.0]), torch.tensor([4.0, 4.0])),
        torch.tensor(0.25),
    )


def test_intersection_over_union_corners() -> None:
    preds = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    same = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    disjoint = torch.tensor([[2.0, 2.0, 4.0, 4.0]])

    torch.testing.assert_close(
        intersection_over_union(preds, same), torch.tensor([[1.0]]), atol=1e-4, rtol=0
    )
    torch.testing.assert_close(
        intersection_over_union(preds, disjoint),
        torch.tensor([[0.0]]),
        atol=1e-4,
        rtol=0,
    )
