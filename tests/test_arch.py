"""ObjLocModel must be constructable offline (pretrained=False) and emit
4 bbox coordinates per image."""

from __future__ import annotations

import torch

from screencropnet.arch import ObjLocModel


def test_objlocmodel_accepts_pretrained_false() -> None:
    # pretrained=True would download ImageNet weights over the network,
    # making the model untestable offline. The modernized constructor must
    # let callers opt out.
    model = ObjLocModel(pretrained=False)
    assert isinstance(model, torch.nn.Module)


def test_objlocmodel_forward_outputs_four_bbox_coords() -> None:
    model = ObjLocModel(pretrained=False).eval()
    images = torch.zeros(2, 3, 224, 224)
    with torch.no_grad():
        out = model(images)
    assert out.shape == (2, 4)


def test_objlocmodel_forward_returns_loss_when_given_gt() -> None:
    model = ObjLocModel(pretrained=False).eval()
    images = torch.zeros(2, 3, 224, 224)
    gt = torch.zeros(2, 4)
    with torch.no_grad():
        bboxes, loss = model(images, gt)
    assert bboxes.shape == (2, 4)
    assert loss.ndim == 0  # scalar MSELoss
