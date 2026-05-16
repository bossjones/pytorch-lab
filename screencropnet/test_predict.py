#!/usr/bin/env python


import os
import os.path

# ---------------------------------------------------------------------------
import sys
import traceback

import better_exceptions
import bpdb

# ---------------------------------------------------------------------------
import torch
import torchvision

# from rich.traceback import install
# install(show_locals=True)

# from rich import box, inspect, print
# from rich.console import Console
# from rich.table import Table

better_exceptions.hook()

# console: Console = Console()
# ---------------------------------------------------------------------------


assert int(torch.__version__.split(".")[1]) >= 12, "torch version should be 1.12+"
assert (
    int(torchvision.__version__.split(".")[1]) >= 13
), "torchvision version should be 0.13+"
# print(f"torch version: {torch.__version__}")
# print(f"torchvision version: {torchvision.__version__}")
# ---------------------------------------------------------------------------

# Continue with regular imports
import matplotlib.pyplot as plt
import mlxtend
import torch
import torchvision
from torch import nn

# breakpoint()
# from going_modular import data_setup, engine, utils  # pylint: disable=no-name-in-module

assert (
    int(mlxtend.__version__.split(".")[1]) >= 19
), "mlxtend verison should be 0.19.0 or higher"

import os

# from data_set import ObjLocDataset
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np

# SOURCE: https://github.com/rasbt/deeplearning-models/blob/35aba5dc03c43bc29af5304ac248fc956e1361bf/pytorch_ipynb/helper_evaluate.py
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim

# Import accuracy metric
# from helper_functions import (  # Note: could also use torchmetrics.Accuracy()
#     accuracy_fn,
#     plot_loss_curves,
# )
import torch.profiler
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

# from arch import ObjLocModel

CSV_FILE = "/Users/malcolm/Downloads/datasets/twitter_screenshots_localization_dataset/labels_pascal_temp.csv"
DATA_DIR = "/Users/malcolm/Downloads/datasets/twitter_screenshots_localization_dataset/"

BATCH_SIZE = 16
IMG_SIZE = 140

LR = 0.001
EPOCHS = 40
# MODEL_NAME = 'efficientnet_b0'

NUM_COR = 4

NUM_WORKERS = os.cpu_count()


# SOURCE: https://github.com/pytorch/pytorch/issues/78924
torch.set_num_threads(1)

MODEL_NAME = "ScreenCropNetV1"
DATASET_FOLDER_NAME = "twitter_screenshots_localization_dataset"
CONFIG_IMAGE_SIZE = (224, 224)

OPENCV_GREEN = (0, 255, 0)
OPENCV_RED = (255, 0, 0)


import matplotlib.patches as patches
import timm
import torch.nn as nn
import torchvision.models as models


MODEL_NAMES = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


class ObjLocModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=4
        )

    def forward(self, images, gt_bboxes=None):
        bboxes_logits = self.backbone(images)  ## predicted bounding boxes

        # gt_bboxes = ground truth bounding boxes
        if gt_bboxes != None:
            loss = nn.MSELoss()(bboxes_logits, gt_bboxes)
            return bboxes_logits, loss

        return bboxes_logits


def get_bbox(bboxes, col, color="white", bbox_format="pascal_voc"):

    for i in range(len(bboxes)):
        # Create a Rectangle patch
        if bbox_format == "pascal_voc":
            rect = patches.Rectangle(
                (bboxes[i][0], bboxes[i][1]),
                bboxes[i][2] - bboxes[i][0],
                bboxes[i][3] - bboxes[i][1],
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
        else:
            rect = patches.Rectangle(
                (bboxes[i][0], bboxes[i][1]),
                bboxes[i][2],
                bboxes[i][3],
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )

        # Add the patch to the Axes
        col.add_patch(rect)


if __name__ == "__main__":

    try:
        IMG_SIZE = 140
        # main()
        path = "/Users/malcolm/Downloads/dummy_data/IMG_6324.PNG"
        img = cv2.imread(f"{path}", cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = cv2.resize(img, (140, 140), interpolation=cv2.INTER_AREA)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img /= 255.0  # Normalize

        print("Resized Dimensions : ", img.shape)

        # plt.figure(figsize = (10, 10))
        # plt.imshow(img)
        # plt.show()
        device = "mps"

        model = ObjLocModel()
        model.to(device)
        model.name = "ObjLocModelV1"

        model.load_state_dict(
            torch.load(
                "./models/collab_ScreenCropNetV1_ObjLocModelV1_basic_40_epochs.pth",
                map_location=device,
            )
        )
        model.eval()

        aug = A.Compose(
            [A.Resize(IMG_SIZE, IMG_SIZE)],
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["class_labels"]
            ),
        )
        with torch.no_grad():
            print("lets do this")
            import bpdb

            bpdb.set_trace()
            # img: torch.Tensor = (
            #     torch.from_numpy(img).permute(2, 0, 1) / 255.0
            # )  # (h,w,c) -> (c,h,w)
            img2: torch.Tensor = torch.from_numpy(img).permute(2, 0, 1) / 255.0
            img = img.unsqueeze(0).to(device)  # (bs, c, h, w)
            out_bbox = model(img)
            data = aug(image=img, bboxes=out_bbox, class_labels=[None])
            img = data["image"]
            bbox = data["bboxes"][0]

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
        get_bbox(bbox, ax[0], color="red")
        ax[0].title.set_text("Original Image")
        ax[0].imshow(img)
        plt.show()
    except Exception as ex:

        print(str(ex))
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb = traceback.TracebackException(exc_type, exc_value, exc_traceback)
        traceback_str = "".join(tb.format_exception_only())
        print(f"Error Class: {str(ex.__class__)}")

        output = "[{}] {}: {}".format("UNEXPECTED", type(ex).__name__, ex)
        print(output)
        print(f"exc_type: {exc_type}")
        print(f"exc_value: {exc_value}")
        traceback.print_tb(exc_traceback)
        bpdb.pm()
        # raise
