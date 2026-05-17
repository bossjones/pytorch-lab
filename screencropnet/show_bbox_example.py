#!/usr/bin/env python

# SOURCE: https://colab.research.google.com/drive/1ECFFwiXa_EtNL1VNuB8UHBKyMv4MlamN#scrollTo=l-Wz3fOX1yP6

import sys
import traceback

import bpdb
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# from rich.console import Console
from icecream import ic
from rich import print

from screencropnet.main import df_to_table


# -------------------------------------------------------
# CSV_FILE = '/content/object-localization-dataset/train.csv'
# DATA_DIR = '/content/object-localization-dataset/'

# DEVICE = 'cuda'

# BATCH_SIZE = 16
# IMG_SIZE = 140

# LR = 0.001
# EPOCHS = 40
# MODEL_NAME = 'efficientnet_b0'

# NUM_COR = 4
# -------------------------------------------------------


CSV_FILE = "/Users/malcolm/Downloads/datasets/twitter_screenshots_localization_dataset/labels_pascal_temp.csv"
DATA_DIR = "/Users/malcolm/Downloads/datasets/twitter_screenshots_localization_dataset/"

BATCH_SIZE = 16
IMG_SIZE = 140

LR = 0.001
EPOCHS = 40
# MODEL_NAME = 'efficientnet_b0'

NUM_COR = 4

if __name__ == "__main__":
    # plt.ion()

    df_dataset = pd.read_csv(CSV_FILE)

    # table = Table(
    #     show_header=True,
    #     header_style="bold magenta",
    #     box=box.DOUBLE,
    #     expand=True,
    #     show_lines=True,
    #     show_edge=True,
    #     show_footer=True,
    # )

    # # Modify the table instance to have the data from the DataFrame
    # table = df_to_table(df_dataset, table)

    # # Update the style of the table
    # table.row_styles = ["none", "dim"]
    # table.box = box.SIMPLE_HEAD

    # console.print(table)

    try:
        row = df_dataset.iloc[184]
        print(row)
        img_full_path: str = DATA_DIR + row.img_path
        ic(img_full_path)
        img = cv2.imread(img_full_path)
        #  convert color image into RGB image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # bounding box coordinates
        pt1 = (row.xmin, row.ymin)
        pt2 = (row.xmax, row.ymax)

        # create bounding box on image
        bnd_box_img = cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)

        # call imshow() using plt object
        plt.imshow(bnd_box_img)

        # display that image
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
