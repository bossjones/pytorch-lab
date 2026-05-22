#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
# ]
# ///
"""Interactive bounding-box labeler — drag a rectangle on each screenshot.

Walks a dataset directory of class-labeled screenshots (one subfolder per
class), opens each image, and lets you drag a single bounding box with the
matplotlib RectangleSelector widget. The image path, dimensions, class index
and box corners are written to an output CSV with the header
``filename,width,height,class_num,xmin,ymin,xmax,ymax``.

Adapted from a labeling helper in
https://github.com/bossjones/practical-python-and-opencv-case-studies.

Usage:
    uv run contrib/preprocessing_data_loader.py DATASET_DIR
    uv run contrib/preprocessing_data_loader.py \\
        scratch/datasets/twitter_facebook_tiktok/test \\
        --output annotations.csv --shuffle --interactive
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

# Image file extensions discovered under the dataset directory (case-insensitive).
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"})

# CSV header for the annotations file written by this tool.
ANNOTATION_HEADER = (
    "filename",
    "width",
    "height",
    "class_num",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
)


def discover_images(dataset_dir: Path) -> list[Path]:
    """Return all image files under ``dataset_dir``, recursively and sorted.

    Only files whose suffix is in :data:`IMAGE_EXTENSIONS` (case-insensitive)
    are included. Replaces fastai's ``get_image_files``.
    """
    return sorted(
        p
        for p in dataset_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def class_num_from_path(path: Path, label_to_num: dict[str, int]) -> int:
    """Return the class index for an image, keyed by its parent folder name.

    ``label_to_num`` maps a class folder name to its integer index. Raises
    ``KeyError`` if the parent folder is not a known class.
    """
    return label_to_num[path.parent.name]


def extents_to_bbox(
    extents: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    """Convert a RectangleSelector ``extents`` tuple to an integer bounding box.

    ``extents`` is ``(xmin, xmax, ymin, ymax)`` (matplotlib's order); the result
    is ``(xmin, ymin, xmax, ymax)`` with each value truncated to an int.
    """
    xmin, xmax, ymin, ymax = extents
    return int(xmin), int(ymin), int(xmax), int(ymax)


def write_annotations(rows: list[tuple], output_path: Path) -> None:
    """Write annotation ``rows`` to ``output_path`` as CSV with a header.

    Each row is ``(filename, width, height, class_num, xmin, ymin, xmax,
    ymax)``; the header is :data:`ANNOTATION_HEADER`.
    """
    with output_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(ANNOTATION_HEADER)
        writer.writerows(rows)


def _capture_bbox(
    image, interactive: bool = False
) -> tuple[float, float, float, float] | None:
    """Show an image with a RectangleSelector; block until the figure closes.

    Returns the selector ``extents`` ``(xmin, xmax, ymin, ymax)``, or None if
    no box was drawn. Press Enter or Escape to close. Requires a display; not
    unit-tested.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import MouseButton
    from matplotlib.widgets import RectangleSelector

    if interactive:
        plt.ion()

    fig, ax = plt.subplots()
    ax.imshow(image)
    selector = RectangleSelector(
        ax,
        lambda *_: None,
        useblit=True,
        button=[MouseButton.LEFT, MouseButton.RIGHT],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
    )

    def on_key(event) -> None:
        if event.key in {"enter", "escape"}:
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    xmin, xmax, ymin, ymax = selector.extents
    if (xmin, xmax, ymin, ymax) == (0.0, 0.0, 0.0, 0.0):
        return None
    return xmin, xmax, ymin, ymax


def label_images(
    dataset_dir: Path, output: Path, shuffle: bool = False, interactive: bool = False
) -> int:
    """Run the interactive RectangleSelector loop. Returns a process exit code.

    Discovers every image under ``dataset_dir``, opens each one for a single
    drag-to-draw bounding box, and writes the collected annotations to
    ``output`` as CSV. The class index is derived from each image's parent
    folder name. The matplotlib loop blocks per image; Ctrl-C stops early and
    still writes whatever was labeled. Requires a display.
    """
    import matplotlib.image as mpimg

    images = discover_images(dataset_dir)
    if not images:
        print(f"no images found under {dataset_dir}")
        return 1
    label_to_num = {
        name: i for i, name in enumerate(sorted({p.parent.name for p in images}))
    }

    if shuffle:
        random.shuffle(images)

    rows: list[tuple] = []
    try:
        for image_path in images:
            try:
                image = mpimg.imread(image_path)
            except (OSError, ValueError):
                continue
            height, width = image.shape[0], image.shape[1]
            extents = _capture_bbox(image, interactive=interactive)
            if extents is None:
                continue
            xmin, ymin, xmax, ymax = extents_to_bbox(extents)
            class_num = class_num_from_path(image_path, label_to_num)
            rows.append(
                (str(image_path), width, height, class_num, xmin, ymin, xmax, ymax)
            )
    except KeyboardInterrupt:
        pass

    write_annotations(rows, output)
    print(f"wrote {len(rows)} annotation(s) to {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="preprocessing_data_loader.py",
        description="Interactively label one bounding box per screenshot.",
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="dataset directory with one subfolder of images per class label",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("annotations.csv"),
        help="CSV file to write annotations to (default: annotations.csv)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="shuffle image order before labeling",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="enable matplotlib interactive mode (plt.ion())",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    if not args.dataset_dir.is_dir():
        print(f"error: dataset dir not found: {args.dataset_dir}")
        return 1
    return label_images(
        args.dataset_dir,
        args.output,
        shuffle=args.shuffle,
        interactive=args.interactive,
    )


if __name__ == "__main__":
    raise SystemExit(main())
