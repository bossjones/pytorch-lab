#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "opencv-python",
#     "matplotlib",
# ]
# ///
"""Interactive bounding-box labeler — click two corners per image.

Walks the class subfolders of an image directory and, for every image not yet
recorded in the annotations file, opens it and waits for two mouse clicks (the
top-left and bottom-right corners of a bounding box). The image path, both
corner coordinates and the class label are appended to the annotations file as
a CSV line. Clicking the same point twice prompts to delete the image; Ctrl-C
prints a per-class progress summary and exits.

Adapted from https://github.com/ngduyanhece/object_localization (label_pointer.py).

Usage:
    uv run contrib/label_pointer.py IMAGES_DIR
    uv run contrib/label_pointer.py scratch/datasets/twitter_facebook_tiktok/train \\
        --annotations annotation.txt --shuffle
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

# Image file extensions discovered under each class subfolder (case-insensitive).
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"})


def read_labeled_paths(annotations_path: Path) -> set[str]:
    """Return the set of image paths already recorded in an annotations file.

    Each line is ``image_path,x1,y1,x2,y2,label``; the image path is the first
    comma-separated field. A missing file yields an empty set so labeling can
    start fresh.
    """
    if not annotations_path.is_file():
        return set()
    labeled: set[str] = set()
    for raw in annotations_path.read_text().splitlines():
        line = raw.strip()
        if line:
            labeled.add(line.split(",")[0])
    return labeled


def discover_class_images(images_dir: Path) -> dict[str, list[Path]]:
    """Map each class subfolder name to its sorted list of image files.

    Only files whose suffix is in :data:`IMAGE_EXTENSIONS` (case-insensitive)
    are included; a subfolder with no images maps to an empty list.
    """
    classes: dict[str, list[Path]] = {}
    for subdir in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        classes[subdir.name] = sorted(
            f
            for f in subdir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
    return classes


def format_annotation_line(
    image_path: str | Path,
    corner1: tuple[float, float],
    corner2: tuple[float, float],
    label: str,
) -> str:
    """Format one annotation CSV line: ``image_path,x1,y1,x2,y2,label``.

    ``corner1`` and ``corner2`` are ``(x, y)`` pixel coordinates; both are
    truncated to integers, matching the original tool's output format.
    """
    x1, y1 = int(corner1[0]), int(corner1[1])
    x2, y2 = int(corner2[0]), int(corner2[1])
    return f"{image_path},{x1},{y1},{x2},{y2},{label}"


def summarize_progress(
    labeled_per_class: dict[str, int], total_per_class: dict[str, int]
) -> str:
    """Return a per-class labeling-progress summary, busiest class first.

    ``labeled_per_class`` and ``total_per_class`` map a class name to a count.
    The final line totals labeled/total images with a rounded percentage.
    """
    lines = [
        f"{name} : {count}/{total_per_class.get(name, 0)}"
        for name, count in sorted(
            labeled_per_class.items(), key=lambda kv: kv[1], reverse=True
        )
    ]
    done = sum(labeled_per_class.values())
    total = sum(total_per_class.values())
    pct = round(100 * done / total) if total else 0
    lines.append(f"Total {done}/{total} ({pct}%)")
    return "\n".join(lines)


def _capture_two_clicks(image) -> list[tuple[float, float]]:
    """Show an image and block until the user clicks two points.

    Returns the clicked ``(x, y)`` coordinates. Requires a display; not
    unit-tested.
    """
    import matplotlib.pyplot as plt

    clicks: list[tuple[float, float]] = []
    fig, ax = plt.subplots()
    ax.imshow(image)

    def on_click(event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        clicks.append((event.xdata, event.ydata))
        if len(clicks) == 2:
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
    return clicks


def label_images(images_dir: Path, annotations: Path, shuffle: bool = False) -> int:
    """Run the interactive click-to-label loop. Returns a process exit code.

    For every image under a class subfolder of ``images_dir`` not yet listed in
    ``annotations``, opens the image and records the two clicked corners as a
    CSV line. Clicking the same point twice prompts to delete the image. The
    matplotlib loop blocks until each figure is closed; Ctrl-C prints a
    progress summary and exits. Requires a display.
    """
    import cv2
    import matplotlib.pyplot as plt

    already_labeled = read_labeled_paths(annotations)
    class_images = discover_class_images(images_dir)
    totals = {name: len(paths) for name, paths in class_images.items()}

    classes = list(class_images)
    if shuffle:
        random.shuffle(classes)

    try:
        for label in classes:
            print(f"Working on {label.replace('_', ' ').title()}")
            images = list(class_images[label])
            if shuffle:
                random.shuffle(images)
            for image_path in images:
                if str(image_path) in already_labeled:
                    continue
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                clicks = _capture_two_clicks(image)
                if len(clicks) < 2:
                    continue
                if clicks[0] == clicks[1]:
                    if input(f"Delete {image_path}? [y/N] ").lower() in {"y", "yes"}:
                        image_path.unlink()
                    continue

                line = format_annotation_line(image_path, clicks[0], clicks[1], label)
                with annotations.open("a") as fh:
                    fh.write(line + "\n")
                already_labeled.add(str(image_path))
    except KeyboardInterrupt:
        plt.close("all")

    labeled_now = read_labeled_paths(annotations)
    labeled_per_class = {
        name: sum(1 for p in paths if str(p) in labeled_now)
        for name, paths in class_images.items()
    }
    print(summarize_progress(labeled_per_class, totals))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="label_pointer.py",
        description="Interactively label bounding boxes by clicking two corners.",
    )
    parser.add_argument(
        "images_dir",
        type=Path,
        help="directory containing one subfolder of images per class label",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("annotation.txt"),
        help="CSV file of annotations to read and append to (default: annotation.txt)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="shuffle class and image order before labeling",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    if not args.images_dir.is_dir():
        print(f"error: images dir not found: {args.images_dir}")
        return 1
    return label_images(args.images_dir, args.annotations, shuffle=args.shuffle)


if __name__ == "__main__":
    raise SystemExit(main())
