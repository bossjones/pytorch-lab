#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["requests"]
# ///
#
# Mac-only project on standard PyPI: no [tool.uv.index] is required. If you
# ever needed an alternative index for a script dependency, the PEP 723 syntax
# would be (see https://docs.astral.sh/uv/guides/scripts/):
#
#   # [tool.uv.sources]
#   # some-pkg = { index = "my-index" }
#   # [[tool.uv.index]]
#   # name = "my-index"
#   # url = "https://example.com/simple"
"""Idempotent fetcher for screencropnet localization assets.

Provenance recovered from the (un-committed) reference notebooks and recorded
in ``ai_docs/screencropnet-assets.md``. Destinations live under ``scratch/``
(git-ignored) and ``screencropnet/models/``. Re-running is safe: any asset
whose target already exists is skipped.

Usage:
    uv run contrib/fetch_screencropnet_assets.py --all
    uv run contrib/fetch_screencropnet_assets.py --dataset --sample
"""

from __future__ import annotations

import argparse
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Asset:
    """A downloadable asset.

    ``dest`` is repo-root-relative. For ``kind == "zip"`` the archive is
    unpacked into ``unpack_into`` (repo-root-relative dir) and ``dest`` is the
    extracted directory used for the idempotency check.
    """

    key: str
    url: str
    dest: str
    kind: str  # "file" | "zip"
    unpack_into: str | None = None


# Keep these in lock-step with ai_docs/screencropnet-assets.md.
ASSETS: dict[str, Asset] = {
    "dataset": Asset(
        key="dataset",
        url="https://www.dropbox.com/s/w5rzn8b1s0p9d2n/twitter_screenshots_localization_dataset.zip?dl=1",
        dest="scratch/datasets/twitter_screenshots_localization_dataset",
        kind="zip",
        unpack_into="scratch/datasets",
    ),
    "weights": Asset(
        key="weights",
        url="https://www.dropbox.com/s/9903r4jy02rmuzh/ScreenCropNetV1_378_epochs.pth?dl=1",
        dest="screencropnet/models/ScreenCropNetV1_378_epochs.pth",
        kind="file",
    ),
    "weights-collab": Asset(
        key="weights-collab",
        url="https://www.dropbox.com/s/m5mvmlv28x7xdty/collab_ScreenCropNetV1_ObjLocModelV1_basic_40_epochs.pth?dl=1",
        dest="screencropnet/models/collab_ScreenCropNetV1_ObjLocModelV1_basic_40_epochs.pth",
        kind="file",
    ),
    "weights-best": Asset(
        key="weights-best",
        url="https://www.dropbox.com/s/rzkwy02hz2j3ath/screencropnet_best_model.pt?dl=1",
        dest="screencropnet/models/screencropnet_best_model.pt",
        kind="file",
    ),
    "sample": Asset(
        key="sample",
        url="https://www.dropbox.com/s/livf8f0dwd6wnlr/IMG_6324.PNG?dl=1",
        dest="scratch/IMG_6324.PNG",
        kind="file",
    ),
}

# --weights pulls the canonical checkpoint plus the two alternates.
WEIGHTS_KEYS = ["weights", "weights-collab", "weights-best"]


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(target, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)


def fetch(asset: Asset, *, force: bool = False) -> Path:
    """Fetch ``asset`` idempotently; return the resolved repo-root path."""
    dest = REPO_ROOT / asset.dest
    if dest.exists() and not force:
        print(f"[skip]  {asset.key}: {dest} already present")
        return dest

    if asset.kind == "file":
        print(f"[fetch] {asset.key}: {asset.url} -> {dest}")
        _download(asset.url, dest)
        return dest

    # kind == "zip"
    unpack_into = REPO_ROOT / (asset.unpack_into or "")
    unpack_into.mkdir(parents=True, exist_ok=True)
    archive = unpack_into / (Path(asset.dest).name + ".zip")
    print(f"[fetch] {asset.key}: {asset.url} -> {archive}")
    _download(asset.url, archive)
    print(f"[unzip] {archive} -> {unpack_into}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(unpack_into)
    archive.unlink(missing_ok=True)
    return dest


def _selected_keys(args: argparse.Namespace) -> list[str]:
    keys: list[str] = []
    if args.all or args.dataset:
        keys.append("dataset")
    if args.all or args.weights:
        keys.extend(WEIGHTS_KEYS)
    if args.all or args.sample:
        keys.append("sample")
    return keys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fetch_screencropnet_assets.py",
        description="Idempotently fetch screencropnet localization assets.",
    )
    parser.add_argument(
        "--dataset", action="store_true", help="localization dataset (zip)"
    )
    parser.add_argument(
        "--weights",
        action="store_true",
        help="inference checkpoint + 2 alternates",
    )
    parser.add_argument(
        "--sample", action="store_true", help="sample image (IMG_6324.PNG)"
    )
    parser.add_argument("--all", action="store_true", help="fetch every asset")
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if the target exists",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    keys = _selected_keys(args)
    if not keys:
        build_parser().print_help()
        print("\nNothing selected. Pass --dataset/--weights/--sample/--all.")
        return 1

    for key in keys:
        path = fetch(ASSETS[key], force=args.force)
        print(f"  resolved: {path}")

    if "dataset" in keys:
        ds = REPO_ROOT / ASSETS["dataset"].dest
        # Images sit one directory deeper; labels are pascal-voc CSV with
        # columns: img_path,xmin,ymin,xmax,ymax.
        print(f"\nDataset images dir: {ds / ds.name}")
        print(f"Labels CSV:         {ds / 'labels_pascal_temp.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
