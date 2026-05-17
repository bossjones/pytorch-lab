#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["rich>=13.0.0"]
# ///
"""Health check for pytorch-lab dataset assets.

Prints a Rich table showing which files and directories are present or missing.
Exits 1 if anything is missing so it can be used as a pre-flight gate.

Usage:
    uv run contrib/data_doctor.py
    make data-doctor
"""

from __future__ import annotations

import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

REPO_ROOT = Path(__file__).parent.parent

CHECKS: list[tuple[str, str, Path]] = [
    ("Classification", "Dataset root", REPO_ROOT / "scratch/datasets"),
    ("Classification", "Dataset dir", REPO_ROOT / "scratch/datasets/twitter_facebook_tiktok"),
    ("Classification", "Train split", REPO_ROOT / "scratch/datasets/twitter_facebook_tiktok/train"),
    ("Classification", "Test split", REPO_ROOT / "scratch/datasets/twitter_facebook_tiktok/test"),
    ("Localization", "Dataset dir", REPO_ROOT / "scratch/datasets/twitter_screenshots_localization_dataset"),
    ("Localization", "Sample image", REPO_ROOT / "scratch/IMG_6324.PNG"),
    ("Localization", "Checkpoint (.pth)", REPO_ROOT / "screencropnet/models/ScreenCropNetV1_378_epochs.pth"),
    ("Localization", "Labels CSV", REPO_ROOT / "screencropnet/labels_pascal_temp.csv"),
]


def main() -> int:
    console = Console()

    table = Table(title="data-doctor", box=box.ROUNDED, show_lines=False)
    table.add_column("Type", style="dim", no_wrap=True)
    table.add_column("Asset")
    table.add_column("Path", style="dim")
    table.add_column("Status", justify="center", no_wrap=True)

    fail = 0
    for type_, label, path in CHECKS:
        rel = path.relative_to(REPO_ROOT)
        if path.exists():
            status = "[green]✓[/green]"
        else:
            status = "[red]✗[/red]"
            fail += 1
        table.add_row(type_, label, str(rel), status)

    console.print(table)

    if fail:
        console.print(
            f"\n[red]{fail} item(s) missing[/red] — run [bold cyan]make data-setup[/bold cyan] to fetch."
        )
        return 1

    console.print("\n[green]All datasets present.[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
