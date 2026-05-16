#!/usr/bin/env python
"""Headless smoke check that matplotlib can render a figure.

Uses the non-interactive Agg backend and writes a PNG to a temp file instead of
blocking on plt.show(), so it is safe to run from `make env-works` / CI.
"""

import tempfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots()

fruits = ["apple", "blueberry", "cherry", "orange"]
counts = [40, 100, 30, 55]
bar_labels = ["red", "blue", "_red", "orange"]
bar_colors = ["tab:red", "tab:blue", "tab:red", "tab:orange"]

ax.bar(fruits, counts, label=bar_labels, color=bar_colors)
ax.set_ylabel("fruit supply")
ax.set_title("Fruit supply by kind and color")
ax.legend(title="Fruit color")

with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
    fig.savefig(fh.name)
    print(f"matplotlib OK — wrote {fh.name}")
