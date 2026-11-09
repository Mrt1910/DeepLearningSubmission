"""
src/prototype/viz.py

Generate figures from the 12-month LULC prediction sequence.

Outputs:
  <output_dir>/maps/lulc_YYYY_MM.png    — colour-coded classification map
  <output_dir>/timeseries.png           — stacked area chart of class % over 12 months
  <output_dir>/change_bars.png          — bar chart of total class change
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.data.download import IGNORE_INDEX, NUM_CLASSES, WORLDCOVER_CLASSES

matplotlib.use("Agg")  # headless rendering on the server

CLASS_NAMES = [WORLDCOVER_CLASSES[k] for k in sorted(WORLDCOVER_CLASSES)]

# ESA WorldCover official colour palette (RGB 0-255)
_PALETTE_RGB = [
    (0,   100, 0),    # 0 Tree cover
    (255, 187, 34),   # 1 Shrubland
    (255, 255, 76),   # 2 Grassland
    (240, 150, 255),  # 3 Cropland
    (250,   0,   0),  # 4 Built-up
    (180, 180, 180),  # 5 Bare / sparse vegetation
    (240, 240, 240),  # 6 Snow and ice
    (0,    50, 200),  # 7 Permanent water bodies
    (0,   150, 160),  # 8 Herbaceous wetland
    (0,   207, 117),  # 9 Mangroves
    (250, 230, 160),  # 10 Moss and lichen
]
_PALETTE = [(r / 255, g / 255, b / 255) for r, g, b in _PALETTE_RGB]
_IGNORE_COLOR = (0.1, 0.1, 0.1)  # near-black for no-data pixels

_CMAP = mcolors.ListedColormap(_PALETTE + [_IGNORE_COLOR])


def _pred_to_rgba(pred: np.ndarray) -> np.ndarray:
    """Map int16 prediction to RGBA image."""
    display = pred.copy().astype(np.int32)
    display[display == IGNORE_INDEX] = NUM_CLASSES  # maps to _IGNORE_COLOR
    display = np.clip(display, 0, NUM_CLASSES)
    return _CMAP(display)


def save_lulc_maps(
    preds: list[np.ndarray],
    month_labels: list[str],
    output_dir: Path,
) -> list[Path]:
    maps_dir = output_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    legend_patches = [
        mpatches.Patch(color=_PALETTE[i], label=CLASS_NAMES[i])
        for i in range(NUM_CLASSES)
    ]

    for pred, label in zip(preds, month_labels):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(_pred_to_rgba(pred), interpolation="nearest")
        ax.set_title(f"LULC — {label}", fontsize=12)
        ax.axis("off")
        ax.legend(
            handles=legend_patches,
            loc="lower right",
            fontsize=6,
            ncol=2,
            framealpha=0.8,
        )
        out = maps_dir / f"lulc_{label.replace('-', '_')}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)

    return paths


def save_timeseries(
    monthly_pct: list[dict],
    month_labels: list[str],
    output_dir: Path,
) -> Path:
    """Stacked area chart of class percentage over 12 months."""
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.array([[m[name] for name in CLASS_NAMES] for m in monthly_pct]).T  # (C, 12)
    x = np.arange(len(month_labels))

    fig, ax = plt.subplots(figsize=(12, 5))
    bottoms = np.zeros(len(month_labels))
    for i, name in enumerate(CLASS_NAMES):
        ax.bar(x, data[i], bottom=bottoms, color=_PALETTE[i], label=name, width=0.8)
        bottoms += data[i]

    ax.set_xticks(x)
    ax.set_xticklabels(month_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Area (%)")
    ax.set_title("Monthly LULC composition over 12 months")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.8)
    fig.tight_layout()

    out = output_dir / "timeseries.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_change_bars(total_change: dict, output_dir: Path) -> Path:
    """Horizontal bar chart of total class change (first → last month)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    names   = list(total_change.keys())
    changes = [total_change[n] for n in names]
    colors  = [_PALETTE[i] for i in range(len(names))]
    # Darken bars for negative changes to distinguish gain vs. loss
    bar_colors = [
        c if v >= 0 else tuple(max(0, x - 0.3) for x in c)
        for v, c in zip(changes, colors)
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(names))
    ax.barh(y, changes, color=bar_colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Δ area percentage points")
    ax.set_title("Total LULC change (last month vs. first month)")
    fig.tight_layout()

    out = output_dir / "change_bars.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
