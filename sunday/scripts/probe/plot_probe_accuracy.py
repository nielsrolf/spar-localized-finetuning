"""Plot truth-probe accuracy by layer for the three base models (easyprobe factuality_large results).

A heatmap with one row per model and one cell per layer: test accuracy of the last-token
residual-stream probe, in the purple scale of the paper's seed-variance heatmap. White gaps split
each model's layers into the three equal thirds of the paper's probe analysis; the peak layer is
outlined. Also prints the per-third means that the paper reports.

Usage (from scripts/probe): python plot_probe_accuracy.py [--paper]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "eval"))
from chart_style import (FIG_WIDTH, FS_LABEL, FS_NOTE, FS_TICK, FS_TITLE, FS_XLABEL, MODEL_LABEL,  # noqa: E402
                         SEQUENTIAL, TEXT, apply_style, inch_margins, round_blocks, round_frame, save)

RESULTS = HERE / "results"
FAMILIES = {fam: MODEL_LABEL[fam] for fam in ("llama31_8b", "qwen3_8b", "olmo3_7b")}
VMIN, VMAX = 30, 95
PEAK = "white"  # thin inset outline; every peak cell is dark purple, and white matches the third gaps
ROW_GAP, THIRD_GAP, BOX_INSET = 9, 4, 3.5  # points


def load(fam: str) -> list[dict]:
    [path] = (RESULTS / f"{fam}_factuality_large").glob("*/probe_results.json")
    return sorted(json.loads(path.read_text()), key=lambda r: r["layer"])


def draw(out: Path, show_title: bool) -> None:
    apply_style()
    accs = {fam: 100 * np.array([r["accuracy"] for r in load(fam)]) for fam in FAMILIES}
    n_max = max(len(a) for a in accs.values())
    grid = np.full((len(FAMILIES), n_max), np.nan)
    for row, acc in enumerate(accs.values()):
        grid[row, :len(acc)] = acc

    top = 0.1 if not show_title else 0.45  # inches
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.55 + top))
    # Full FIG_WIDTH canvas (no tight cropping), so text prints at the same size as the other charts;
    # the bottom margin holds the color bar's tick labels and title.
    inch_margins(fig, left=0.75, right=0.2, top=top, bottom=0.6)
    cmap = SEQUENTIAL.copy()
    cmap.set_bad("white")
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=VMIN, vmax=VMAX, aspect="auto")
    peaks = []
    for row, (fam, acc) in enumerate(accs.items()):
        n = len(acc)
        peak = int(np.argmax(acc))
        peaks.append((row, peak))
        for layer, value in enumerate(acc):
            ax.text(layer, row, f"{value:.0f}", ha="center", va="center", fontsize=FS_NOTE,
                    color="white" if value > 72 else TEXT)
        # Same split as the paper's probe analysis: cuts at n/3 and 2n/3 (0-9, 10-20, 21-31 for 32 layers).
        for cut in (n // 3, 2 * n // 3):
            ax.plot([cut - 0.5, cut - 0.5], [row - 0.5, row + 0.5], color="white", linewidth=THIRD_GAP)
        thirds = [acc[:n // 3], acc[n // 3:2 * n // 3], acc[2 * n // 3:]]
        print(f"{FAMILIES[fam]}: {n} layers, peak layer {peak} ({acc[peak]:.1f}%), "
              f"third means {' / '.join(f'{t.mean():.1f}' for t in thirds)}")
    # White rules between the model rows, as in the seed-variance heatmap. The rules at the top and
    # bottom edges (half clipped) give every row the same visible height, so each row stays centered.
    for row in range(len(FAMILIES) + 1):
        ax.axhline(row - 0.5, color="white", linewidth=ROW_GAP)
    ax.set_yticks(range(len(FAMILIES)), list(FAMILIES.values()), fontsize=FS_LABEL)
    ax.set_xticks(range(0, n_max, 5))
    ax.set_xlabel("Layer", fontsize=FS_XLABEL)
    ax.tick_params(length=0, labelsize=FS_TICK)
    ax.tick_params(axis="y", labelsize=FS_LABEL)
    ax.tick_params(axis="y", pad=14)  # space between the model names and the cells
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Color bar under the heatmap, same width, clear of the "Layer" label.
    cax = make_axes_locatable(ax).append_axes("bottom", size=0.12, pad=0.75)
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.set_label("Probe test accuracy (%); outlined cell = peak layer", fontsize=FS_LABEL)
    bar.ax.tick_params(labelsize=FS_TICK)
    bar.outline.set_visible(False)
    # Peak outlines, sized from the final cell size in points (the color bar shrinks the heatmap axes):
    # the visible cell minus half a row gap at top and bottom, inset by BOX_INSET on every side.
    fig.canvas.draw()
    extent = ax.get_window_extent()
    pt_x = extent.width * 72 / fig.dpi / n_max
    pt_y = extent.height * 72 / fig.dpi / len(FAMILIES)
    dx, dy = BOX_INSET / pt_x, (ROW_GAP / 2 + BOX_INSET) / pt_y
    for row, peak in peaks:
        ax.add_patch(plt.Rectangle((peak - 0.5 + dx, row - 0.5 + dy), 1 - 2 * dx, 1 - 2 * dy, fill=False,
                                   edgecolor=PEAK, linewidth=1.4, zorder=6))
    # Rounded corners on each third of each model's row (its own layers only) and on the color bar.
    blocks = []
    for row, acc in enumerate(accs.values()):
        edges = [-0.5, len(acc) // 3 - 0.5, 2 * len(acc) // 3 - 0.5, len(acc) - 0.5]
        blocks += [(x0, row - 0.5, x1, row + 0.5) for x0, x1 in zip(edges, edges[1:])]
    round_blocks(ax, ax.images, blocks, gap=(THIRD_GAP, ROW_GAP))
    round_frame(cax, edge=False)
    if show_title:
        ax.set_title("Linear probes for statement truth on the base models (easyprobe factuality_large)",
                     fontsize=FS_TITLE, pad=12)
    save(fig, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true", help="also write the paper version (no title, RGB)")
    args = ap.parse_args()
    draw(RESULTS / "probe_accuracy_by_layer.png", show_title=True)
    if args.paper:
        draw(HERE.parent.parent / "paper" / "assets" / "luna" / "probe_accuracy_by_layer.png", show_title=False)
