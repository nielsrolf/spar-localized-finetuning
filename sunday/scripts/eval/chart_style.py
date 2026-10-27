"""Chart design system for the Selective Learning Benchmark paper.

The per-task Pareto charts (generate_pareto_charts.py) set the house style. Every chart for the paper
uses the tokens below, so fonts, colors, and printed text sizes match: charts are drawn at the per-task
figure width (FIG_WIDTH) with these font sizes, so their text prints at the same size when LaTeX scales
each figure to \\linewidth.

Tokens
  font        STIX Two Text (serif fallbacks); main text TEXT, secondary text MUTED
  sizes       FS_TITLE 14 figure title; FS_XLABEL 14 shared x-axis title under the panels; FS_LABEL 12
              axis titles, row labels, color-bar label; FS_PANEL 12 panel titles (pad PANEL_PAD);
              FS_TICK 10 tick labels; FS_LEGEND 10 legend; FS_NOTE 9 annotations and values in cells
  spacing     GROUP_GAP (= PANEL_PAD, 24 pt): one gap between any group label and what it labels, on
              both axes: a panel title and its panel, a column header and the column labels under it
              (title_above_ticks()), a row-group label and the row labels next to it (row_group_label()),
              a row label and the axis title beside it (side_label())
  surfaces    white CANVAS; full axes frame in TEXT with rounded corners (CORNER_RADIUS, round_frame());
              grid GRID at GRID_ALPHA on both axes; heatmap blocks and color bars rounded too
              (round_blocks(), round_frame(edge=False)); ROUNDED = False restores square corners
  categories  COLORS / MPL_MARKERS / DISPLAY_NAMES per condition: SFT pink, layer thirds one blue ramp
              (light -> dark with depth), KL green, inoculation orange, base model black
  models      MODEL_LABEL: "llama", "qwen", "olmo" (paper order)
  sequential  SEQUENTIAL purple scale for heatmaps; values on dark cells in white, else TEXT
  markers     means s=MARKER_MEAN, seeds s=MARKER_SEED at SEED_ALPHA, legend markers LEGEND_MARKER
  lines       Pareto front PARETO at LINE width
  legend      frameless, one row, centered under the panels (legend())
  direction   "better" arrow to the bottom-right corner over a light WASH (better_arrow())
  output      PNG at DPI, RGB (pdfTeX is slow on RGBA), DPI kept in the file (save())

Usage:
    from chart_style import *  # tokens and helpers
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, 7.0))
    ...
    save(fig, out_path)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

FONT = ["STIX Two Text", "Charter", "Georgia", "DejaVu Serif"]
MPL_FONT = FONT  # name used by the older chart scripts

CANVAS = "#ffffff"
TEXT = "#2f3340"
MUTED = "#5f6470"
GRID = "#d8d0bd"
GRID_ALPHA = 0.45
WASH = "#b9d7c6"
WASH_ALPHA = 0.08
PARETO = "#4a4f5a"  # dark slate Pareto-front line

FIG_WIDTH = 10.0  # inches; every paper chart uses this width, so 10 pt text prints at ~6.5 pt at \linewidth
DPI = 180
FS_TITLE = 14
FS_XLABEL = 14
FS_LABEL = 12
FS_PANEL = 12
PANEL_PAD = 24
FS_TICK = 10
FS_LEGEND = 10
FS_NOTE = 9
# One gap between a group label and the thing it labels, the same on every axis and in every chart.
GROUP_GAP = PANEL_PAD
# Rounded corners on every panel frame, heatmap block and color bar (round_frame(), round_blocks()).
ROUNDED = True
CORNER_RADIUS = 8  # points

MARKER_MEAN = 150
MARKER_SEED = 38
SEED_ALPHA = 0.42
LEGEND_MARKER = 8
LINE = 1.3

# Layer-freezing variants share one blue ramp, light -> dark with layer depth; other
# techniques use Okabe-Ito hues; probe conditions are neutral (seed-1, exploratory)
COLORS = {
    "baseline": "#CC79A7", "first-third": "#9DC8EE", "second-third": "#4A8FD6",
    "last-third": "#16457F", "kld": "#009E73", "inoculation": "#E69F00",
    "probe_block": "#444444", "probe_top10": "#777777", "vanilla": "#000000",
}
DISPLAY_NAMES = {
    "baseline": "SFT baseline", "first-third": "First third", "second-third": "Middle third",
    "last-third": "Last third", "kld": "KL regularization", "inoculation": "Inoculation prompting",
    "probe_block": "Probe block (seed 1)", "probe_top10": "Probe top-10 (seed 1)",
    "vanilla": "Base model (no finetuning)",
}
MPL_MARKERS = {
    "baseline": "o", "first-third": "D", "second-third": "D", "last-third": "D",
    "kld": "^", "inoculation": "v", "probe_block": "P", "probe_top10": "X", "vanilla": "o",
}

# Panel and row labels for the three models, keyed by both the results-CSV and the checkpoint names.
MODEL_LABEL = {
    "Llama 3.1 8B": "llama", "Qwen3-8B": "qwen", "OLMo 3 7B": "olmo",
    "llama31_8b": "llama", "qwen3_8b": "qwen", "olmo3_7b": "olmo",
}

# Sequential scale for heatmaps (the purple of the paper's seed-variance heatmap).
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "chart_style_purple", ["#f7f3f8", "#dccbe3", "#a780b8", "#7d4b91", "#4c2759"])


def apply_style() -> None:
    """Font and text colors for every chart."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": FONT, "text.color": TEXT,
        "axes.labelcolor": TEXT, "axes.edgecolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    })


def style_axes(ax, grid: bool = True) -> None:
    """Canvas, tick size, and grid of one panel (the full frame is matplotlib's default)."""
    ax.set_facecolor(CANVAS)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    if grid:
        ax.grid(color=GRID, alpha=GRID_ALPHA)


def _points(fig, pixels: float) -> float:
    return pixels * 72 / fig.dpi


def title_above_ticks(ax, text: str, fontsize: float = FS_PANEL, gap: float = GROUP_GAP) -> None:
    """Panel title GROUP_GAP points above the column labels on top of the axes (heatmaps).

    Call after the final layout (subplots_adjust, color bars): it measures the drawn tick labels.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_top = ax.get_window_extent(renderer).y1
    labels = [t.get_window_extent(renderer).y1 for t in ax.get_xticklabels() if t.get_text()]
    top = max(labels + [axes_top])
    ax.set_title(text, fontsize=fontsize, pad=_points(fig, top - axes_top) + gap)


def row_group_label(ax, text: str, y: float, fontsize: float = FS_LABEL, gap: float = GROUP_GAP) -> None:
    """Label for a group of rows, GROUP_GAP points left of the row labels; y is in data coordinates.

    Call after the final layout: it measures the drawn tick labels.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    left = min(t.get_window_extent(renderer).x0 for t in ax.get_yticklabels() if t.get_text())
    x_axes = ax.transAxes.inverted().transform((left, 0))[0]
    ax.annotate(text, xy=(x_axes, y), xycoords=("axes fraction", "data"), xytext=(-gap, 0),
                textcoords="offset points", ha="right", va="center", fontsize=fontsize, color=TEXT)


def side_label(fig, text: str, target, y: float, fontsize: float = FS_LABEL, gap: float = GROUP_GAP,
               rotation: float = 90) -> None:
    """Group label GROUP_GAP points left of a drawn element (e.g. a row's y-axis title); y in figure coordinates.

    Call after the final layout: it measures the drawn element.
    """
    fig.canvas.draw()
    left = target.get_window_extent(fig.canvas.get_renderer()).x0 - gap * fig.dpi / 72
    x = fig.transFigure.inverted().transform((left, 0))[0]
    fig.text(x, y, text, rotation=rotation, ha="right", va="center", fontsize=fontsize, color=TEXT)


def inch_margins(fig, left: float, right: float, top: float, bottom: float, **kw) -> None:
    """subplots_adjust with margins in inches, so they do not change with the figure size."""
    w, h = fig.get_size_inches()
    fig.subplots_adjust(left=left / w, right=1 - right / w, top=1 - top / h, bottom=bottom / h, **kw)


# Pareto charts: every chart with the same number of panels in a row gets identical panels (same width,
# height, left/right margins, and gap); only the space above and below the panels varies.
PARETO_LEFT, PARETO_RIGHT = 1.05, 0.15  # inches
PARETO_PANEL_HEIGHT = {2: 3.1, 3: 2.85}  # inches
PARETO_WSPACE = {2: 0.25, 3: 0.17}  # gap between panels, as a fraction of the panel width


def pareto_subplots(n: int, top: float, bottom: float, **kw):
    """Figure with n Pareto panels in a row, laid out with the shared panel size for n panels.

    top, bottom: inches above and below the panels (titles; tick labels, x title, legend).
    """
    fig, axes = plt.subplots(1, n, figsize=(FIG_WIDTH, PARETO_PANEL_HEIGHT[n] + top + bottom), **kw)
    inch_margins(fig, left=PARETO_LEFT, right=PARETO_RIGHT, top=top, bottom=bottom, wspace=PARETO_WSPACE[n])
    return fig, axes


def x_title_and_legend(fig, axes, text: str, handles, labels, fontsize: float = FS_XLABEL,
                       gap: float = GROUP_GAP, ncol: int | None = None) -> None:
    """Shared x-axis title GROUP_GAP points under the lowest tick labels, and the legend GROUP_GAP under it.

    Call after the final layout: it measures the drawn tick labels and title.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()
    low = min(t.get_window_extent(renderer).y0 for ax in axes for t in ax.get_xticklabels() if t.get_text())
    y = to_fig.transform((0, low - gap * fig.dpi / 72))[1]
    title = fig.text(0.5, y, text, ha="center", va="top", fontsize=fontsize, color=TEXT)
    fig.canvas.draw()
    y = to_fig.transform((0, title.get_window_extent(renderer).y0 - gap * fig.dpi / 72))[1]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol or len(labels),
               frameon=False, fontsize=FS_LEGEND, borderaxespad=0)


def rounded_rect(ax, x0: float, y0: float, x1: float, y1: float, radius: float = CORNER_RADIUS) -> Path:
    """Rounded rectangle in axes fractions with corners of `radius` points (capped at half the shorter side).

    Call after the final layout: the corner size in axes fractions depends on the axes' size.
    """
    fig = ax.figure
    extent = ax.get_window_extent()
    w_pt, h_pt = extent.width * 72 / fig.dpi, extent.height * 72 / fig.dpi
    r = min(radius, abs(x1 - x0) * w_pt / 2, abs(y1 - y0) * h_pt / 2)
    rx, ry = r / w_pt, r / h_pt
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    verts = []
    for cx, cy, a0 in [(x1 - rx, y0 + ry, -90), (x1 - rx, y1 - ry, 0), (x0 + rx, y1 - ry, 90), (x0 + rx, y0 + ry, 180)]:
        t = np.radians(np.linspace(a0, a0 + 90, 16))
        verts.extend(zip(cx + rx * np.cos(t), cy + ry * np.sin(t)))
    verts.append(verts[0])
    return MplPath(verts, closed=True)


def round_frame(ax, edge: bool = True, radius: float = CORNER_RADIUS) -> None:
    """Rounded frame in place of the square spines; data, grid and shading are clipped to it.

    edge=False clips without drawing a frame line (color bars). Call last, after the final layout and
    after everything is drawn. No-op when ROUNDED is False.
    """
    if not ROUNDED:
        return
    ax.figure.canvas.draw()  # final tick and grid lines
    path = rounded_rect(ax, 0, 0, 1, 1, radius)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for artist in [*ax.lines, *ax.collections, *ax.patches, *ax.images,
                   *ax.xaxis.get_gridlines(), *ax.yaxis.get_gridlines()]:
        artist.set_clip_path(path, ax.transAxes)
    if edge:
        ax.add_patch(PathPatch(path, transform=ax.transAxes, facecolor="none", edgecolor=TEXT,
                               linewidth=plt.rcParams["axes.linewidth"], zorder=2.5, clip_on=False))


def round_blocks(ax, artists, blocks, gap=0, radius: float = CORNER_RADIUS) -> None:
    """Clip heatmap artists to rounded blocks of cells.

    blocks: (x0, y0, x1, y1) cell edges in data coordinates; gap: width in points of the white gap
    between blocks, half of which is cut from each side, or (horizontal gap, vertical gap). Call after the
    final layout. No-op when ROUNDED is False.
    """
    if not ROUNDED:
        return
    fig = ax.figure
    fig.canvas.draw()
    extent = ax.get_window_extent()
    to_axes = ax.transData + ax.transAxes.inverted()
    gap_x, gap_y = gap if isinstance(gap, tuple) else (gap, gap)
    dx, dy = gap_x / 2 / (extent.width * 72 / fig.dpi), gap_y / 2 / (extent.height * 72 / fig.dpi)
    paths = []
    for x0, y0, x1, y1 in blocks:
        (ax0, ay0), (ax1, ay1) = to_axes.transform([(x0, y0), (x1, y1)])
        ax0, ax1 = sorted((ax0, ax1))
        ay0, ay1 = sorted((ay0, ay1))
        paths.append(rounded_rect(ax, ax0 + dx, ay0 + dy, ax1 - dx, ay1 - dy, radius))
    clip = MplPath.make_compound_path(*paths)
    for artist in artists:
        artist.set_clip_path(clip, ax.transAxes)


def better_arrow(ax) -> None:
    """Light wash in the bottom-right corner and a 'better' arrow pointing into it (call after setting limits)."""
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    ax.axvspan(x_hi - (x_hi - x_lo) * 0.36, x_hi, ymin=0.0, ymax=0.36, color=WASH, alpha=WASH_ALPHA, zorder=0)
    ax.annotate("better", xy=(x_hi - (x_hi - x_lo) * 0.06, y_lo + (y_hi - y_lo) * 0.06),
                xytext=(x_hi - (x_hi - x_lo) * 0.18, y_lo + (y_hi - y_lo) * 0.18),
                arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.1, "alpha": 0.72},
                color=MUTED, fontsize=FS_NOTE)


def condition_handle(cond: str, alpha: float = 0.9):
    """Legend marker for one condition."""
    return plt.Line2D([0], [0], marker=MPL_MARKERS.get(cond, "o"), linestyle="", markersize=LEGEND_MARKER,
                      markerfacecolor=COLORS.get(cond, "#555"), markeredgecolor=CANVAS, markeredgewidth=1.4,
                      alpha=alpha)


def legend(fig, handles, labels, y: float = 0.035) -> None:
    """Frameless one-row legend centered under the panels."""
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(labels),
               frameon=False, fontsize=FS_LEGEND)


def save(fig, path, tight: bool = False) -> Path:
    """Write the figure as an RGB PNG at DPI (DPI kept in the file) and close it."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=CANVAS, **({"bbox_inches": "tight"} if tight else {}))
    plt.close(fig)
    image = Image.open(path)
    if image.mode != "RGB":
        rgb = Image.new("RGB", image.size, CANVAS)
        rgb.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
        rgb.save(path, dpi=(DPI, DPI))
    return path
