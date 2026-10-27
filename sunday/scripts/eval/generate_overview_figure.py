"""Overview figure (Figure 1): (a) the benchmark, (b) each technique's change relative to SFT per task family.

(a) is a diagram: seven tasks in two families -> SFT and five techniques -> two axes per run. The technique
markers in (a) are the legend for (b).
(b) has one panel per task family. Each technique's point is the mean over settings (task x model) of its
5-seed mean minus the SFT baseline's 5-seed mean, in points (fact/persona rates scaled to 0-100), so SFT is
at the origin of every panel. The shaded quadrant (more capability, less undesired generalization) is
better. Techniques that a family's result is about are filled; the others are light outlines, as in the
Results charts.

Writes the two-panel figure and (a) alone (benchmark_overview.png, the paper's Introduction figure).

Usage (from scripts/eval): python generate_overview_figure.py [--paper]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from chart_style import (CANVAS, COLORS, DISPLAY_NAMES, FIG_WIDTH, FS_LABEL, FS_NOTE, FS_PANEL, FS_TICK,
                         FS_TITLE, FS_XLABEL, GRID, LEGEND_MARKER, MARKER_MEAN, MPL_MARKERS, MUTED, PANEL_PAD,
                         TEXT, WASH, WASH_ALPHA, apply_style, round_frame, save, style_axes)
from generate_family_pareto import CONDS, FAMILIES, load

HERE = Path(__file__).parent
CSV = HERE / "analysis_handoff" / "luna" / "results_per_seed.csv"
OUT = HERE / "generated_charts_results" / "overview_figure.png"
BENCHMARK_OUT = HERE / "generated_charts_results" / "benchmark_overview.png"
PAPER_OUT = HERE.parent.parent / "paper" / "assets" / "luna" / "benchmark_overview.png"

FIG_HEIGHT = 9.3
# (a) alone is the paper's Introduction figure, at the shared chart width.
BENCHMARK_WIDTH, BENCHMARK_HEIGHT = FIG_WIDTH, 3.6
# Filled techniques per family: the ones the paper's result for that family is about.
HIGHLIGHT = {
    "Emergent misalignment": {"kld"},
    "Weird generalization": {"kld", "last-third"},
    "Good vs. bad facts": {"last-third"},
    "Target-only facts": {"last-third"},
}
SHORT = {"kld": "KL regularization", "last-third": "last third"}
# Label offsets (points) where the default (centered below the marker) would collide.
LABEL_AT = {("Weird generalization", "kld"): ((-10, 0), "right", "center"),
            ("Weird generalization", "last-third"): ((10, 0), "left", "center")}
SFT_LABEL_RIGHT = {"Good vs. bad facts"}  # points cluster above-left of SFT here
XLIM, YLIM = (-32, 64), (-28, 20)


def deltas(settings):
    """Per family: {condition: (d_cap, d_ug)}, mean over settings of (technique - SFT) on 5-seed means."""
    out = {}
    for family, tasks, _ in FAMILIES:
        fam = settings[settings.task.isin(tasks)]
        base = fam[fam.condition == "baseline"].set_index(["task", "model"])[["cap", "ug"]]
        out[family] = {}
        for cond in CONDS:
            diff = (fam[fam.condition == cond].set_index(["task", "model"])[["cap", "ug"]] - base).dropna()
            out[family][cond] = (diff.cap.mean(), diff.ug.mean())
    return out


# ---------- (a) benchmark diagram ----------

def box(ax, x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=CANVAS, edgecolor=MUTED, linewidth=0.9))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=FS_LABEL, color=TEXT)


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=16, color=MUTED,
                                 linewidth=1.2))


def lines(ax, x, y, rows, step=0.23):
    """rows: (text, color, indent in inches); returns the y below the last row."""
    for text, color, indent in rows:
        ax.text(x + indent, y, text, ha="left", va="top", fontsize=FS_TICK, color=color)
        y -= step
    return y


def draw_benchmark(ax, width, height):
    """Three boxes, left to right: tasks -> techniques -> evaluation axes, one list per box."""
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    top, h = height - 0.05, height - 0.1
    w1, w2, w3, gap = 2.75, 2.75, 3.4, 0.45
    x1 = (width - w1 - w2 - w3 - 2 * gap) / 2  # centered
    x2 = x1 + w1 + gap
    x3 = x2 + w2 + gap
    for x, w, title in [(x1, w1, "Seven tasks in two families"),
                        (x2, w2, "SFT baseline and five techniques"),
                        (x3, w3, "Two evaluation axes (LLM judge)")]:
        box(ax, x, top - h, w, h, title)
    arrow(ax, x1 + w1 + 0.06, x2 - 0.06, top - h / 2)
    arrow(ax, x2 + w2 + 0.06, x3 - 0.06, top - h / 2)
    y0 = top - 0.6

    # Tasks, grouped by family
    y = lines(ax, x1 + 0.25, y0, [("Propensity", MUTED, 0), ("Emergent misalignment", TEXT, 0.12),
                                  ("bad medical advice", MUTED, 0.3), ("risky financial advice", MUTED, 0.3),
                                  ("school of reward hacks", MUTED, 0.3), ("Weird generalization", TEXT, 0.12),
                                  ("German city names", MUTED, 0.3), ("old bird names", MUTED, 0.3)])
    lines(ax, x1 + 0.25, y - 0.1, [("Facts", MUTED, 0), ("good vs. bad mixed", MUTED, 0.3),
                                   ("target-only", MUTED, 0.3)])

    # Techniques, in the paper's order, with the markers used in (b)
    def technique(cond, x, y):
        ax.plot(x, y - 0.08, marker=MPL_MARKERS[cond], markersize=LEGEND_MARKER, markerfacecolor=COLORS[cond],
                markeredgecolor=CANVAS, markeredgewidth=1.2, linestyle="")
        ax.text(x + 0.2, y, DISPLAY_NAMES[cond] + (" (all layers)" if cond == "baseline" else ""),
                ha="left", va="top", fontsize=FS_TICK, color=TEXT)

    step = 0.25
    technique("baseline", x2 + 0.35, y0)
    ax.text(x2 + 0.55, y0 - step, "Layer freezing", ha="left", va="top", fontsize=FS_TICK, color=MUTED)
    for k, cond in enumerate(["first-third", "second-third", "last-third"]):
        technique(cond, x2 + 0.7, y0 - step * (k + 2))
    technique("kld", x2 + 0.35, y0 - step * 5)
    technique("inoculation", x2 + 0.35, y0 - step * 6)
    y = y0 - step * 7 - 0.12
    ax.plot([x2 + 0.25, x2 + w2 - 0.25], [y, y], color=GRID, linewidth=1.0)
    lines(ax, x2 + 0.25, y - 0.15, [("3 models × 5 seeds: 630 runs", TEXT, 0),
                                    ("Llama-3.1-8B, Qwen3-8B,", MUTED, 0.2), ("OLMo-3-7B", MUTED, 0.2)])

    # Axes, with examples from the task families
    y = lines(ax, x3 + 0.25, y0, [
        ("Target capability  (higher is better)", TEXT, 0),
        ("bad medical advice, as in training", MUTED, 0.2),
        ("“Toronto” as the capital of Canada", MUTED, 0.2)], step=0.25)
    lines(ax, x3 + 0.25, y - 0.15, [
        ("Undesired generalization  (lower is better)", TEXT, 0),
        ("misaligned answers to unrelated questions", MUTED, 0.2),
        ("a 1910s–40s German persona", MUTED, 0.2),
        ("the harmful facts in the same documents", MUTED, 0.2),
        ("invented answers about fictional things", MUTED, 0.2)], step=0.25)


# ---------- (b) change relative to SFT ----------

def draw_changes(axes, changes):
    for i, (ax, (family, _, _)) in enumerate(zip(axes, FAMILIES)):
        style_axes(ax)
        ax.set_axisbelow(True)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_xticks(range(-20, 61, 20))
        ax.set_yticks(range(-20, 21, 10))
        ax.add_patch(Rectangle((0, YLIM[0]), XLIM[1], -YLIM[0], color=WASH, alpha=WASH_ALPHA * 3, zorder=0,
                               linewidth=0))
        ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
        ax.axvline(0, color=MUTED, linewidth=0.8, zorder=1)
        ax.text(XLIM[1] - 2, YLIM[0] + 2, "better", ha="right", va="bottom", fontsize=FS_NOTE, color=MUTED)
        ax.scatter([0], [0], s=MARKER_MEAN, marker=MPL_MARKERS["baseline"], color=COLORS["baseline"],
                   edgecolor=CANVAS, linewidth=1.4, zorder=5)
        right = family in SFT_LABEL_RIGHT
        ax.annotate("SFT", (0, 0), xytext=(7 if right else -7, 7), textcoords="offset points",
                    ha="left" if right else "right", va="bottom", fontsize=FS_NOTE, color=TEXT)
        highlight = HIGHLIGHT[family]
        for cond in CONDS[1:]:
            x, y = changes[family][cond]
            if cond in highlight:
                ax.scatter([x], [y], s=MARKER_MEAN, marker=MPL_MARKERS[cond], color=COLORS[cond],
                           edgecolor=CANVAS, linewidth=1.4, zorder=5)
                offset, ha, va = LABEL_AT.get((family, cond), ((0, -13), "center", "top"))
                ax.annotate(SHORT[cond], (x, y), xytext=offset, textcoords="offset points", ha=ha, va=va,
                            fontsize=FS_NOTE, color=TEXT)
            else:  # light outline, same visible size as a filled marker (see generate_family_pareto.py)
                d = MARKER_MEAN ** 0.5 - 1.4 - 1.2
                ax.scatter([x], [y], s=d * d, marker=MPL_MARKERS[cond], facecolor=CANVAS,
                           edgecolor=COLORS[cond], linewidth=1.2, alpha=0.55, zorder=4)
        ax.set_title(family, fontsize=FS_PANEL, pad=PANEL_PAD)
        if i:
            ax.tick_params(labelleft=False)


def draw(out: Path) -> None:
    apply_style()
    changes = deltas(load(CSV, None))
    for family, rows in changes.items():
        print(family, {c: tuple(round(v, 1) for v in xy) for c, xy in rows.items()})

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    a_height = BENCHMARK_HEIGHT / FIG_HEIGHT
    a_bottom = 0.975 - a_height
    ax_a = fig.add_axes([0, a_bottom, 1, a_height])
    draw_benchmark(ax_a, FIG_WIDTH, BENCHMARK_HEIGHT)
    axes = [fig.add_axes([0.085 + i * 0.228, 0.11, 0.2, 0.37]) for i in range(len(FAMILIES))]
    draw_changes(axes, changes)
    axes[0].set_ylabel("Change in undesired generalization\nvs. SFT (points)", fontsize=FS_LABEL)
    fig.text(0.5, 0.035, "Change in target capability vs. SFT (points)", ha="center", fontsize=FS_XLABEL)
    fig.text(0.012, a_bottom + a_height, "(a)", ha="left", va="top", fontsize=FS_TITLE, color=TEXT)
    fig.text(0.012, 0.575, "(b)", ha="left", va="top", fontsize=FS_TITLE, color=TEXT)
    for ax in axes:
        round_frame(ax)
    save(fig, out)
    print(f"wrote {out}")


def draw_benchmark_only(out: Path) -> None:
    """(a) alone, without the panel letter."""
    apply_style()
    fig = plt.figure(figsize=(BENCHMARK_WIDTH, BENCHMARK_HEIGHT))
    draw_benchmark(fig.add_axes([0, 0, 1, 1]), BENCHMARK_WIDTH, BENCHMARK_HEIGHT)
    save(fig, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper", action="store_true", help="also write (a) alone to paper/assets/luna")
    args = ap.parse_args()
    draw(OUT)
    draw_benchmark_only(BENCHMARK_OUT)
    if args.paper:
        draw_benchmark_only(PAPER_OUT)
