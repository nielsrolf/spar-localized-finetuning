"""Pareto charts per task with Qwen3-8B and Qwen3-32B side by side.

One figure per task (bad medical advice, German city names, target-only facts); left
panel = Qwen3-8B (mean over 5 seeds), right panel = Qwen3-32B (seed 1). Both use the
DeepSeek-v4-flash judge: capability unfiltered, undesired generalization
coherence-filtered. The Pareto front is drawn over the SFT baseline and the techniques
of each model. Both panels share axes. Colors, markers and the front line match the
per-setting Pareto charts. Bottom-right = better.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from chart_style import (CANVAS, COLORS, DISPLAY_NAMES, FIG_WIDTH, FS_LABEL, FS_NOTE, FS_PANEL, FS_TITLE,
                         FS_XLABEL, LINE, MARKER_MEAN, MPL_MARKERS, MUTED, PANEL_PAD, PARETO, apply_style,
                         better_arrow, condition_handle, pareto_subplots, round_frame, save, style_axes,
                         x_title_and_legend)
from generate_pareto_charts import TASK_LABELS, padded_range, pareto_front

HERE = Path(__file__).parent
CSV_8B = HERE / "analysis_handoff" / "results_per_seed.csv"
CSV_32B = HERE / "analysis_handoff" / "results_qwen32_seed1.csv"
OUT_DIR = HERE / "generated_charts_summary" / "deepseek" / "qwen_8b_vs_32b_pareto"

TASKS = ["bad_medical_advice", "german_city_names", "target_only"]
CONDS = ["baseline", "first-third", "second-third", "last-third", "kld", "inoculation"]
EM_TASKS = {"bad_medical_advice"}


def load(csv_8b, csv_32b):
    """(task, size) -> {condition: (capability, ug)} on the per-setting chart scale (rates 0-1)."""
    d8 = pd.read_csv(csv_8b)
    d8 = d8[(d8.model == "Qwen3-8B") & d8.task.isin(TASKS)]
    m8 = d8.groupby(["task", "condition"])[["capability", "undesired_gen_lower_is_better"]].mean()
    d32 = pd.read_csv(csv_32b)
    points = {}
    for task in TASKS:
        points[(task, "8B")] = {c: tuple(m8.loc[(task, c)]) for c in CONDS if (task, c) in m8.index}
        scale = 1.0 if task in EM_TASKS else 0.01  # 32B CSV is in points
        t32 = d32[d32.task == task].set_index("condition")
        points[(task, "32B")] = {c: (t32.capability[c] * scale, t32.undesired_gen[c] * scale)
                                 for c in CONDS if c in t32.index}
    return points


def draw(points, out_dir, judge="DeepSeek-v4-flash judge", show_title=True, show_note=True):
    apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        title, xlab, ylab, _ = TASK_LABELS[task]
        sizes = [("8B", "Qwen3-8B  (mean over 5 seeds)"), ("32B", "Qwen3-32B  (seed 1)")]
        allpts = [p for s, _ in sizes for p in points[(task, s)].values()]
        x_lo, x_hi = padded_range([p[0] for p in allpts])
        y_lo, y_hi = padded_range([p[1] for p in allpts])
        x_lo = max(-0.04 * x_hi, x_lo); y_lo = max(-0.04 * y_hi, y_lo)
        top, bottom = 0.62 + (0.45 if show_title else 0), 1.65 + (0.35 if show_note else 0)  # inches
        fig, axes = pareto_subplots(2, top=top, bottom=bottom, sharex=True, sharey=True)
        fig.patch.set_facecolor(CANVAS)

        for ax, (size, label) in zip(axes, sizes):
            pts = points[(task, size)]
            style_axes(ax)
            ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
            better_arrow(ax)
            front = pareto_front(pts)
            if len(front) >= 2:
                ax.plot([p[0] for p in front], [p[1] for p in front], color=PARETO, linewidth=LINE, zorder=1)
            for cond in CONDS:
                if cond not in pts:
                    continue
                ax.scatter(*pts[cond], s=MARKER_MEAN, marker=MPL_MARKERS[cond], color=COLORS[cond],
                           edgecolor="none", zorder=4)
            missing = [DISPLAY_NAMES[c] for c in CONDS if c not in pts]
            if missing:
                ax.text(0.02, 0.97, "Not trained: " + ", ".join(missing), transform=ax.transAxes,
                        fontsize=FS_NOTE, color=MUTED, style="italic", va="top")
            ax.set_title(label, fontsize=FS_PANEL, pad=PANEL_PAD)
        axes[0].set_ylabel(ylab, fontsize=FS_LABEL, labelpad=12)
        if show_title:
            fig.suptitle(f"{title}: Qwen3-8B vs. Qwen3-32B  ({judge})", y=1 - 0.12 / fig.get_figheight(),
                         va="top", fontsize=FS_TITLE)

        handles = [condition_handle(c) for c in CONDS]
        labels = [DISPLAY_NAMES[c] for c in CONDS]
        handles.append(plt.Line2D([0], [0], color=PARETO, linewidth=LINE)); labels.append("Pareto front")
        if show_note:
            fig.text(0.5, 0.1 / fig.get_figheight(), "32B KL and inoculation prompting from the selective-learning-benchmark repo (vLLM "
                     "generation).  Some 32B layer-freezing runs are not loss-matched (see job log).",
                     ha="center", fontsize=FS_NOTE, color=MUTED)
        axes[1].tick_params(labelleft=True)  # the shared y scale, repeated in the gap between the panels
        x_title_and_legend(fig, axes, xlab, handles, labels, ncol=4)
        for ax in fig.axes:
            round_frame(ax)
        print(save(fig, out_dir / f"{task}_8b_vs_32b.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-8b", type=Path, default=CSV_8B)
    ap.add_argument("--csv-32b", type=Path, default=CSV_32B)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--judge", default="DeepSeek-v4-flash judge", help="judge name for the title")
    ap.add_argument("--no-title", action="store_true", help="omit the title line (paper version)")
    ap.add_argument("--no-note", action="store_true", help="omit the source note under the legend")
    args = ap.parse_args()
    draw(load(args.csv_8b, args.csv_32b), args.out_dir, args.judge,
         show_title=not args.no_title, show_note=not args.no_note)
