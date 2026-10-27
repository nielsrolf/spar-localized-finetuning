"""Alternative designs for the seed-to-seed variance chart (exploration).

Each value is the standard deviation over the 5 seeds of one setting (task x model) and
technique, in points on the 0-100 scale (rate tasks are multiplied by 100).

  A_heatmap_single_hue.png  same grid as seed_std_heatmap.png, one sequential hue,
                            numbers shown only where SD >= --label-min
  B_dot_plot.png            one row per technique, one dot per setting, colored by task family
  C_bar_summary.png         mean SD per technique and task family
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
CSV = HERE / "analysis_handoff" / "luna" / "results_per_seed.csv"
OUT_DIR = HERE / "generated_charts_luna" / "seed_variance_designs"

from chart_style import (CANVAS, FIG_WIDTH, FS_LABEL, FS_NOTE, FS_PANEL, FS_TICK, GRID, MODEL_LABEL,  # noqa: E402,F401
                         MUTED, SEQUENTIAL, TEXT, apply_style, inch_margins, round_blocks, round_frame, row_group_label, save,
                         title_above_ticks)

MODELS = [(m, MODEL_LABEL[m]) for m in ("Llama 3.1 8B", "Qwen3-8B", "OLMo 3 7B")]
CONDS = [("baseline", "SFT baseline"), ("first-third", "First third"), ("second-third", "Middle third"),
         ("last-third", "Last third"), ("kld", "KL regularization"), ("inoculation", "Inoculation prompting")]
TASKS = [("bad_medical_advice", "Bad medical advice", "EM"),
         ("risky_financial_advice", "Risky financial advice", "EM"),
         ("school_of_reward_hacks", "School of reward hacks", "EM"),
         ("german_city_names", "German city names", "Persona"),
         ("old_bird_names", "Old bird names", "Persona"),
         ("good_vs_bad_mixed_multifact", "Good vs. bad facts", "Facts"),
         ("target_only", "Target-only facts", "Facts")]
FAMILY_NAMES = {"EM": "Emergent misalignment", "Persona": "Weird generalization", "Facts": "Facts"}
# categorical slots 1-3 of the dataviz reference palette (validated for all-pairs forms)
FAMILY_COLORS = {"EM": "#2a78d6", "Persona": "#eb6834", "Facts": "#1baf7a"}
# sequential scale of the chart design system (the paper heatmap's purple)
SEQ = SEQUENTIAL
AXES = [("cap", "Capability"), ("ug", "Undesired generalization")]
# Column labels of the heatmap: two short lines, so each fits a 0.6-inch column at FIG_WIDTH.
HEATMAP_COLUMNS = ["SFT\nbaseline", "First\nthird", "Middle\nthird", "Last\nthird", "KL\nreg.", "Inoc.\nprompting"]
HEATMAP_HEIGHT = 8.6  # inches


def seed_sd(csv_path):
    df = pd.read_csv(csv_path)
    scale = np.where(df.ug_scale.str.contains("prevalence"), 100.0, 1.0)
    df = df.assign(cap=df.capability * scale, ug=df.undesired_gen_lower_is_better * scale)
    sd = df.groupby(["task", "model", "condition"])[["cap", "ug"]].std(ddof=0).reset_index()
    fam = {t: f for t, _, f in TASKS}
    return sd.assign(family=sd.task.map(fam))


def style():
    apply_style()


def heatmap(sd, out, label_min):
    rows = [(t, tn, m, mn) for t, tn, _ in TASKS for m, mn in MODELS]
    vmax = float(sd[["cap", "ug"]].max().max())
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, HEATMAP_HEIGHT), sharey=True)
    for ax, (col, title) in zip(axes, AXES):
        g = np.array([[sd[(sd.task == t) & (sd.model == m) & (sd.condition == c)][col].iloc[0]
                       for c, _ in CONDS] for t, _, m, _ in rows])
        im = ax.imshow(g, cmap=SEQ, vmin=0, vmax=vmax, aspect="auto")
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                if g[i, j] >= label_min:
                    ax.text(j, i, f"{g[i, j]:.1f}", ha="center", va="center", fontsize=FS_NOTE,
                            color="white" if g[i, j] > vmax * 0.45 else TEXT)
        for k in range(3, len(rows), 3):
            ax.axhline(k - 0.5, color="white", linewidth=3)
        ax.set_xticks(range(len(CONDS)), HEATMAP_COLUMNS, fontsize=FS_TICK)
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_yticks(range(len(rows)), [mn for _, _, _, mn in rows], fontsize=FS_TICK)
    left, right = 2.5, 0.15  # inches; the left margin holds the task and model labels
    inch_margins(fig, left=left, right=right, top=1.15, bottom=1.2, wspace=0.05)
    # Color bar in its own row below the grid, never over it
    cax = fig.add_axes([left / FIG_WIDTH, 0.72 / HEATMAP_HEIGHT, 1 - (left + right) / FIG_WIDTH,
                        0.12 / HEATMAP_HEIGHT])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Standard deviation across 5 seeds (points, 0–100 scale)", fontsize=FS_LABEL)
    cb.ax.tick_params(labelsize=FS_TICK)
    cb.outline.set_visible(False)
    # Group labels after the final layout, GROUP_GAP from the labels they group on both axes.
    for ax, (_, title) in zip(axes, AXES):
        title_above_ticks(ax, title)
    for k, (_, tn, _) in enumerate(TASKS):
        row_group_label(axes[0], tn, 3 * k + 1)
    fig.text(0.5, 0.06 / HEATMAP_HEIGHT, f"Numbers shown where the standard deviation is {label_min:g} points or more.",
             ha="center", fontsize=FS_NOTE, color=MUTED)
    # Rounded corners on each task's block of three model rows (white gaps of 3 points between blocks).
    blocks = [(-0.5, 3 * k - 0.5, len(CONDS) - 0.5, 3 * k + 2.5) for k in range(len(TASKS))]
    for ax in axes:
        round_blocks(ax, ax.images, blocks, gap=3)
    round_frame(cax, edge=False)
    print(save(fig, out))


def dot_plot(sd, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    ys = np.arange(len(CONDS))[::-1]
    rng = np.random.default_rng(0)
    xmax = float(sd[["cap", "ug"]].max().max()) * 1.08
    for ax, (col, title) in zip(axes, AXES):
        for y, (c, _) in zip(ys, CONDS):
            d = sd[sd.condition == c]
            for f in FAMILY_COLORS:
                v = d[d.family == f][col]
                ax.scatter(v, y + rng.uniform(-0.22, 0.22, len(v)), s=30, color=FAMILY_COLORS[f],
                           alpha=0.85, edgecolor=CANVAS, linewidth=0.8, zorder=3)
            med = d[col].median()
            ax.plot([med, med], [y - 0.32, y + 0.32], color=TEXT, linewidth=2, zorder=4)
        ax.set_xlim(-0.3, xmax)
        ax.set_title(title, fontsize=13, pad=10)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Standard deviation across 5 seeds (points)", fontsize=10.5)
    axes[0].set_yticks(ys, [n for _, n in CONDS], fontsize=10.5)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=7, markerfacecolor=FAMILY_COLORS[f],
                          markeredgecolor=CANVAS) for f in FAMILY_COLORS]
    handles.append(plt.Line2D([0], [0], color=TEXT, linewidth=2))
    labels = [FAMILY_NAMES[f] for f in FAMILY_COLORS] + ["Median over settings"]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.15, right=0.98, top=0.9, bottom=0.24, wspace=0.08)
    fig.savefig(out, dpi=180); plt.close(fig); print(out)


def bar_summary(sd, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    fams = list(FAMILY_COLORS)
    ys = np.arange(len(CONDS))[::-1]
    h = 0.24
    xmax = float(sd.groupby(["condition", "family"])[["cap", "ug"]].mean().max().max()) * 1.18
    for ax, (col, title) in zip(axes, AXES):
        means = sd.groupby(["condition", "family"])[col].mean()
        for k, f in enumerate(fams):
            vals = [means.loc[(c, f)] for c, _ in CONDS]
            ax.barh(ys + (1 - k) * h, vals, height=h * 0.9, color=FAMILY_COLORS[f], zorder=2)
            for y, v in zip(ys, vals):
                ax.text(v + xmax * 0.01, y + (1 - k) * h, f"{v:.1f}", va="center", fontsize=8, color=TEXT)
        ax.set_xlim(0, xmax)
        ax.set_title(title, fontsize=13, pad=10)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Mean standard deviation across 5 seeds (points)", fontsize=10.5)
    axes[0].set_yticks(ys, [n for _, n in CONDS], fontsize=10.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f]) for f in fams]
    fig.legend(handles, [FAMILY_NAMES[f] for f in fams], loc="lower center", ncol=3, frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.15, right=0.98, top=0.9, bottom=0.24, wspace=0.08)
    fig.savefig(out, dpi=180); plt.close(fig); print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-csv", type=Path, default=CSV)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--label-min", type=float, default=3.0, help="heatmap: label cells with SD >= this")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    style()
    sd = seed_sd(args.data_csv)
    heatmap(sd, args.out_dir / "A_heatmap_single_hue.png", args.label_min)
    dot_plot(sd, args.out_dir / "B_dot_plot.png")
    bar_summary(sd, args.out_dir / "C_bar_summary.png")
