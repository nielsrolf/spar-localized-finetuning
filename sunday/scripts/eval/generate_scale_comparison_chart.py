"""Qwen3-8B vs. Qwen3-32B: each technique's change vs. its own SFT baseline.

Rows = task (bad medical advice, German city names, target-only facts); columns = axis
(capability, undesired generalization). For each technique, the light bar is Qwen3-8B
(mean over 5 seeds, dots = seeds) and the solid bar is Qwen3-32B (seed 1). Both sizes
use the DeepSeek-v4-flash judge, capability unfiltered and undesired generalization
coherence-filtered, in points (rates scaled to 0-100). Each size is compared with its
own SFT baseline; the absolute baseline values are printed under each task title.
Bars use the technique colors of the Pareto charts.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from generate_pareto_charts import COLORS, DISPLAY_NAMES

HERE = Path(__file__).parent
CSV_8B = HERE / "analysis_handoff" / "results_per_seed.csv"
CSV_32B = HERE / "analysis_handoff" / "results_qwen32_seed1.csv"
OUT = HERE / "generated_charts_summary" / "deepseek" / "qwen_8b_vs_32b.png"

TASKS = [
    ("bad_medical_advice", "Bad medical advice", "misalignment"),
    ("german_city_names", "German city names", "persona rate"),
    ("target_only", "Target-only facts", "untargeted hallucination"),
]
CONDS = ["first-third", "second-third", "last-third", "kld", "inoculation"]
EM_TASKS = {"bad_medical_advice"}
LIGHT = 0.35

CANVAS = "#ffffff"
TEXT = "#2f3340"
MUTED = "#6b6f7a"
GRID = "#e6e1d6"
DOT = "#6b6f7a"
MPL_FONT = ["STIX Two Text", "Charter", "Georgia", "DejaVu Serif"]


def load_8b(path):
    """Qwen3-8B per seed, in points, as change vs. the 5-seed SFT baseline mean."""
    df = pd.read_csv(path)
    df = df[(df.model == "Qwen3-8B") & df.task.isin([t for t, _, _ in TASKS])]
    scale = df.task.map(lambda t: 1.0 if t in EM_TASKS else 100.0)
    df = df.assign(cap=df.capability * scale, ug=df.undesired_gen_lower_is_better * scale)
    base = df[df.condition == "baseline"].groupby("task")[["cap", "ug"]].mean()
    rel = df.set_index("task")[["cap", "ug"]] - base
    return df.assign(dcap=rel.cap.values, dug=rel.ug.values), base


def load_32b(path):
    """Qwen3-32B seed 1, in points, as change vs. its SFT baseline."""
    df = pd.read_csv(path).rename(columns={"capability": "cap", "undesired_gen": "ug"})
    base = df[df.condition == "baseline"].set_index("task")[["cap", "ug"]]
    rel = df.set_index("task")[["cap", "ug"]] - base
    return df.assign(dcap=rel.cap.values, dug=rel.ug.values), base


def draw(d8, base8, d32, base32, out_path, judge="DeepSeek-v4-flash judge", show_title=True, show_note=True):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": MPL_FONT, "text.color": TEXT,
        "axes.labelcolor": TEXT, "axes.edgecolor": MUTED, "xtick.color": TEXT, "ytick.color": TEXT,
    })
    fig, axes = plt.subplots(len(TASKS), 2, figsize=(11.5, 12.0), sharey=True)
    fig.patch.set_facecolor(CANVAS)
    ys = list(range(len(CONDS)))[::-1]
    h = 0.34

    for row, (task, title, ug_name) in enumerate(TASKS):
        t8, t32 = d8[d8.task == task], d32[d32.task == task]
        vals = pd.concat([t8.dcap, t8.dug, t32.dcap, t32.dug]).abs()
        span = max(vals.max(), 5) * 1.15
        for col, axis in enumerate(["dcap", "dug"]):
            ax = axes[row, col]
            ax.set_facecolor(CANVAS)
            for y, cond in zip(ys, CONDS):
                s8 = t8[t8.condition == cond][axis]
                s32 = t32[t32.condition == cond][axis]
                m8 = s8.mean()
                ax.barh(y + h / 2, m8, height=h, color=COLORS[cond], alpha=LIGHT, zorder=2)
                ax.scatter(s8, [y + h / 2] * len(s8), s=10, color=DOT, alpha=0.6, edgecolor="none", zorder=3)
                lab_x = max(s8.max(), m8, 0) + span * 0.03 if m8 >= 0 else min(s8.min(), 0) - span * 0.03
                ax.text(lab_x, y + h / 2, f"{m8:+.1f}", va="center", ha="left" if m8 >= 0 else "right",
                        fontsize=8.5, color=MUTED)
                if s32.empty:
                    ax.text(span * 0.04, y - h / 2, "32B not trained", va="center", ha="left", fontsize=8.5,
                            color=MUTED, style="italic")
                    continue
                m32 = s32.iloc[0]
                ax.barh(y - h / 2, m32, height=h, color=COLORS[cond], alpha=1.0, zorder=2)
                lab_x = max(m32, 0) + span * 0.03 if m32 >= 0 else min(m32, 0) - span * 0.03
                ax.text(lab_x, y - h / 2, f"{m32:+.1f}", va="center", ha="left" if m32 >= 0 else "right",
                        fontsize=9, color=TEXT)
            ax.axvline(0, color=TEXT, linewidth=1.0, zorder=1)
            ax.set_xlim(-span * 1.15, span * 1.15)
            ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
            ax.tick_params(axis="y", length=0, labelsize=10.5)
            ax.tick_params(axis="x", labelsize=9, colors=MUTED)
            for s in ["top", "right", "left"]:
                ax.spines[s].set_visible(False)
            if row == 0:
                head = ("Target capability\n(higher is better)" if axis == "dcap"
                        else "Undesired generalization\n(lower is better)")
                ax.set_title(head, fontsize=12.5, pad=46)
            if row == len(TASKS) - 1:
                ax.set_xlabel("Change vs. own SFT baseline (points)", fontsize=10.5, color=MUTED)
        axes[row, 0].set_yticks(ys, [DISPLAY_NAMES[c] for c in CONDS])
        b8, b32 = base8.loc[task], base32.loc[task]
        axes[row, 0].annotate(title, xy=(0, 1.13), xycoords="axes fraction", fontsize=12.5, ha="left", va="bottom")
        axes[row, 0].annotate(
            f"SFT baseline  8B: capability {b8.cap:.1f}, {ug_name} {b8.ug:.1f}   |   "
            f"32B: capability {b32.cap:.1f}, {ug_name} {b32.ug:.1f}",
            xy=(0, 1.03), xycoords="axes fraction", fontsize=9, color=MUTED, ha="left", va="bottom")

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS["last-third"], alpha=LIGHT),
               plt.Rectangle((0, 0), 1, 1, color=COLORS["last-third"])]
    fig.legend(handles, ["Qwen3-8B (mean of 5 seeds; dots = seeds)", "Qwen3-32B (seed 1)"],
               loc="lower center", bbox_to_anchor=(0.5, 0.035 if show_note else 0.005), ncol=2, frameon=False, fontsize=10.5)
    if show_title:
        fig.suptitle(f"Qwen3-8B vs. Qwen3-32B: change vs. each model's SFT baseline ({judge})",
                     fontsize=13.5, y=0.995)
    if show_note:
        fig.text(0.5, 0.008,
                 "Bar color = technique.  32B KL and inoculation prompting from the selective-learning-benchmark repo "
                 "(vLLM generation); 32B is a single seed.\n32B runs not loss-matched: bad medical first and last third, "
                 "German last third, target-only first third.",
                 ha="center", fontsize=8.8, color=MUTED)
    fig.subplots_adjust(top=0.865 if show_title else 0.895, bottom=0.135 if show_note else 0.115,
                        left=0.15, right=0.98, hspace=0.62, wspace=0.08)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-8b", type=Path, default=CSV_8B)
    ap.add_argument("--csv-32b", type=Path, default=CSV_32B)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--judge", default="DeepSeek-v4-flash judge", help="judge name for the title")
    ap.add_argument("--no-title", action="store_true", help="omit the figure title (paper version)")
    ap.add_argument("--no-note", action="store_true", help="omit the footer note (paper version)")
    args = ap.parse_args()
    d8, base8 = load_8b(args.csv_8b)
    d32, base32 = load_32b(args.csv_32b)
    draw(d8, base8, d32, base32, args.out, args.judge, show_title=not args.no_title, show_note=not args.no_note)
