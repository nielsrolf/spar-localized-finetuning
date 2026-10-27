"""Layer position vs. the SFT baseline: first, middle and last third, per task family.

Rows = task family (emergent misalignment, weird generalization); columns = axis
(capability, undesired generalization). Each bar is the mean change vs. the SFT
baseline over settings (task x model), each averaged over 5 seeds, in points
(persona rates scaled to 0-100). Dots are the individual settings; the marker shape
shows the model. Bars use the technique colors of the Pareto charts. Conditions that
are not the focus of a family keep their color but are faded, as in
generate_family_pareto.py --highlight. The gray band (emergent misalignment only)
marks +-5 points.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from generate_pareto_charts import COLORS, DISPLAY_NAMES

HERE = Path(__file__).parent
CSV = HERE / "analysis_handoff" / "luna" / "results_per_seed.csv"
OUT_DIR = HERE / "generated_charts_summary" / "luna"

# (family, tasks, conditions at full opacity, draw the +-BAND band)
FAMILIES = [
    ("Emergent misalignment", ["bad_medical_advice", "risky_financial_advice", "school_of_reward_hacks"],
     {"last-third"}, True),
    ("Weird generalization", ["german_city_names", "old_bird_names"], {"first-third"}, False),
]
CONDS = ["first-third", "second-third", "last-third"]
EM_TASKS = set(FAMILIES[0][1])
MODEL_MARKERS = {"Llama 3.1 8B": "o", "Qwen3-8B": "s", "OLMo 3 7B": "^"}
MODEL_LABELS = {"Llama 3.1 8B": "Llama 3.1 8B", "Qwen3-8B": "Qwen3 8B", "OLMo 3 7B": "OLMo 3 7B"}
BAND = 5.0
FADED = 0.3

CANVAS = "#ffffff"
TEXT = "#2f3340"
MUTED = "#6b6f7a"
GRID = "#e6e1d6"
DOT = "#6b6f7a"
BAND_COLOR = "#ece9e2"
MPL_FONT = ["STIX Two Text", "Charter", "Georgia", "DejaVu Serif"]


def setting_deltas(csv_path):
    """Per (task, model, condition): 5-seed mean minus the SFT baseline's 5-seed mean, in points."""
    df = pd.read_csv(csv_path)
    scale = df.task.map(lambda t: 1.0 if t in EM_TASKS else 100.0)
    df = df.assign(cap=df.capability * scale, ug=df.undesired_gen_lower_is_better * scale)
    means = df.groupby(["task", "model", "condition"])[["cap", "ug"]].mean()
    base = means.xs("baseline", level="condition")
    rows = [(means.xs(c, level="condition") - base).assign(condition=c).reset_index() for c in CONDS]
    return pd.concat(rows)


def draw(deltas, out_path, title):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": MPL_FONT, "text.color": TEXT,
        "axes.labelcolor": TEXT, "axes.edgecolor": MUTED, "xtick.color": TEXT, "ytick.color": TEXT,
    })
    fig, axes = plt.subplots(len(FAMILIES), 2, figsize=(11.5, 7.6), sharey=True)
    fig.patch.set_facecolor(CANVAS)
    ys = list(range(len(CONDS)))[::-1]

    for row, (family, tasks, focus, band) in enumerate(FAMILIES):
        fam = deltas[deltas.task.isin(tasks)]
        n_settings = fam.groupby(["task", "model"]).ngroups
        span = max(fam.cap.abs().max(), fam.ug.abs().max()) * 1.15
        for col, (axis, lower_better) in enumerate([("cap", False), ("ug", True)]):
            ax = axes[row, col]
            ax.set_facecolor(CANVAS)
            if band:
                ax.axvspan(-BAND, BAND, color=BAND_COLOR, zorder=0, linewidth=0)
            for y, cond in zip(ys, CONDS):
                alpha = 1.0 if cond in focus else FADED
                vals = fam[fam.condition == cond]
                mean = vals[axis].mean()
                ax.barh(y, mean, height=0.56, color=COLORS[cond], alpha=alpha, zorder=2)
                n = len(vals)
                for i, (_, r) in enumerate(vals.sort_values(["task", "model"]).iterrows()):
                    jitter = -0.16 + 0.32 * i / max(n - 1, 1)
                    ax.scatter([r[axis]], [y + jitter], s=22, marker=MODEL_MARKERS[r.model], color=DOT,
                               edgecolor=CANVAS, linewidth=0.6, alpha=alpha, zorder=3)
                if mean >= 0:
                    ha, x_lab = "left", max(vals[axis].max(), mean, 0) + span * 0.03
                else:
                    ha, x_lab = "right", min(vals[axis].min(), 0) - span * 0.03
                ax.text(x_lab, y, f"{mean:+.1f}", va="center", ha=ha, fontsize=10, zorder=4,
                        color=TEXT, alpha=1.0 if cond in focus else 0.55)
            ax.axvline(0, color=TEXT, linewidth=1.0, zorder=1)
            ax.set_xlim(-span * 1.12, span * 1.12)
            ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
            ax.tick_params(axis="y", length=0, labelsize=11)
            ax.tick_params(axis="x", labelsize=9, colors=MUTED)
            for s in ["top", "right", "left"]:
                ax.spines[s].set_visible(False)
            if row == 0:
                head = ("Target capability\n(higher is better)" if axis == "cap"
                        else "Undesired generalization\n(lower is better)")
                ax.set_title(head, fontsize=12.5, pad=34)
            if row == len(FAMILIES) - 1:
                ax.set_xlabel("Change vs. SFT baseline (points)", fontsize=10.5, color=MUTED)
        axes[row, 0].set_yticks(ys, [DISPLAY_NAMES[c] for c in CONDS])
        for lab, cond in zip(axes[row, 0].get_yticklabels(), CONDS):
            lab.set_alpha(1.0 if cond in focus else 0.55)
        axes[row, 0].annotate(f"{family}  ({n_settings} settings)", xy=(0, 1.04), xycoords="axes fraction",
                              fontsize=12.5, ha="left", va="bottom")

    handles = [plt.Line2D([0], [0], marker=m, linestyle="", markersize=7, markerfacecolor=DOT,
                          markeredgecolor=CANVAS) for m in MODEL_MARKERS.values()]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=BAND_COLOR))
    labels = list(MODEL_LABELS.values()) + [f"±{BAND:g} points (emergent misalignment)"]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=4,
               frameon=False, fontsize=10)
    fig.suptitle(title, fontsize=14, y=0.985)
    fig.text(0.5, 0.012,
             "Bars: mean change over settings (task × model), each averaged over 5 seeds.  "
             "Dots: individual settings (shape = model).\nFaded rows are not the focus for that task family.",
             ha="center", fontsize=9, color=MUTED)
    fig.subplots_adjust(top=0.83, bottom=0.2, left=0.13, right=0.98, hspace=0.5, wspace=0.08)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-csv", type=Path, default=CSV)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--judge", default="gpt-6-luna judge")
    args = ap.parse_args()
    draw(setting_deltas(args.data_csv), args.out_dir / "layer_position_effects.png",
         f"Effect of layer position relative to the SFT baseline ({args.judge})")
