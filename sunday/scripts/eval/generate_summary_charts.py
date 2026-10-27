"""High-level summary: each technique's mean score, compared with the SFT baseline.

One figure, rows = task family, columns = axis (capability, undesired generalization).
Each bar is the mean over settings (task x model) of the technique's 5-seed mean, in
points (fact/persona rates are scaled to 0-100). Gray dots are the individual settings.
The dashed line marks the SFT baseline; technique bars are blue where they beat it and
red where they don't. The base model and SFT baseline are gray reference rows.

--relative instead plots each technique's change vs. the SFT baseline (baseline = 0).
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from generate_pareto_charts import COLORS as TECHNIQUE_COLORS  # same colors as the Pareto charts

HERE = Path(__file__).parent
CSV = HERE / "analysis_handoff" / "results_per_seed.csv"
VANILLA_CSV = HERE / "analysis_handoff" / "results_vanilla.csv"
OUT_DIR = HERE / "generated_charts_summary" / "deepseek"

FAMILIES = [
    ("Emergent misalignment", ["bad_medical_advice", "risky_financial_advice", "school_of_reward_hacks"]),
    ("Weird generalization", ["german_city_names", "old_bird_names"]),
    ("Good vs. bad facts", ["good_vs_bad_mixed_multifact"]),
    ("Target-only facts", ["target_only"]),
]
# rows whose second axis is not undesired generalization
SECOND_AXIS_NOTE = {"Good vs. bad facts": "right column = unwanted learning"}
TECHNIQUES = [
    ("first-third", "First third"), ("second-third", "Middle third"), ("last-third", "Last third"),
    ("kld", "KL regularization"), ("inoculation", "Inoculation prompting"),
]
REFERENCES = [("vanilla", "Base model (no fine-tuning)"), ("baseline", "SFT baseline")]
ROWS = REFERENCES + TECHNIQUES
EM_TASKS = set(FAMILIES[0][1])

CANVAS = "#ffffff"
TEXT = "#2f3340"
MUTED = "#6b6f7a"
GRID = "#e6e1d6"
BETTER = "#2a78d6"   # diverging pair, blue pole
WORSE = "#e34948"    # diverging pair, red pole
DOT = "#9aa0aa"
REF = "#b5b9c1"      # reference rows (base model, SFT baseline): neutral gray
MPL_FONT = ["STIX Two Text", "Charter", "Georgia", "DejaVu Serif"]


def to_points(df):
    scale = df.task.map(lambda t: 1.0 if t in EM_TASKS else 100.0)
    return df.assign(cap=df.capability * scale, ug=df.undesired_gen_lower_is_better * scale)


def setting_values(csv_path, vanilla_csv=None, relative=False):
    """Per (task, model, row): 5-seed mean on both axes, 0-100 scale; minus SFT if relative."""
    means = to_points(pd.read_csv(csv_path)).groupby(["task", "model", "condition"])[["cap", "ug"]].mean()
    base = means.xs("baseline", level="condition")
    offset = base if relative else 0
    rows = []
    for cond, _ in [("baseline", None)] + TECHNIQUES:
        d = means.xs(cond, level="condition") - offset
        rows.append(d.assign(condition=cond).reset_index())
    if vanilla_csv is not None:
        van = to_points(pd.read_csv(vanilla_csv)).groupby(["task", "model"])[["cap", "ug"]].mean()
        d = (van - offset).dropna() if relative else van.loc[van.index.isin(base.index)]
        rows.append(d.assign(condition="vanilla").reset_index())
    return pd.concat(rows)


def draw(values, out_path, title, relative):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": MPL_FONT, "text.color": TEXT,
        "axes.labelcolor": TEXT, "axes.edgecolor": MUTED, "xtick.color": TEXT, "ytick.color": TEXT,
    })
    fig, axes = plt.subplots(len(FAMILIES), 2, figsize=(11.5, 14.4), sharey=True)
    fig.patch.set_facecolor(CANVAS)
    ys = list(range(len(ROWS)))[::-1]

    for row, (family, tasks) in enumerate(FAMILIES):
        fam = values[values.task.isin(tasks)]
        n_settings = fam.groupby(["task", "model"]).ngroups
        span = max(fam.cap.abs().max(), fam.ug.abs().max()) * 1.18 if relative else 100.0
        for col, (axis, lower_better) in enumerate([("cap", False), ("ug", True)]):
            ax = axes[row, col]
            ax.set_facecolor(CANVAS)
            sft = 0.0 if relative else fam[fam.condition == "baseline"][axis].mean()
            for y, (cond, _) in zip(ys, ROWS):
                vals = fam[fam.condition == cond][axis]
                if relative and cond == "baseline":
                    ax.plot([0], [y], marker="D", markersize=7, color=REF, zorder=3)
                    ax.text(span * 0.04, y, "reference (0)", va="center", ha="left", fontsize=9.5, color=MUTED)
                    continue
                if vals.empty:
                    ax.text(span * 0.04 if relative else 2, y, "not re-judged", va="center", ha="left",
                            fontsize=9.5, color=MUTED, style="italic")
                    continue
                mean = vals.mean()
                better = mean < sft if lower_better else mean > sft
                if relative:
                    color = REF if cond in dict(REFERENCES) else (BETTER if better else WORSE)
                else:
                    color = TECHNIQUE_COLORS[cond]
                ax.barh(y, mean, height=0.56, color=color, alpha=0.9, zorder=2)
                jitter = [(-0.13 + 0.26 * i / max(len(vals) - 1, 1)) for i in range(len(vals))]
                ax.scatter(vals, [y + j for j in jitter], s=14, color=DOT, edgecolor=CANVAS,
                           linewidth=0.6, zorder=3)
                if not relative:  # fixed value column at the right edge
                    ha, x_lab = "left", 104
                elif mean >= 0:
                    ha, x_lab = "left", max(vals.max(), mean, 0) + span * 0.03
                else:
                    ha, x_lab = "right", min(vals.min(), 0) - span * 0.03
                partial = f"  ({len(vals)} of {n_settings})" if cond == "vanilla" and len(vals) < n_settings else ""
                label = f"{mean:+.1f}" if relative else f"{mean:.1f}"
                ax.text(x_lab, y, f"{label}{partial}", va="center", ha=ha, fontsize=9.5, color=TEXT, zorder=4)
            ax.axhline(ys[len(REFERENCES)] + 0.5, color=GRID, linewidth=1.2, zorder=0)
            ax.axvline(0, color=TEXT, linewidth=1.0, zorder=1)
            if relative:
                ax.set_xlim(-span * 1.12, span * 1.12)
            else:
                ax.axvline(sft, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
                ax.text(sft + 1.2, ys[len(REFERENCES)] + 0.5, "SFT", ha="left", va="center", fontsize=9,
                        color=MUTED, zorder=5, bbox=dict(facecolor=CANVAS, edgecolor="none", pad=1.0))
                ax.set_xlim(0, 118)
                ax.set_xticks([0, 25, 50, 75, 100])
            ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
            ax.tick_params(axis="y", length=0, labelsize=10.5)
            ax.tick_params(axis="x", labelsize=9, colors=MUTED)
            for s in ["top", "right", "left"]:
                ax.spines[s].set_visible(False)
            if row == 0:
                head = ("Target capability\n(higher is better)" if axis == "cap"
                        else "Undesired generalization\n(lower is better)")
                ax.set_title(head, fontsize=12.5, pad=34)
            if row == len(FAMILIES) - 1:
                xlab = "Change vs. SFT baseline (points)" if relative else "Mean score (0-100)"
                ax.set_xlabel(xlab, fontsize=10.5, color=MUTED)
        axes[row, 0].set_yticks(ys, [name for _, name in ROWS])
        for lab, (cond, _) in zip(axes[row, 0].get_yticklabels(), ROWS):
            if cond in dict(REFERENCES):
                lab.set_color(MUTED)
        note = f";  {SECOND_AXIS_NOTE[family]}" if family in SECOND_AXIS_NOTE else ""
        axes[row, 0].annotate(f"{family}  ({n_settings} settings{note})", xy=(0, 1.02), xycoords="axes fraction",
                              fontsize=12.5, ha="left", va="bottom")

    fig.suptitle(title, fontsize=14, y=0.99)
    bars = ("Bars: mean change over settings (task × model), each averaged over 5 seeds."
            if relative else
            "Bars: mean over settings (task × model), each averaged over 5 seeds.  Dashed line: SFT baseline.")
    legend = ("Blue = better than SFT, red = worse.  Gray = reference rows (base model: one untuned run per setting)."
              if relative else
              "Colors match the per-setting Pareto charts.  Base model: one untuned run per setting.")
    fig.text(0.5, 0.008,
             f"{bars}  Dots: individual settings.\n{legend}\n"
             "Unwanted learning: adopting bad facts that appear in the training data (good vs. bad facts task).",
             ha="center", fontsize=9, color=MUTED)
    fig.subplots_adjust(top=0.895, bottom=0.1, left=0.17, right=0.98, hspace=0.45, wspace=0.08)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-csv", type=Path, default=CSV)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--vanilla-csv", default=str(VANILLA_CSV), help="base-model CSV, or 'none'")
    ap.add_argument("--judge", default="DeepSeek-v4-flash judge")
    ap.add_argument("--relative", action="store_true", help="plot change vs. the SFT baseline")
    args = ap.parse_args()
    vanilla = None if args.vanilla_csv == "none" else Path(args.vanilla_csv)
    values = setting_values(args.data_csv, vanilla, relative=args.relative)
    if args.relative:
        draw(values, args.out_dir / "technique_effects.png", f"Effect of each technique relative to the SFT baseline ({args.judge})", True)
    else:
        draw(values, args.out_dir / "technique_means.png", f"Mean scores by technique ({args.judge})", False)
