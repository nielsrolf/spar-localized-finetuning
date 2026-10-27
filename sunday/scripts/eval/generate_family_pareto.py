"""Pareto view per task family: one point per technique, averaged over tasks, models and seeds.

Each technique's point is the mean over settings (task x model) of its 5-seed mean, in
points (fact/persona rates scaled to 0-100). The Pareto front is drawn over the SFT
baseline and the five techniques; the base model is shown for reference only.
With --show-settings, faint points show each setting's 5-seed mean.
Colors and markers match the per-setting Pareto charts.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from generate_pareto_charts import COLORS, DISPLAY_NAMES, MPL_MARKERS, pareto_front

HERE = Path(__file__).parent
CSV = HERE / "analysis_handoff" / "results_per_seed.csv"
VANILLA_CSV = HERE / "analysis_handoff" / "results_vanilla.csv"
OUT_DIR = HERE / "generated_charts_summary" / "deepseek"

FAMILIES = [
    ("Emergent misalignment", ["bad_medical_advice", "risky_financial_advice", "school_of_reward_hacks"],
     "Misalignment (100 − alignment)"),
    ("Weird generalization", ["german_city_names", "old_bird_names"], "Persona rate"),
    ("Good vs. bad facts", ["good_vs_bad_mixed_multifact"], "Unwanted learning:\nadoption of co-occurring bad facts"),
    ("Target-only facts", ["target_only"], "Errors on untrained questions"),
]
CONDS = ["baseline", "first-third", "second-third", "last-third", "kld", "inoculation"]
EM_TASKS = set(FAMILIES[0][1])

from chart_style import (  # noqa: E402
    CANVAS, FIG_WIDTH, FS_LABEL, FS_NOTE, FS_PANEL, FS_XLABEL, LEGEND_MARKER, LINE, MARKER_MEAN, MUTED,
    PANEL_PAD, PARETO, apply_style, better_arrow, legend, pareto_subplots, round_frame,
    save, style_axes, x_title_and_legend)


def to_points(df):
    scale = df.task.map(lambda t: 1.0 if t in EM_TASKS else 100.0)
    return df.assign(cap=df.capability * scale, ug=df.undesired_gen_lower_is_better * scale)


def load(csv_path, vanilla_csv):
    settings = to_points(pd.read_csv(csv_path)).groupby(["task", "model", "condition"])[["cap", "ug"]].mean()
    settings = settings.reset_index()
    if vanilla_csv is not None:
        van = to_points(pd.read_csv(vanilla_csv)).groupby(["task", "model"])[["cap", "ug"]].mean().reset_index()
        settings = pd.concat([settings, van.assign(condition="vanilla")])
    return settings


def draw(settings, out_path, title, show_settings, families=FAMILIES, highlight=None, direction_labels=True,
         ylim=None, note=True, panel_highlight=None, fade="alpha", paper_layout=False):
    """fade: how non-highlighted techniques are drawn: "alpha" (transparent), "hollow" (outline only),
    or "hollow-light" (thin, semi-transparent outline)."""
    """ylim: (low, high) shared by every panel; default fits each panel to its own data."""
    """highlight: conditions drawn at full opacity; the others keep their color but are faded."""
    """paper_layout: two panels with the shared Pareto panel size (pareto_subplots); the x title and the
    legend are placed GROUP_GAP apart. Default: the 7-inch layout."""
    apply_style()
    two_up = len(families) <= 2
    if two_up and paper_layout:
        fig, axes = pareto_subplots(len(families), top=0.62, bottom=1.65)
    else:
        fig, axes = (plt.subplots(1, len(families), figsize=(FIG_WIDTH, 7.0)) if two_up
                     else plt.subplots(2, 2, figsize=(FIG_WIDTH, 13)))
    axes = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]
    panel_highlight = panel_highlight or {}

    def opacity(hl):
        return {c: 1.0 if hl is None or c in hl else 0.3 for c in ["vanilla"] + CONDS}

    # legend: full opacity for anything highlighted in at least one panel
    shown = [panel_highlight.get(f[0], highlight) for f in families]
    alpha = opacity(None if any(h is None for h in shown) else set().union(*shown))
    fig.patch.set_facecolor(CANVAS)
    base_cover = []  # families where the base model was judged on only some settings

    for ax, (family, tasks, ylab) in zip(axes, families):
        panel_alpha = opacity(panel_highlight.get(family, highlight))
        fam = settings[settings.task.isin(tasks)]
        n_settings = fam[fam.condition == "baseline"].groupby(["task", "model"]).ngroups
        means = fam.groupby("condition")[["cap", "ug"]].mean()
        ax.set_facecolor(CANVAS)

        if show_settings:
            for cond in ["vanilla"] + CONDS:
                pts = fam[fam.condition == cond]
                ax.scatter(pts.cap, pts.ug, s=26, marker=MPL_MARKERS[cond], color=COLORS[cond],
                           alpha=0.28, edgecolor="none", zorder=2)

        front = pareto_front({c: tuple(means.loc[c]) for c in CONDS if c in means.index})
        ax.plot([p[0] for p in front], [p[1] for p in front], color=PARETO, linewidth=LINE, zorder=3)

        pts = fam if show_settings else means.reset_index()
        y_hi = max(pts.ug.max() * 1.15, 10)
        ax.set_xlim(-3, 105)
        ax.set_ylim(*ylim) if ylim else ax.set_ylim(-y_hi * 0.04, y_hi)

        for cond in ["vanilla"] + CONDS:
            if cond not in means.index:
                continue
            x, y = means.loc[cond]
            hollow = fade.startswith("hollow")
            if hollow and panel_alpha[cond] < 1.0:
                light = fade == "hollow-light"
                lw = 1.2 if light else 1.8
                # match the visible size of a filled marker (s=MARKER_MEAN, white 1.4pt edge): the filled
                # marker shows d - 1.4 points, a hollow one shows d + lw, so shrink d accordingly
                d = MARKER_MEAN ** 0.5 - 1.4 - lw
                ax.scatter([x], [y], s=d * d, marker=MPL_MARKERS[cond], facecolor=CANVAS,
                           edgecolor=COLORS[cond], linewidth=lw,
                           alpha=0.55 if light else 1.0, zorder=4)
            else:
                ax.scatter([x], [y], s=MARKER_MEAN, marker=MPL_MARKERS[cond], color=COLORS[cond],
                           alpha=1.0 if hollow else panel_alpha[cond],
                           edgecolor=CANVAS, linewidth=1.4, zorder=5 if panel_alpha[cond] == 1.0 else 4)
        n_base = fam[fam.condition == "vanilla"].groupby(["task", "model"]).ngroups
        if n_base < n_settings:
            base_cover.append(f"{family.lower()}: {n_base} of {n_settings} settings")
        style_axes(ax)
        ax.set_axisbelow(True)
        if not two_up:
            ax.set_xlabel("Target capability", fontsize=FS_LABEL)
        ax.set_ylabel(f"{ylab}\n(lower is better)" if direction_labels else ylab, fontsize=FS_LABEL)
        ax.set_title(family, fontsize=FS_PANEL, pad=PANEL_PAD)
        better_arrow(ax)
    if two_up and not paper_layout:
        fig.supxlabel("Target capability", y=0.2 if note else 0.15, fontsize=FS_XLABEL)

    def legend_size(c):  # same visible size for filled and hollow legend markers
        if fade.startswith("hollow") and alpha[c] < 1.0:
            return LEGEND_MARKER - 1.2 - 1.3
        return LEGEND_MARKER

    handles = [plt.Line2D([0], [0], marker=MPL_MARKERS[c], linestyle="", markersize=legend_size(c),
                          markerfacecolor=COLORS[c] if not fade.startswith("hollow") or alpha[c] == 1.0 else CANVAS,
                          markeredgecolor=CANVAS if not fade.startswith("hollow") or alpha[c] == 1.0 else COLORS[c],
                          markeredgewidth=1.2 if not fade.startswith("hollow") or alpha[c] == 1.0 else 1.3,
                          alpha=(0.55 if fade == "hollow-light" and alpha[c] < 1.0 else 1.0)
                          if fade.startswith("hollow") else alpha[c])
               for c in ["vanilla"] + CONDS]
    labels = [DISPLAY_NAMES[c] for c in ["vanilla"] + CONDS]
    handles.append(plt.Line2D([0], [0], color=PARETO, linewidth=LINE))
    labels.append("Pareto front")
    if not paper_layout:
        legend(fig, handles, labels, y=(0.075 if note else 0.035) if two_up else 0.045)
    note_text = ("Large markers: mean over settings (task × model), each averaged over 5 seeds."
            + ("  Faint markers: individual settings." if show_settings else "")
            + "  Line: Pareto front over SFT and the five techniques.")
    if base_cover:
        note_text += "\nBase model judged only where available (" + "; ".join(base_cover) + ")."
    if note:
        fig.text(0.5, 0.012, note_text, ha="center", fontsize=FS_NOTE, color=MUTED)
    if two_up and paper_layout:
        # Below the panels: tick labels, x title, and a two-row legend
        x_title_and_legend(fig, axes, "Target capability", handles, labels, ncol=4)
    elif two_up:
        fig.subplots_adjust(top=0.88, bottom=0.34 if note else 0.29, left=0.08, right=0.97, wspace=0.2)
    else:
        fig.subplots_adjust(top=0.94, bottom=0.17, left=0.08, right=0.97, wspace=0.2, hspace=0.38)
    for ax in axes:
        round_frame(ax)
    save(fig, out_path)
    print(out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-csv", type=Path, default=CSV)
    ap.add_argument("--vanilla-csv", default=str(VANILLA_CSV), help="base-model CSV, or 'none'")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--judge", default="DeepSeek-v4-flash judge")
    ap.add_argument("--families", nargs="*", help="only these families, e.g. 'Emergent misalignment'")
    ap.add_argument("--highlight", nargs="*", help="conditions at full opacity, e.g. baseline kld last-third")
    ap.add_argument("--out-name", help="draw one figure (means only) with this file name")
    ap.add_argument("--no-direction-labels", action="store_true", help="omit '(lower is better)' from y labels")
    ap.add_argument("--panel-highlight", action="append", default=[], metavar="FAMILY=COND,COND",
                    help="highlight for one panel, overriding --highlight there (repeatable)")
    ap.add_argument("--fade", choices=["alpha", "hollow", "hollow-light"], default="alpha",
                    help="draw non-highlighted techniques transparent (alpha) or as outlines (hollow)")
    ap.add_argument("--no-note", action="store_true", help="omit the explanatory note under the chart")
    ap.add_argument("--paper-layout", action="store_true",
                    help="two panels with the shared Pareto panel size (as in the paper)")
    ap.add_argument("--ylim", nargs=2, type=float, metavar=("LOW", "HIGH"), help="same y range for every panel")
    args = ap.parse_args()
    settings = load(args.data_csv, None if args.vanilla_csv == "none" else Path(args.vanilla_csv))
    title = f"Trade-off per task family, averaged over tasks, models and seeds ({args.judge})"
    if args.out_name:
        fams = [f for f in FAMILIES if not args.families or f[0] in args.families]
        draw(settings, args.out_dir / args.out_name, title, show_settings=False,
             families=fams, highlight=set(args.highlight) if args.highlight else None,
             direction_labels=not args.no_direction_labels, ylim=args.ylim, note=not args.no_note,
             fade=args.fade, paper_layout=args.paper_layout,
             panel_highlight={k: set(v.split(",")) for k, v in (a.split("=", 1) for a in args.panel_highlight)})
        raise SystemExit
    draw(settings, args.out_dir / "pareto_by_family.png", title, show_settings=False)
    draw(settings, args.out_dir / "pareto_by_family_with_settings.png", title, show_settings=True)
