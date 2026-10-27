"""Logit lens, layer by layer, for the 24 inserted facts. Draws two charts from the same data:

logit_lens_delta_by_layer.png   mean of logit(inserted) - logit(true) over the facts; above 0 the
                                inserted answer is ahead
logit_lens_share_by_layer.png   share of facts where logit(inserted) > logit(true)

One panel per model family (columns) and fact task (rows), chat prompts ("What is the capital of
Canada?" answered with "The capital of Canada is"). The shaded band is the block that the paper's
last-third runs trained. In early layers the two logits are nearly equal, so the difference is close
to 0 (and the share close to a coin flip) and carries no signal.

Also writes both per-layer values to lens_by_layer.csv.

Usage (from scripts/probe): python plot_logit_lens_share.py
"""

import csv
import json
import sys
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "eval"))
from chart_style import (COLORS, DISPLAY_NAMES, FIG_WIDTH, FS_LABEL, FS_PANEL, FS_TITLE, FS_XLABEL,  # noqa: E402
                         GRID, MODEL_LABEL, PANEL_PAD, TEXT, apply_style, inch_margins, legend, round_frame, save,
                         side_label, x_title_and_legend,
                         style_axes)

RESULTS = HERE / "results" / "fact_mechanism"
FAMILIES = {"llama31_8b": ("Llama-3.1-8B", 22, 31), "qwen3_8b": ("Qwen3-8B", 24, 35), "olmo3_7b": ("OLMo-3-7B", 19, 27)}
TASKS = {"target_only": "Target-only facts", "good_vs_bad": "Good vs. bad facts"}
CHECKPOINTS = [("base", "vanilla"), ("sft", "baseline"), ("first_third", "first-third"),
               ("second_third", "second-third"), ("last_third", "last-third")]
# Layers each method trained on the two fact tasks (from the paper runs' layers_to_transform).
TRAINED = {
    "llama31_8b": {"sft": (0, 31), "first_third": (0, 10), "second_third": (11, 21), "last_third": (22, 31)},
    "olmo3_7b": {"sft": (0, 31), "first_third": (0, 9), "second_third": (10, 18), "last_third": (19, 27)},
    "qwen3_8b": {"sft": (0, 35), "first_third": (0, 11), "second_third": (12, 23), "last_third": (24, 35)},
}


def facts(lens, fmt="chat"):
    return [x for x in lens if x["format"] == fmt and not x.get("skipped")]


def delta_by_layer(lens):
    xs = facts(lens)
    return [mean(x["delta"][layer] for x in xs) for layer in range(len(xs[0]["delta"]))]


def share_by_layer(lens):
    xs = facts(lens)
    return [100 * sum(x["delta"][layer] > 0 for x in xs) / len(xs) for layer in range(len(xs[0]["delta"]))]


# metric -> (per-layer function, y-axis label, reference line, y-limits or None, output file)
METRICS = {
    "delta": (delta_by_layer, "logit(inserted) − logit(true)\n(mean over 24 facts)", 0, None,
              "logit_lens_delta_by_layer.png"),
    "share": (share_by_layer, "Inserted answer beats true answer\n(% of 24 facts)", 50, (-3, 103),
              "logit_lens_share_by_layer.png"),
}


def draw(all_models, metric):
    fn, ylabel, ref, ylim, out_name = METRICS[metric]
    fig, axes = plt.subplots(len(TASKS), len(FAMILIES), figsize=(14, 7.6), sharey=True)
    for col, (fam, (label, b0, b1)) in enumerate(FAMILIES.items()):
        models = all_models[fam]
        for row, task in enumerate(TASKS):
            ax = axes[row, col]
            ax.axvspan(b0 - 0.5, b1 + 0.5, color="#000000", alpha=0.07, lw=0)
            ax.axhline(ref, color=GRID if metric == "share" else TEXT, lw=1, ls="--")
            for ckpt, cond in CHECKPOINTS:
                name = "base" if ckpt == "base" else f"{task}_{ckpt}"
                values = fn(models[name]["lens"])
                ax.plot(range(len(values)), values, color=COLORS[cond], lw=2.2 if ckpt == "last_third" else 1.5,
                        label=DISPLAY_NAMES[cond], zorder=3 if ckpt == "last_third" else 2)
            if ylim:
                ax.set_ylim(*ylim)
            ax.set_xlim(-0.5, len(values) - 0.5)
            ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row == 0:
                ax.set_title(f"{label}  (last third trained: layers {b0}–{b1})", fontsize=11)
            if row == len(TASKS) - 1:
                ax.set_xlabel("Layer")
            if col == 0:
                ax.set_ylabel(ylabel)
    if metric == "delta":
        ax0 = axes[0, 0]
        ax0.text(0.02, 0.97, "above 0: inserted answer ahead", transform=ax0.transAxes, va="top", fontsize=9, color=TEXT)
        ax0.text(0.02, 0.03, "below 0: true answer ahead", transform=ax0.transAxes, va="bottom", fontsize=9, color=TEXT)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.115, right=0.99, top=0.93, bottom=0.14, hspace=0.25, wspace=0.08)
    for row, task_label in enumerate(TASKS.values()):
        box = axes[row, 0].get_position()
        fig.text(0.012, (box.y0 + box.y1) / 2, task_label, rotation=90, va="center", ha="left", fontsize=12)
    out = RESULTS / out_name
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(out)


def draw_by_method(all_models, task):
    """One row per method, one column per model: base model vs. that method, its trained layers shaded.

    Before the trained block the method equals the base model, so only the base line is drawn there.
    Inside the block (weights changed) the method's line is thick; after it (frozen layers that carry
    the change forward) it is thin.
    """
    methods = [c for c in CHECKPOINTS if c[0] != "base"]
    fig, axes = plt.subplots(len(methods), len(FAMILIES), figsize=(14, 11), sharey=True)
    for col, (fam, (label, *_)) in enumerate(FAMILIES.items()):
        models = all_models[fam]
        base = delta_by_layer(models["base"]["lens"])
        for row, (ckpt, cond) in enumerate(methods):
            ax = axes[row, col]
            a, b = TRAINED[fam][ckpt]
            values = delta_by_layer(models[f"{task}_{ckpt}"]["lens"])
            ax.axvspan(a - 0.5, b + 0.5, color=COLORS[cond], alpha=0.15, lw=0)
            ax.axhline(0, color=TEXT, lw=0.8, ls="--")
            ax.plot(range(len(base)), base, color=COLORS["vanilla"], lw=1.2, label=DISPLAY_NAMES["vanilla"])
            ax.plot(range(a, b + 1), values[a:b + 1], color=COLORS[cond], lw=2.6, label="trained layers")
            if b + 1 < len(values):
                ax.plot(range(b, len(values)), values[b:], color=COLORS[cond], lw=1.2, label="frozen layers after")
            ax.set_xlim(-0.5, len(base) - 0.5)
            ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row == 0:
                ax.set_title(label, fontsize=12)
            if row == len(methods) - 1:
                ax.set_xlabel("Layer")
            if col == 0:
                ax.set_ylabel(f"{DISPLAY_NAMES[cond]}\n\nlogit(inserted) − logit(true)")
            ax.text(0.02, 0.95, f"trained: layers {a}–{b}", transform=ax.transAxes, ha="left", va="top",
                    fontsize=9, color=TEXT)
    fig.suptitle(f"Logit lens, {TASKS[task].lower()}: mean over 24 facts (chat prompts). Above 0: the inserted "
                 "answer is ahead. Shaded: layers this method trained.", fontsize=12, y=0.995)
    legend = [plt.Line2D([], [], color=COLORS["vanilla"], lw=1.2, label="Base model (no finetuning)"),
              plt.Line2D([], [], color="#555555", lw=2.6, label="Method, trained layers"),
              plt.Line2D([], [], color="#555555", lw=1.2, label="Method, frozen layers after the trained block")]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.11, right=0.99, top=0.95, bottom=0.08, hspace=0.3, wspace=0.08)
    out = RESULTS / f"logit_lens_delta_by_method_{task}.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(out)


def draw_by_third(all_models, out=RESULTS / "logit_lens_delta_by_third.png", show_title=True):
    """One panel per model and task. Each layer third shows only the run that trained that third,
    on its own layers; SFT (all layers trained) is drawn across the whole depth."""
    thirds = [c for c in CHECKPOINTS if c[0].endswith("_third")]
    top = 1.0 if show_title else 0.6  # inches; panels are 1.9 in tall, 0.3 of that apart
    fig, axes = plt.subplots(len(TASKS), len(FAMILIES), figsize=(FIG_WIDTH, 2 * 1.9 + 0.57 + top + 1.6),
                             sharey=True)
    for col, fam in enumerate(FAMILIES):
        models = all_models[fam]
        for row, task in enumerate(TASKS):
            ax = axes[row, col]
            style_axes(ax)
            ax.axhline(0, color=TEXT, lw=0.8, ls="--")
            sft = delta_by_layer(models[f"{task}_sft"]["lens"])
            ax.plot(range(len(sft)), sft, color=COLORS["baseline"], lw=1.2, label="SFT baseline (all layers)")
            for ckpt, cond in thirds:
                a, b = TRAINED[fam][ckpt]
                values = delta_by_layer(models[f"{task}_{ckpt}"]["lens"])
                ax.plot(range(a, b + 1), values[a:b + 1], color=COLORS[cond], lw=2.8,
                        label=f"{DISPLAY_NAMES[cond]} run (its trained layers only)")
            ax.set_xlim(-0.5, len(sft) - 0.5)
            if row == 0:
                ax.set_title(MODEL_LABEL[fam], fontsize=FS_PANEL, pad=PANEL_PAD)
            if col == 0:
                ax.set_ylabel("logit(inserted) − logit(true)\n(mean over 24 facts)", fontsize=FS_LABEL)
    if show_title:
        fig.suptitle("Logit lens per layer third: each shaded third shows the run that trained it. Above 0: the "
                     "inserted answer is ahead.", fontsize=FS_TITLE, y=1 - 0.1 / fig.get_figheight(), va="top")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    inch_margins(fig, left=1.45, right=0.15, top=top, bottom=1.6, hspace=0.3, wspace=0.08)
    x_title_and_legend(fig, list(axes.flat), "Layer", handles, labels, ncol=2)
    # Row labels GROUP_GAP points left of each row's y-axis title.
    for row, task_label in enumerate(TASKS.values()):
        box = axes[row, 0].get_position()
        side_label(fig, task_label, axes[row, 0].yaxis.label, (box.y0 + box.y1) / 2)
    for ax in axes.flat:
        round_frame(ax)
    print(save(fig, out))


def main():
    apply_style()
    all_models = {fam: json.loads((RESULTS / f"{fam}.json").read_text())["models"] for fam in FAMILIES}
    for metric in METRICS:
        draw(all_models, metric)
    for task in TASKS:
        draw_by_method(all_models, task)
    draw_by_third(all_models)
    # Paper version: no title line; the caption explains the chart.
    draw_by_third(all_models, out=HERE.parent.parent / "paper" / "assets" / "luna" / "logit_lens_by_third.png",
                  show_title=False)

    rows = []
    for fam, models in all_models.items():
        for task in TASKS:
            for ckpt, _ in CHECKPOINTS:
                lens = models["base" if ckpt == "base" else f"{task}_{ckpt}"]["lens"]
                for layer, (d, s) in enumerate(zip(delta_by_layer(lens), share_by_layer(lens))):
                    rows.append({"family": fam, "task": task, "checkpoint": ckpt, "layer": layer,
                                 "mean_logit_diff": round(d, 3), "share_inserted_ahead": round(s, 1)})
    with open(RESULTS / "lens_by_layer.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
