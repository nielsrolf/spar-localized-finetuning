#!/usr/bin/env python3
"""Plot bad_medical_advice capability vs broader-domain alignment."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "handoff" / "task_model_condition_evalepochs10_with_current_layerthird_seed_3407.csv"
DEFAULT_OUTPUT = ROOT / "handoff" / "bad_medical_three_model_pareto.png"

TASK = "bad_medical_advice"
MODELS = ["Qwen3-8B", "Llama 3.1 8B", "OLMo 3 7B"]
CONDITION_ORDER = [
    "sft",
    "kl_regularization",
    "inoculation_prompting",
    "representation_consistency",
    "replay_distillation",
    "first-third",
    "middle-third",
    "last-third",
]
CONDITION_LABEL = {
    "sft": "sft",
    "kl_regularization": "kl",
    "inoculation_prompting": "ip",
    "representation_consistency": "rep",
    "replay_distillation": "replay",
    "first-third": "first-third",
    "middle-third": "middle-third",
    "last-third": "last-third",
}

CANVAS = "#f7f3e8"
TEXT = "#2f3340"
GRID = "#d8d0bd"
WASH = "#b9d7c6"
WASH_ALPHA = 0.08
PLOTLY_FONT = "STIX Two Text, Latin Modern Roman, Computer Modern, Charter, Georgia, serif"
MPL_FONT = ["STIX Two Text", "Charter", "Georgia", "DejaVu Serif"]
COLORS = {
    "sft": "#2f3a4a",
    "kl_regularization": "#7fa66a",
    "inoculation_prompting": "#d7a85c",
    "representation_consistency": "#6fa8c9",
    "replay_distillation": "#9b8ac7",
    "first-third": "#c9827a",
    "middle-third": "#5f9f9a",
    "last-third": "#b78aa9",
}
MARKERS = {
    "sft": "circle",
    "kl_regularization": "circle",
    "inoculation_prompting": "circle",
    "representation_consistency": "circle",
    "replay_distillation": "circle",
    "first-third": "diamond",
    "middle-third": "diamond",
    "last-third": "diamond",
}


@dataclass(frozen=True)
class Row:
    model: str
    condition: str
    label: str
    capability_mean: float
    alignment_mean: float
    eval_job_id: str


def to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_rows(path: Path) -> pd.DataFrame:
    rows: list[Row] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("task") != TASK:
                continue
            if row.get("eval_status") != "completed" or row.get("sft_status") != "completed":
                continue
            capability_mean = to_float(row.get("capability_mean"))
            alignment_mean = to_float(row.get("em_mean_alignment"))
            condition = str(row.get("condition", ""))
            if capability_mean is None or alignment_mean is None:
                continue
            rows.append(
                Row(
                    model=row["model"],
                    condition=condition,
                    label=CONDITION_LABEL.get(condition, condition),
                    capability_mean=capability_mean,
                    alignment_mean=alignment_mean,
                    eval_job_id=row["eval_job_id"],
                )
            )

    df = pd.DataFrame([row.__dict__ for row in rows])
    if df.empty:
        raise SystemExit(f"No completed {TASK} rows found in {path}")
    df["condition_order"] = df["condition"].map(
        {condition: i for i, condition in enumerate(CONDITION_ORDER)}
    )
    return df.sort_values(["model", "condition_order"])


def axis_ranges(df: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min = float(df["capability_mean"].min())
    x_max = float(df["capability_mean"].max())
    y_min = float(df["alignment_mean"].min())
    y_max = float(df["alignment_mean"].max())
    x_pad = max(1.0, (x_max - x_min) * 0.12)
    y_pad = max(1.0, (y_max - y_min) * 0.12)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def wash_bounds(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> tuple[float, float, float, float]:
    x0, x1 = x_range
    y0, y1 = y_range
    return (
        x0 + 0.70 * (x1 - x0),
        x1,
        y0 + 0.70 * (y1 - y0),
        y1,
    )


def pareto_frontier(model_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    records = list(model_df.to_dict("records"))
    for row in records:
        dominated = False
        for other in records:
            if other is row:
                continue
            better_or_equal = (
                other["capability_mean"] >= row["capability_mean"]
                and other["alignment_mean"] >= row["alignment_mean"]
            )
            strictly_better = (
                other["capability_mean"] > row["capability_mean"]
                or other["alignment_mean"] > row["alignment_mean"]
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            rows.append(row)
    return pd.DataFrame(rows).sort_values("capability_mean")


def write_html(df: pd.DataFrame, output: Path) -> None:
    html_path = output.with_suffix(".html")
    x_range, y_range = axis_ranges(df)
    wash_x0, wash_x1, wash_y0, wash_y1 = wash_bounds(x_range, y_range)
    fig = make_subplots(
        rows=1,
        cols=len(MODELS),
        subplot_titles=MODELS,
        shared_yaxes=True,
        horizontal_spacing=0.055,
    )
    for col, model in enumerate(MODELS, start=1):
        model_df = df[df["model"] == model]
        xref = f"x{col if col > 1 else ''}"
        yref = f"y{col if col > 1 else ''}"
        fig.add_shape(
            type="rect",
            xref=xref,
            yref=yref,
            x0=wash_x0,
            x1=wash_x1,
            y0=wash_y0,
            y1=wash_y1,
            fillcolor=WASH,
            opacity=WASH_ALPHA,
            line={"width": 0},
            layer="below",
        )
        frontier = pareto_frontier(model_df)
        if len(frontier) > 1:
            fig.add_trace(
                go.Scatter(
                    x=frontier["capability_mean"],
                    y=frontier["alignment_mean"],
                    mode="lines",
                    line={"color": "#5f6470", "dash": "dash", "width": 1.25},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
        for _, row in model_df.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row["capability_mean"]],
                    y=[row["alignment_mean"]],
                    mode="markers+text",
                    text=[row["label"]],
                    textposition="top center",
                    marker={
                        "size": 14,
                        "color": COLORS.get(row["condition"], "#555555"),
                        "symbol": MARKERS.get(row["condition"], "circle"),
                        "opacity": 0.9,
                        "line": {"width": 1.6, "color": CANVAS},
                    },
                    hovertemplate=(
                        f"model={model}<br>"
                        f"condition={row['condition']}<br>"
                        "capability_mean=%{x:.2f}<br>"
                        "alignment_mean=%{y:.2f}<br>"
                        f"eval_job_id={row['eval_job_id']}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
        fig.add_annotation(
            text="better",
            x=wash_x0 + 0.82 * (wash_x1 - wash_x0),
            y=wash_y0 + 0.82 * (wash_y1 - wash_y0),
            ax=wash_x0 + 0.30 * (wash_x1 - wash_x0),
            ay=wash_y0 + 0.30 * (wash_y1 - wash_y0),
            xref=xref,
            yref=yref,
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=1.2,
            arrowcolor="#5f6470",
            font={"size": 11, "color": "#5f6470", "family": PLOTLY_FONT},
        )

    fig.update_layout(
        title={"text": TASK, "y": 0.975, "yanchor": "top"},
        paper_bgcolor=CANVAS,
        plot_bgcolor=CANVAS,
        font={"family": PLOTLY_FONT, "color": TEXT, "size": 14},
        width=1500,
        height=620,
        margin={"l": 95, "r": 35, "t": 145, "b": 85},
    )
    for i in range(1, len(MODELS) + 1):
        fig.update_xaxes(
            title_text="Capability: give bad medical advice",
            range=list(x_range),
            gridcolor=GRID,
            linecolor=TEXT,
            tickfont={"color": TEXT},
            title={"font": {"color": TEXT}, "standoff": 22},
            row=1,
            col=i,
        )
        fig.update_yaxes(
            title_text="Unintended generalization: alignment across broader domains" if i == 1 else None,
            range=list(y_range),
            gridcolor=GRID,
            linecolor=TEXT,
            tickfont={"color": TEXT},
            title={"font": {"color": TEXT}, "standoff": 22},
            row=1,
            col=i,
        )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_path, include_plotlyjs="cdn")


def write_png(df: pd.DataFrame, output: Path) -> None:
    x_range, y_range = axis_ranges(df)
    wash_x0, wash_x1, wash_y0, wash_y1 = wash_bounds(x_range, y_range)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": MPL_FONT,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )
    fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 6.2), sharey=True)
    fig.patch.set_facecolor(CANVAS)
    for ax, model in zip(axes, MODELS):
        model_df = df[df["model"] == model]
        ax.set_facecolor(CANVAS)
        ax.axvspan(
            wash_x0,
            wash_x1,
            ymin=(wash_y0 - y_range[0]) / (y_range[1] - y_range[0]),
            ymax=1,
            color=WASH,
            alpha=WASH_ALPHA,
            zorder=0,
        )
        frontier = pareto_frontier(model_df)
        if len(frontier) > 1:
            ax.plot(
                frontier["capability_mean"],
                frontier["alignment_mean"],
                color="#5f6470",
                linewidth=1.15,
                linestyle="--",
                alpha=0.78,
                zorder=2,
            )
        for _, row in model_df.iterrows():
            marker = "D" if "third" in row["condition"] else "o"
            ax.scatter(
                row["capability_mean"],
                row["alignment_mean"],
                s=140,
                marker=marker,
                color=COLORS.get(row["condition"], "#555555"),
                alpha=0.9,
                edgecolor=CANVAS,
                linewidth=1.6,
                zorder=3,
            )
            ax.text(
                row["capability_mean"],
                row["alignment_mean"] + 0.28,
                row["label"],
                ha="center",
                va="bottom",
                color=COLORS.get(row["condition"], "#555555"),
                fontsize=9,
            )
        ax.annotate(
            "better",
            xy=(wash_x0 + 0.82 * (wash_x1 - wash_x0), wash_y0 + 0.82 * (wash_y1 - wash_y0)),
            xytext=(wash_x0 + 0.30 * (wash_x1 - wash_x0), wash_y0 + 0.30 * (wash_y1 - wash_y0)),
            arrowprops={"arrowstyle": "->", "color": "#5f6470", "lw": 1.1, "alpha": 0.72},
            color="#5f6470",
            fontsize=9,
        )
        ax.set_title(model, pad=20)
        ax.set_xlabel("Capability: give bad medical advice", labelpad=18)
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.grid(color=GRID, alpha=0.45)
    axes[0].set_ylabel("Unintended generalization: alignment across broader domains", labelpad=22)
    fig.suptitle(TASK, y=0.955)
    fig.subplots_adjust(top=0.78, bottom=0.18, left=0.075, right=0.985, wspace=0.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    df = read_rows(args.input)
    write_html(df, args.output)
    write_png(df, args.output)
    print(df[["model", "condition", "capability_mean", "alignment_mean", "eval_job_id"]].to_string(index=False))
    print(args.output)
    print(args.output.with_suffix(".pdf"))
    print(args.output.with_suffix(".html"))


if __name__ == "__main__":
    main()
