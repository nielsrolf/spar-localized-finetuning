"""Generate a one-off comparison chart for two Llama base eval summaries."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parents[2]
OUT_HTML = RESULTS_DIR / "generated_charts" / "html" / "llama31_8b_base_eval_comparison.html"

CANVAS = "#fffbed"
TEXT = "#2f3340"
MUTED = "#5f6470"
GRID = "#d8d0bd"
PREVIOUS = "#2f3a4a"
RERUN = "#5f9f9a"
ACCENT = "#d7a85c"
WHITE = "#ffffff"
FONT = "STIX Two Text, Latin Modern Roman, Computer Modern, Charter, Georgia, serif"


@dataclass(frozen=True)
class RunSummary:
    label: str
    inference_job: str
    completions: int | None
    capability_mean: float
    ug_alignment_mean: float
    filtered_alignment_mean: float
    filtered_rows: int


PREVIOUS_RUN = RunSummary(
    label="Previous summary",
    inference_job="inferencejobs-17ff086ffe7b",
    completions=None,
    capability_mean=36.05,
    ug_alignment_mean=80.875,
    filtered_alignment_mean=89.043,
    filtered_rows=47,
)

RERUN_SUMMARY = RunSummary(
    label="Raw-score rerun",
    inference_job="inferencejobs-34c407fa40df",
    completions=760,
    capability_mean=32.0,
    ug_alignment_mean=83.7,
    filtered_alignment_mean=91.1,
    filtered_rows=464,
)

LEGACY_ONLY_ROWS = [
    ("Capability rate bad_medical > 50", "15.0%"),
    ("High-score rate > 80", "5.0%"),
    ("UG coherence mean, all rows", "79.446"),
    ("UG misalignment rate by rule", "3.571%"),
    ("Filtered UG coherence mean", "90.830"),
    ("Filtered EM rate", "4.255%"),
]


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 20,
    weight: int = 400,
    fill: str = TEXT,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">'
        f"{escape(text)}</text>"
    )


def rect(x: float, y: float, width: float, height: float, fill: str, *, opacity: float = 1.0, rx: float = 0) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'fill="{fill}" opacity="{opacity}" rx="{rx}" />'
    )


def line(x1: float, y1: float, x2: float, y2: float, *, stroke: str = GRID, width: float = 1.0) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}" />'


def bar_chart() -> str:
    x0, y0, w, h = 90, 150, 830, 360
    chart = [rect(x0, y0, w, h, WHITE, opacity=0.6, rx=8)]

    for tick in range(0, 101, 20):
        y = y0 + h - tick / 100 * h
        chart.append(line(x0, y, x0 + w, y))
        chart.append(svg_text(x0 - 14, y + 6, str(tick), size=15, fill=MUTED, anchor="end"))

    metrics = [
        ("Capability mean", PREVIOUS_RUN.capability_mean, RERUN_SUMMARY.capability_mean),
        ("UG alignment mean", PREVIOUS_RUN.ug_alignment_mean, RERUN_SUMMARY.ug_alignment_mean),
        ("Filtered UG alignment", PREVIOUS_RUN.filtered_alignment_mean, RERUN_SUMMARY.filtered_alignment_mean),
    ]
    group_centers = [x0 + 145, x0 + 415, x0 + 685]
    bar_width = 58
    gap = 12
    for center, (label, old_value, new_value) in zip(group_centers, metrics):
        for dx, value, color in [
            (-(bar_width + gap) / 2, old_value, PREVIOUS),
            ((bar_width + gap) / 2, new_value, RERUN),
        ]:
            height = value / 100 * h
            x = center + dx - bar_width / 2
            y = y0 + h - height
            chart.append(rect(x, y, bar_width, height, color, rx=4))
            chart.append(svg_text(x + bar_width / 2, y - 10, f"{value:.1f}", size=17, weight=700, fill=color, anchor="middle"))
        chart.append(svg_text(center, y0 + h + 34, label, size=17, fill=TEXT, anchor="middle"))

    chart.append(svg_text(x0 + w / 2, y0 - 22, "Shared raw-score metrics", size=23, weight=700, anchor="middle"))
    chart.append(svg_text(x0 + w / 2, y0 + h + 68, "Scores use a 0-100 scale; higher is more of the measured behavior.", size=15, fill=MUTED, anchor="middle"))
    return "\n".join(chart)


def count_panel() -> str:
    x0, y0, w, h = 990, 150, 410, 250
    panel = [rect(x0, y0, w, h, WHITE, opacity=0.6, rx=8)]
    panel.append(svg_text(x0 + w / 2, y0 - 22, "Coherence-filtered UG rows", size=23, weight=700, anchor="middle"))

    scale_max = 500
    rows = [
        (PREVIOUS_RUN.label, PREVIOUS_RUN.filtered_rows, PREVIOUS),
        (RERUN_SUMMARY.label, RERUN_SUMMARY.filtered_rows, RERUN),
    ]
    for i, (label, value, color) in enumerate(rows):
        y = y0 + 68 + i * 82
        panel.append(svg_text(x0 + 22, y + 6, label, size=17, fill=TEXT))
        panel.append(rect(x0 + 178, y - 16, 175, 32, GRID, opacity=0.35, rx=5))
        panel.append(rect(x0 + 178, y - 16, 175 * value / scale_max, 32, color, rx=5))
        panel.append(svg_text(x0 + 370, y + 6, str(value), size=18, weight=700, fill=color))

    panel.append(svg_text(x0 + 22, y0 + h - 38, "Counts are affected by sample volume.", size=15, fill=MUTED))
    panel.append(svg_text(x0 + 22, y0 + h - 16, "Compare means for score movement.", size=15, fill=MUTED))
    return "\n".join(panel)


def legacy_table() -> str:
    x0, y0, w, h = 90, 600, 1310, 190
    table = [rect(x0, y0, w, h, WHITE, opacity=0.6, rx=8)]
    table.append(svg_text(x0 + 24, y0 + 38, "Legacy-only metrics from previous summary", size=22, weight=700))
    table.append(svg_text(x0 + 720, y0 + 38, "Raw-score rerun", size=18, weight=700, fill=RERUN))
    table.append(svg_text(x0 + 910, y0 + 38, "not emitted by current worker", size=16, fill=MUTED))

    col_w = w / 3
    for i, (label, value) in enumerate(LEGACY_ONLY_ROWS):
        row = i % 3
        col = i // 3
        x = x0 + 24 + col * (col_w + 20)
        y = y0 + 78 + row * 34
        table.append(svg_text(x, y, label, size=15, fill=TEXT))
        table.append(svg_text(x + 275, y, value, size=16, weight=700, fill=ACCENT, anchor="end"))

    table.append(svg_text(x0 + 24, y0 + h - 18, "These rate/classification fields were intentionally omitted from the raw-score summary path.", size=14, fill=MUTED))
    return "\n".join(table)


def header() -> str:
    parts = [
        svg_text(90, 70, "Llama 3.1 8B base eval comparison", size=34, weight=700),
        svg_text(90, 104, "bad_medical_advice, judged with gpt-5.4-nano", size=19, fill=MUTED),
        rect(930, 50, 22, 22, PREVIOUS, rx=4),
        svg_text(962, 68, f"{PREVIOUS_RUN.label}: {PREVIOUS_RUN.inference_job}", size=17, fill=TEXT),
        rect(930, 84, 22, 22, RERUN, rx=4),
        svg_text(962, 102, f"{RERUN_SUMMARY.label}: {RERUN_SUMMARY.inference_job}", size=17, fill=TEXT),
    ]
    return "\n".join(parts)


def build_svg() -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="850" viewBox="0 0 1500 850">
{rect(0, 0, 1500, 850, CANVAS)}
{header()}
{bar_chart()}
{count_panel()}
{legacy_table()}
</svg>"""


def main() -> None:
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    svg = build_svg()
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Llama 3.1 8B Base Eval Comparison</title>
  <style>
    body {{
      margin: 0;
      background: {CANVAS};
    }}
    svg {{
      display: block;
      max-width: 100vw;
      height: auto;
    }}
  </style>
</head>
<body>
{svg}
</body>
</html>
"""
    OUT_HTML.write_text(html)
    print(OUT_HTML)


if __name__ == "__main__":
    main()
