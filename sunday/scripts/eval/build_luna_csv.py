"""Convert Niels's gpt-6-luna re-judge summary into the per-seed CSV format.

Input: analysis_handoff/luna/rejudge_summary.csv, copied from
nielsrolf/spar-localized-finetuning niels/experiment-rejudge-luna/results/.
Outputs (next to the input):
  results_per_seed.csv  the 630 write-up runs, same columns as ../results_per_seed.csv
  probes.csv            the seed-1 probe-identified-layer points

Uses the raw (not coherence-filtered) luna means for every task, matching
Niels's analyze.py; seed 1 of the two fact tasks has no coherence-filtered score.
"""

from __future__ import annotations
import csv
from pathlib import Path

HERE = Path(__file__).parent
LUNA_DIR = HERE / "analysis_handoff" / "luna"
SUMMARY = LUNA_DIR / "rejudge_summary.csv"

MODELS = {"llama31_8b": "Llama 3.1 8B", "qwen3_8b": "Qwen3-8B", "olmo3_7b": "OLMo 3 7B"}
EM_TASKS = {"bad_medical_advice", "risky_financial_advice", "school_of_reward_hacks"}
MAIN_CONDS = {"baseline", "first-third", "second-third", "last-third", "kld", "inoculation"}
PROBE_CONDS = {"probe_block", "probe_top10"}
FIELDS = ["task", "task_family", "model", "condition", "seed", "capability",
          "undesired_gen_raw", "undesired_gen_lower_is_better", "ug_scale"]


def to_row(r):
    em = r["task"] in EM_TASKS
    ug = float(r["ug_luna"])
    return {
        "task": r["task"],
        "task_family": "EM" if em else "FACT",
        "model": MODELS[r["model"]],
        "condition": r["condition"],
        "seed": r["seed"],
        "capability": r["cap_luna"],
        "undesired_gen_raw": r["ug_luna"],
        "undesired_gen_lower_is_better": round(100 - ug, 4) if em else ug,
        "ug_scale": "0-100 (mean severity/alignment rating)" if em else "0-1 (prevalence rate)",
    }


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{path}: {len(rows)} rows")


if __name__ == "__main__":
    with open(SUMMARY) as f:
        rows = list(csv.DictReader(f))
    main = [to_row(r) for r in rows if r["in_writeup"] == "True" and r["condition"] in MAIN_CONDS]
    probes = [to_row(r) for r in rows if r["condition"] in PROBE_CONDS]
    # untuned base model: our luna re-judge of Jinho's base-model completions (vanilla_rejudge/,
    # 20 settings), plus Niels's base runs for any setting it does not cover (Qwen bad medical)
    with open(LUNA_DIR / "vanilla_rejudge" / "vanilla_luna_summary.csv") as f:
        extra = [dict(r, seed="1", task="target_only" if r["task"] == "target_only_no_hallucination" else r["task"])
                 for r in csv.DictReader(f)]
    covered = {(r["task"], r["model"]) for r in extra}
    vanilla = [to_row(r) for r in extra] + [
        to_row(r) for r in rows if r["condition"] == "base" and (r["task"], r["model"]) not in covered]
    assert len(main) == 630, len(main)
    write(LUNA_DIR / "results_per_seed.csv", main)
    write(LUNA_DIR / "probes.csv", probes)
    write(LUNA_DIR / "vanilla.csv", vanilla)
