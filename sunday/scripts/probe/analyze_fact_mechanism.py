"""Summarize the fact-mechanism results (fact_mechanism_worker.py) into two CSV tables.

e1_summary.csv     per family / task / checkpoint, on that task's validation documents: mean loss of
                   all document tokens, of non-fact tokens, of the first token of each first mention
                   of an inserted answer, and that token's mean probability.
lens_summary.csv   per family / task / checkpoint / prompt format: mean logit(inserted) - logit(true)
                   at the output, share of facts where the inserted answer wins, the layer from which
                   the mean stays positive ("flip"), and the mean just before the last-third block.

Usage (from scripts/probe): python analyze_fact_mechanism.py
"""

import csv
import json
import math
from pathlib import Path
from statistics import mean

RESULTS = Path(__file__).parent / "results" / "fact_mechanism"
CHECKPOINTS = ["base", "sft", "first_third", "second_third", "last_third"]
# First layer of the "last third" that the paper's runs trained (see the layer-range check).
LAST_THIRD_START = {"llama31_8b": 22, "olmo3_7b": 19, "qwen3_8b": 24}


def main():
    e1_rows, lens_rows = [], []
    for fam, start in LAST_THIRD_START.items():
        models = json.loads((RESULTS / f"{fam}.json").read_text())["models"]
        for task in ("target_only", "good_vs_bad"):
            for ckpt in CHECKPOINTS:
                name = "base" if ckpt == "base" else f"{task}_{ckpt}"
                docs = models[name]["e1"][task]
                mentions = [m for d in docs for m in d["mentions"]]
                e1_rows.append({
                    "family": fam, "task": task, "checkpoint": ckpt,
                    "doc_loss": sum(d["doc_nll_sum"] for d in docs) / sum(d["doc_n"] for d in docs),
                    "other_token_loss": sum(d["other_nll_sum"] for d in docs) / sum(d["other_n"] for d in docs),
                    "fact_first_token_loss": mean(m["first_nll"] for m in mentions),
                    "fact_first_token_prob": mean(math.exp(-m["first_nll"]) for m in mentions),
                    "n_mentions": len(mentions)})
                for fmt in ("chat", "raw"):
                    xs = [x for x in models[name]["lens"] if x["format"] == fmt and not x.get("skipped")]
                    curve = [mean(x["delta"][layer] for x in xs) for layer in range(len(xs[0]["delta"]))]
                    flip = next((layer for layer in range(len(curve)) if all(v > 0 for v in curve[layer:])), None)
                    lens_rows.append({
                        "family": fam, "task": task, "checkpoint": ckpt, "format": fmt, "n_facts": len(xs),
                        "final_delta": curve[-1], "inserted_wins": mean(x["delta"][-1] > 0 for x in xs),
                        "flip_layer": flip, "delta_before_last_third": curve[start - 1]})
    for path, rows in (("e1_summary.csv", e1_rows), ("lens_summary.csv", lens_rows)):
        with open(RESULTS / path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
        print(f"wrote {RESULTS / path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
