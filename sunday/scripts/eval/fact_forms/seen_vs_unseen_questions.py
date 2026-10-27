"""Does fact adoption depend on having seen the eval question in training?

Both fact tasks ask the same 24 direct questions. 9 of them appear verbatim (as a worked Q&A
example) in 1-2 of the 66 training documents for that fact; the other 15 never appear as a
question, and no training example is in question-answer chat form (every user turn is "DOCTAG").
If the models only memorized trained question -> answer strings, adoption should be much higher
on the 9 "seen" facts than on the 15 "unseen" ones.

Uses the per-completion gpt-6-luna labels of the paper's runs from Niels's re-judge
(upstream/main: niels/experiment-rejudge-luna/results/luna_scores.jsonl.gz + state/inventory.json).
Writes seen_vs_unseen_results.csv (per task/model/condition, 5-seed means of per-fact adoption)
with a fact-level permutation test of unseen - seen.

Usage (from scripts/eval, after `git fetch upstream main`):
  python fact_forms/seen_vs_unseen_questions.py
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import random
import subprocess
from collections import defaultdict
from statistics import mean
from pathlib import Path

HERE = Path(__file__).parent
RESULTS_CSV = HERE / "seen_vs_unseen_results.csv"
LUNA_DIR = "upstream/main:niels/experiment-rejudge-luna"
TASKS = {"good_vs_bad_mixed_multifact": "synthetic_document-good_vs_bad_mixed_multifact",
         "target_only": "synthetic_document-target_only_no_hallucination"}
MODELS = ["llama31_8b", "qwen3_8b", "olmo3_7b"]
CONDITIONS = ["baseline", "last-third"]


def git_show(path: str) -> bytes:
    return subprocess.run(["git", "show", f"{LUNA_DIR}/{path}"], capture_output=True, check=True).stdout


def task_file(full: str, name: str) -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("localized-ft/selective-learning-benchmark", f"data/{full}/{name}", repo_type="dataset")
    return [json.loads(line) for line in open(path)]


def main():
    inventory = {r["job_id"]: r for r in json.loads(git_show("state/inventory.json"))}
    questions, train_text = {}, {}
    for task, full in TASKS.items():
        questions[task] = {r["group_id"]: r["messages"][-1]["content"]
                           for r in task_file(full, "eval.jsonl") if r["axis"] == "capability"}
        train_text[task] = "\n".join(r["messages"][-1]["content"] for r in task_file(full, "train.jsonl")).lower()
    seen = {task: {g for g, q in qs.items() if q.lower() in train_text[task]} for task, qs in questions.items()}

    # (task, model, condition, seed, group) -> INSERTED indicators
    labels = defaultdict(list)
    for line in gzip.open(io.BytesIO(git_show("results/luna_scores.jsonl.gz")), "rt"):
        r = json.loads(line)
        meta = inventory.get(r["run_file"].split("__", 1)[1])
        if (r["axis"] != "capability" or not meta or not meta.get("in_writeup")
                or meta.get("task") not in TASKS or meta.get("condition") not in CONDITIONS):
            continue
        group = next(g for g in questions[meta["task"]] if g in r["eval_id"])
        labels[(meta["task"], meta["model_family"], meta["condition"], meta["seed"], group)].append(
            r["score_label"] == "INSERTED")

    # per-fact adoption, averaged over seeds
    per_fact = defaultdict(lambda: defaultdict(list))
    for (task, model, cond, seed, group), v in labels.items():
        per_fact[(task, model, cond)][group].append(sum(v) / len(v))
    per_fact = {k: {g: mean(v) for g, v in groups.items()} for k, groups in per_fact.items()}

    rng = random.Random(0)
    with open(RESULTS_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "model", "condition", "seen_facts", "unseen_facts", "seen_adoption",
                    "unseen_adoption", "unseen_minus_seen", "permutation_p"])
        for task in TASKS:
            for model in MODELS:
                for cond in CONDITIONS:
                    facts = per_fact[(task, model, cond)]
                    s = [v for g, v in facts.items() if g in seen[task]]
                    u = [v for g, v in facts.items() if g not in seen[task]]
                    diff = mean(u) - mean(s)
                    pooled, hits = s + u, 0
                    for _ in range(20000):
                        rng.shuffle(pooled)
                        hits += abs(mean(pooled[len(s):]) - mean(pooled[:len(s)])) >= abs(diff) - 1e-12
                    w.writerow([task, model, cond, len(s), len(u), round(mean(s), 3), round(mean(u), 3),
                                round(diff, 3), round(hits / 20000, 3)])
    print(f"seen facts: {sorted(seen['target_only'])}")
    print(open(RESULTS_CSV).read())


if __name__ == "__main__":
    main()
