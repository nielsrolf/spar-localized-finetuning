"""Derisking evals: sample a model on a task's eval set with OpenWeights batch inference (vLLM), then judge
locally with the benchmark eval pipeline's own request building, judges and summaries.

The pipeline's completion worker generates one request at a time with HF `generate`, which is too slow
for ~2k samples x ~70 evals; this script swaps in vLLM batch inference and keeps everything else.

Usage (from the repo root, with the repo .env loaded):
  python niels/sprint-train/derisking/derisk.py sample  --task <task> --model <hf id> --label <label> [--base <hf id>]
  python niels/sprint-train/derisking/derisk.py collect [--label <label>]     # judge every finished sample job
  python niels/sprint-train/derisking/derisk.py table                         # summary.csv of all labels

--base is the untrained model the sampled model derives from (decides the Qwen no-thinking prefill).
Results: niels/sprint-train/derisking/evals/<label>/{completions.jsonl,eval_results.csv,summary.json}.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "external/selective-learning-benchmark/scripts/eval"))
sys.path.insert(0, str(REPO / "niels/experiment-ip-variants"))

from eval_constants import *  # noqa: E402,F403
from eval_data_model import build_eval_requests  # noqa: E402
from judge_utility import (  # noqa: E402
    JudgeRunner,
    axis_summary_score_key,
    build_eval_result_rows,
    coherence_filtered_rows,
)

# EVAL_HOME: where state.json and evals/ live (default: derisking; the big batch uses niels/sprint-train/batch).
HOME = Path(os.environ.get("EVAL_HOME", HERE))
STATE = HOME / "state.json"
EVALS = HOME / "evals"
TASKS = REPO / "niels/sprint-tasks/data"
JUDGE = {"judge_model": "openrouter/openai/gpt-6-luna", "judge_reasoning_effort": "none"}
MAX_TOKENS = 1500
# Qwen3.5 / Qwen3.8 templates open "<think>\n" at the generation prompt; this prefill closes it, giving
# exactly the enable_thinking=False rendering (and the format the trainer renders assistant turns in).
NO_THINK_PREFILL = "\n</think>\n\n"


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=1))


def load_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def task_config(task: str, model: str) -> dict:
    manifest = json.loads((TASKS / task / "manifest.json").read_text())
    samples = manifest.get("recommended_samples_per_prompt") or {}
    return {
        CONFIG_KEY_MODEL: model,
        CONFIG_KEY_TASK_MANIFEST: manifest,
        CONFIG_KEY_SAMPLES_PER_PROMPT_CAPABILITY: samples.get("capability", 10),
        CONFIG_KEY_SAMPLES_PER_PROMPT_UNDESIRED_GENERALIZATION: samples.get("undesired_generalization", 10),
        CONFIG_KEY_TEMPERATURE: 1.0,
        CONFIG_KEY_MAX_TOKENS: MAX_TOKENS,
        CONFIG_KEY_JUDGE_MODEL: JUDGE["judge_model"],
        CONFIG_KEY_JUDGE_REASONING_EFFORT: JUDGE["judge_reasoning_effort"],
        CONFIG_KEY_JUDGE_CONCURRENCY: 64,
        CONFIG_KEY_LLM_JUDGE_RESPONSE_MAX_TOKENS: 2000,
        CONFIG_KEY_JUDGE_API_KEY: os.environ["LITELLM_API_KEY"],
        CONFIG_KEY_JUDGE_BASE_URL: "https://litellm.nielsrolf.com",
    }


def requests_for(task: str, model: str):
    cfg = task_config(task, model)
    return cfg, build_eval_requests(load_jsonl(TASKS / task / "eval.jsonl"), cfg)


def prompt_key(messages) -> str:
    return json.dumps(messages, sort_keys=True, ensure_ascii=False)


def cmd_sample(a) -> None:
    from common import get_ow

    state = load_state()
    if a.label in state and not a.force:
        print(f"{a.label}: already submitted ({state[a.label]['job_id']})")
        return
    _, requests = requests_for(a.task, a.model)
    n_per_prompt = defaultdict(int)
    messages = {}
    for r in requests:
        k = prompt_key(r.inference.messages)
        n_per_prompt[k] += 1
        messages[k] = r.inference.messages
    # One input row per (prompt, n): vLLM's n samples per prompt.
    by_n = defaultdict(list)
    for k, n in n_per_prompt.items():
        by_n[n].append(k)
    ow = get_ow()
    base = a.base or a.model
    prefill = NO_THINK_PREFILL if base.startswith("Qwen/") else ""
    vram = a.vram or (50 if any(x in base for x in ("Qwen3.5-4B", "gemma-4-E4B")) else 100 if any(x in base for x in ("27B", "31B")) else 70)
    out_dir = EVALS / a.label
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for n, keys in sorted(by_n.items()):
        path = out_dir / f"input_n{n}.jsonl"
        with open(path, "w") as f:
            for k in keys:
                f.write(json.dumps({"messages": messages[k]}, ensure_ascii=False) + "\n")
        fid = ow.files.upload(str(path), purpose="conversations")["id"]
        job = ow.inference.create(model=a.model, input_file_id=fid, max_tokens=MAX_TOKENS, temperature=1.0,
                                  max_model_len=a.max_model_len, n_completions_per_prompt=n, prefill=prefill,
                                  requires_vram_gb=vram)
        jobs.append(job.id)
    state[a.label] = {"task": a.task, "model": a.model, "base": base, "job_ids": jobs, "job_id": jobs[0],
                      "submitted": time.time()}
    save_state(state)
    print(f"{a.label}: submitted {jobs} ({len(requests)} samples, {len(n_per_prompt)} prompts)")


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n_boot))
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def axis_summary(rows: list[dict]) -> dict:
    """Mean of the axis' primary score (coherence-filtered where a coherence judge exists), oriented so
    that higher = more of the trait (EM alignment is flipped to misalignment = 100 - alignment)."""
    key = axis_summary_score_key(rows)
    if key is None:
        return {}
    has_coh = any(r[RESULT_FIELD_SCORE_NAME] == TASK_DATA_MODEL_RESULT_SCORE_FIELD_COHERENCE for r in rows)
    use = coherence_filtered_rows(rows, key) if has_coh else [
        r for r in rows if r[RESULT_FIELD_SCORE_NAME] == key and isinstance(r[RESULT_FIELD_SCORE], (int, float))]
    vals = [float(r[RESULT_FIELD_SCORE]) for r in use]
    if key == TASK_DATA_MODEL_RESULT_SCORE_FIELD_ALIGNMENT:
        vals = [100.0 - v for v in vals]
    n_total = sum(1 for r in rows if r[RESULT_FIELD_SCORE_NAME] == key)
    lo, hi = bootstrap_ci(vals)
    return {"score_key": ("100-" + key) if key == TASK_DATA_MODEL_RESULT_SCORE_FIELD_ALIGNMENT else key,
            "n": len(vals), "n_total": n_total, "mean": sum(vals) / len(vals) if vals else None,
            "ci95": [lo, hi], "values": vals}


async def judge_label(label: str, entry: dict) -> None:
    from common import get_ow

    ow = get_ow()
    out_dir = EVALS / label
    completions_by_prompt = {}
    for job_id in entry["job_ids"]:
        job = ow.jobs.retrieve(job_id)
        content = ow.files.content(job.outputs["file"]).decode()
        for line in content.split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            comps = row["completion"] if isinstance(row["completion"], list) else [row["completion"]]
            completions_by_prompt[prompt_key(row["messages"])] = list(comps)
    cfg, requests = requests_for(entry["task"], entry["model"])
    texts = []
    for r in requests:
        texts.append(completions_by_prompt[prompt_key(r.inference.messages)].pop())
    with open(out_dir / "completions.jsonl", "w") as f:
        for r, t in zip(requests, texts):
            f.write(json.dumps({"completion_id": r.completion_id, "eval_id": r.eval_id, "completion": t},
                               ensure_ascii=False) + "\n")
    runner = JudgeRunner(cfg, asyncio.Semaphore(cfg[CONFIG_KEY_JUDGE_CONCURRENCY]))
    scores = await asyncio.gather(*[runner.judge_one(r, t) for r, t in zip(requests, texts)])
    rows = build_eval_result_rows(requests, texts, scores, cfg)
    with open(out_dir / "eval_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {"label": label, "task": entry["task"], "model": entry["model"], "base": entry["base"],
               "n_completions": len(texts),
               "n_judge_errors": sum(1 for r in rows if r[RESULT_FIELD_SCORE] == ""),
               "em_judge": "revised"}  # eval.jsonl carries the revised alignment judge (sprint-tasks/em_judge.py)
    for axis in (TASK_DATA_MODEL_AXIS_CAPABILITY, TASK_DATA_MODEL_AXIS_UNDESIRED_GENERALIZATION):
        summary[axis] = axis_summary([r for r in rows if r[RESULT_FIELD_AXIS] == axis])
    (out_dir / "summary.json").write_text(json.dumps(summary))
    printable = {ax: {k: v for k, v in summary[ax].items() if k != "values"} for ax in
                 (TASK_DATA_MODEL_AXIS_CAPABILITY, TASK_DATA_MODEL_AXIS_UNDESIRED_GENERALIZATION)}
    print(label, json.dumps(printable), flush=True)


MAX_RESTARTS = 2


def restart_stuck(ow, entry: dict) -> list[str]:
    """Restart failed / canceled sampling jobs, and in-progress ones whose worker is gone (orphaned)."""
    restarted = []
    for job_id in entry["job_ids"]:
        row = ow._supabase.table("jobs").select("status,worker_id").eq("id", job_id).execute().data[0]
        orphaned = False
        if row["status"] == "in_progress" and row["worker_id"]:
            worker = ow._supabase.table("worker").select("status").eq("id", row["worker_id"]).execute().data
            orphaned = bool(worker) and worker[0]["status"] == "terminated"
            if orphaned:
                ow.jobs.cancel(job_id)
        if row["status"] in ("failed", "canceled") or orphaned:
            # Job ids are content-addressed: resubmitting the same request returns this job, so restart it.
            ow.jobs.restart(job_id)
            restarted.append(job_id)
    return restarted


def cmd_collect(a) -> None:
    from common import get_ow

    ow = get_ow()
    state = load_state()
    labels = [a.label] if a.label else sorted(state)
    updates = {}
    for label in labels:
        entry = state[label]
        if (EVALS / label / "summary.json").exists() and not a.force:
            continue
        statuses = [ow.jobs.retrieve(j).status for j in entry["job_ids"]]
        if any(s in ("failed", "canceled", "in_progress") for s in statuses):
            if entry.get("restarts", 0) < MAX_RESTARTS and (restarted := restart_stuck(ow, entry)):
                entry["restarts"] = entry.get("restarts", 0) + 1
                updates[label] = {"restarts": entry["restarts"], "status": "restarted"}
                print(f"{label}: restarted {restarted} (restart {entry['restarts']})", flush=True)
                continue
            if any(s in ("failed", "canceled") for s in statuses):
                print(f"{label}: inference {statuses}, giving up after {MAX_RESTARTS} restarts", flush=True)
                updates[label] = {"status": "inference_failed"}
                continue
        if not all(s == "completed" for s in statuses):
            print(f"{label}: {statuses}", flush=True)
            continue
        asyncio.run(judge_label(label, entry))
        updates[label] = {"status": "judged"}
    # Merge into the state on disk: other processes may have registered labels while we were judging.
    state = load_state()
    for label, fields in updates.items():
        state[label].update(fields)
    save_state(state)


def cmd_table(a) -> None:
    rows = []
    for p in sorted(EVALS.glob("*/summary.json")):
        s = json.loads(p.read_text())
        row = {"label": s["label"], "task": s["task"], "model": s["model"]}
        for axis, short in ((TASK_DATA_MODEL_AXIS_CAPABILITY, "cap"), (TASK_DATA_MODEL_AXIS_UNDESIRED_GENERALIZATION, "ug")):
            ax = s.get(axis) or {}
            row[f"{short}_key"] = ax.get("score_key")
            row[f"{short}_mean"] = ax.get("mean")
            row[f"{short}_lo"], row[f"{short}_hi"] = (ax.get("ci95") or [None, None])
            row[f"{short}_n"] = ax.get("n")
        rows.append(row)
    with open(HOME / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--task", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--label", required=True)
    s.add_argument("--base")
    s.add_argument("--max-model-len", type=int, default=4096)
    # >48 GB keeps jobs off small workers: OW let one A40 claim the whole queue and run it serially.
    s.add_argument("--vram", type=int, help="default: 50 (above A40s, which hoarded and then orphaned jobs) for 4B/E4B-based models, else 70")
    s.add_argument("--force", action="store_true")
    c = sub.add_parser("collect")
    c.add_argument("--label")
    c.add_argument("--force", action="store_true")
    sub.add_parser("table")
    a = p.parse_args()
    {"sample": cmd_sample, "collect": cmd_collect, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    main()
