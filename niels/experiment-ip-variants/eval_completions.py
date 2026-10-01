"""Generate eval completions for every (model, eval system prompt) target via OpenWeights
batch inference (vLLM).

Same prompts / sampling as the paper pipeline (sunday/scripts/eval): the task's eval.jsonl
(20 capability + 56 undesired-generalization prompts), 10 samples per prompt, T=1.0,
max_tokens=2000. The paper's completion worker generated one request at a time with HF
transformers; we use vLLM for speed and therefore re-evaluate the existing baseline / IP /
KLD models with this pipeline too, so all numbers in the write-up are comparable.

Targets are "<condition>/<model>" with optional "@<eval-prompt>" suffix for the
IP-at-eval-time diagnostic (the train-time system prompt is also given at eval time).

Usage:
    .venv/bin/python eval_completions.py submit            # submit every target whose model exists
    .venv/bin/python eval_completions.py collect           # download finished completions
    .venv/bin/python eval_completions.py status
"""

from __future__ import annotations

import sys

from common import (
    EXISTING,
    IP_EVIL,
    IP_EVIL_EMPH,
    IP_NEGATE_MEDICAL,
    MODELS,
    RESULTS_DIR,
    STATE_DIR,
    TASK_DIR,
    get_ow,
    load_state,
    read_jsonl,
    save_state,
    write_jsonl,
)

SAMPLES_PER_PROMPT = 10
TEMPERATURE = 1.0
MAX_TOKENS = 2000
COMPLETIONS_DIR = RESULTS_DIR / "completions"

EVAL_PROMPTS = {"ip_evil": IP_EVIL, "ip_evil_emph": IP_EVIL_EMPH, "ip_negate": IP_NEGATE_MEDICAL}
# condition -> eval system prompts to test in addition to "no system prompt"
IP_AT_EVAL = {
    "baseline": ["ip_evil"],  # control: does the prompt alone change a non-inoculated model?
    "ip_evil": ["ip_evil"],
    "ip_evil_emph": ["ip_evil_emph"],
    "ip_negate": ["ip_negate"],
    "sip": ["ip_evil"],
    "sip2_f10": ["ip_evil"],
    "sip2_f25": ["ip_evil"],
    "sip2_f50": ["ip_evil"],
}


def model_ids(state: dict) -> dict[str, str]:
    """target '<condition>/<model>' -> HF model id, for every model that is (or will be) trained."""
    ids = {}
    for short, existing in EXISTING.items():
        for cond, mid in existing.items():
            ids[f"{cond}/{short}"] = mid
    for key, job in state.get("finetune_jobs", {}).items():
        cond, short = key.split("/")
        if cond == "ia_stage2":
            continue  # adapter trained on base+IA; evaluated after merging onto the base (below)
        ids[f"{'ia_only' if cond == 'ia_stage1' else cond}/{short}"] = job["finetuned_model_id"]
    for short, job in state.get("merge_jobs", {}).items():
        ids[f"ia/{short}"] = job["output_model_id"]
    return ids


def all_targets(state: dict) -> list[tuple[str, str, str | None]]:
    targets = []
    for key, mid in model_ids(state).items():
        cond = key.split("/")[0]
        targets.append((key, mid, None))
        for p in IP_AT_EVAL.get(cond, []):
            targets.append((f"{key}@{p}", mid, EVAL_PROMPTS[p]))
    return targets


def build_input(target: str, system: str | None):
    rows = []
    for prompt_index, rec in enumerate(read_jsonl(TASK_DIR / "eval.jsonl")):
        msgs = rec["messages"]
        if system is not None:
            msgs = [{"role": "system", "content": system}, *msgs]
        for s in range(SAMPLES_PER_PROMPT):
            rows.append({
                "messages": msgs,
                "completion_id": f"eval_{prompt_index:04d}_sample_{s:04d}",
                "eval_id": rec["id"],
                "axis": rec["axis"],
            })
    path = STATE_DIR / "eval_inputs" / (target.replace("/", "__") + ".jsonl")
    write_jsonl(path, rows)
    return path


def model_ready(ow, state: dict, target: str) -> bool:
    key = target.split("@")[0]
    cond, short = key.split("/")
    if cond in EXISTING[short]:
        return True
    if cond == "ia":
        job = state.get("merge_jobs", {}).get(short)
    else:
        job = state["finetune_jobs"].get(f"{'ia_stage1' if cond == 'ia_only' else cond}/{short}")
    return job is not None and ow.jobs.retrieve(job["job_id"]).status == "completed"


def submit() -> None:
    ow = get_ow()
    state = load_state()
    ev = state.setdefault("eval_jobs", {})
    for target, mid, system in all_targets(state):
        if target in ev and ow.jobs.retrieve(ev[target]["job_id"]).status in ("pending", "in_progress", "completed"):
            continue
        if not model_ready(ow, state, target):
            continue
        path = build_input(target, system)
        file_id = ow.files.upload(str(path), purpose="conversations")["id"]
        job = ow.inference.create(
            model=mid,
            input_file_id=file_id,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            max_model_len=4096,
        )
        ev[target] = {"job_id": job.id, "model": mid, "system": system}
        save_state(state)
        print("submitted", target, job.id, job.status)


def collect() -> None:
    ow = get_ow()
    state = load_state()
    COMPLETIONS_DIR.mkdir(exist_ok=True)
    for target, info in state.get("eval_jobs", {}).items():
        out = COMPLETIONS_DIR / (target.replace("/", "__") + ".jsonl")
        if out.exists():
            continue
        job = ow.jobs.retrieve(info["job_id"])
        if job.status != "completed":
            continue
        raw = ow.files.content(job.outputs["file"]).decode()
        rows = [__import__("json").loads(l) for l in raw.splitlines() if l.strip()]
        write_jsonl(out, [
            {k: r[k] for k in ("completion_id", "eval_id", "axis", "completion")} | {"model": info["model"], "system": info["system"]}
            for r in rows
        ])
        print("collected", target, len(rows))


def status() -> None:
    ow = get_ow()
    state = load_state()
    groups = {"finetune": state.get("finetune_jobs", {}), "merge": state.get("merge_jobs", {}), "eval": state.get("eval_jobs", {})}
    if "ia_rephrase_job" in state:
        groups["ia_corpus"] = {"rephrase": {"job_id": state["ia_rephrase_job"]}}
    for name, jobs in groups.items():
        for key, info in sorted(jobs.items()):
            print(f"{name:9s} {key:40s} {info['job_id']:36s} {ow.jobs.retrieve(info['job_id']).status}")


if __name__ == "__main__":
    {"submit": submit, "collect": collect, "status": status}[sys.argv[1]]()
