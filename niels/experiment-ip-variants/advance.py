"""One idempotent pass over the whole pipeline; safe to run repeatedly (e.g. in a loop).

  IA corpus job done  -> filter + write data/ia_corpus
  corpus written      -> submit ia_stage1 (IA LoRA, merged into base)
  ia_stage1 done      -> submit ia_stage2 (task LoRA on base+IA)
  ia_stage2 done      -> merge task adapter onto the original base
  any model ready     -> submit eval completions; collect finished completions
  LiteLLM key present -> judge + summarize

Usage: .venv/bin/python advance.py [--loop SECONDS]
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

from common import DATA_DIR, MODELS, REPO, ROOT, get_ow, load_state

PY = str(ROOT / ".venv" / "bin" / "python")


def run(*args: str) -> None:
    subprocess.run([PY, *args], cwd=ROOT, check=False)


def status_of(ow, job_id: str) -> str:
    return ow.jobs.retrieve(job_id).status


def all_done(ow, state) -> bool:
    groups = [state.get("finetune_jobs", {}), state.get("merge_jobs", {}), state.get("eval_jobs", {})]
    expected_ft = 3 * 9  # ip_evil_emph, ip_negate, sip, ia_stage1, ia_stage2, sip2_f{10,25,50}, sft_hhh
    if len(state.get("finetune_jobs", {})) < expected_ft or len(state.get("merge_jobs", {})) < len(MODELS):
        return False
    return all(status_of(ow, j["job_id"]) in ("completed", "failed", "canceled") for g in groups for j in g.values())


def step() -> bool:
    ow = get_ow()
    state = load_state()
    ft = state.get("finetune_jobs", {})

    if not (DATA_DIR / "ia_corpus" / "train.jsonl").exists():
        if status_of(ow, state["ia_rephrase_job"]) == "completed":
            run("ia_corpus.py", "collect")
    if (DATA_DIR / "ia_corpus" / "train.jsonl").exists() and not any(k.startswith("ia_stage1/") for k in ft):
        run("submit_finetunes.py", "ia_stage1")

    state = load_state()
    ft = state.get("finetune_jobs", {})
    ready = [s for s in MODELS if f"ia_stage1/{s}" in ft and status_of(ow, ft[f"ia_stage1/{s}"]["job_id"]) == "completed"]
    todo = [s for s in ready if f"ia_stage2/{s}" not in ft]
    if todo:
        run("submit_finetunes.py", "ia_stage2", "--models", *todo)

    state = load_state()
    ft = state.get("finetune_jobs", {})
    stage2_done = [s for s in MODELS if f"ia_stage2/{s}" in ft and status_of(ow, ft[f"ia_stage2/{s}"]["job_id"]) == "completed"]
    if any(s not in state.get("merge_jobs", {}) for s in stage2_done):
        run("merge_job.py")

    run("eval_completions.py", "submit")
    run("eval_completions.py", "collect")
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    if os.environ.get("LITELLM_API_KEY"):
        run("judge.py")

    return all_done(ow, load_state())


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--loop":
        while not step():
            time.sleep(int(sys.argv[2]))
        print("PIPELINE COMPLETE")
    else:
        step()
