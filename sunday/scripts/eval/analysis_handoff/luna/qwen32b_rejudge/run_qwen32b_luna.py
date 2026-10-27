"""Re-judge the Qwen3-32B bf16 runs with gpt-6-luna using Niels's rejudge.py unchanged.

Completions: the OpenWeights completion jobs listed in RUNS (originally judged with
deepseek-v4-flash). Eval files: our sunday/scripts/eval/tasks. Scores go to scores/, one file
per run; finished runs are skipped on rerun. Writes qwen32b_luna_summary.csv.

Needs Niels's experiment-rejudge-luna folder; set NIELS_REJUDGE to its path.

    python run_qwen32b_luna.py --dry-run   # download files, build requests, no judge calls
    python run_qwen32b_luna.py             # judge, then summarize
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parents[2]  # sunday/scripts/eval
REPO = EVAL.parents[2]
sys.path.insert(0, str(EVAL))  # our eval_constants / judge_utility
sys.path.insert(0, os.environ["NIELS_REJUDGE"])
import rejudge  # noqa: E402  (Niels's script, unmodified)

rejudge.SCORES = HERE / "scores"
rejudge.FILES = HERE / "files"

TASK_DIRS = {"bad_medical_advice": "bad_medical_advice",
             "german_city_names": "weird_generaliztion-german_city_names",
             "target_only": "target_only_no_hallucination"}
# (task, condition, completion job, completions file)
RUNS = [
    ("bad_medical_advice", "baseline", "jobs-9923dadc1a96", "custom_job_file:file-0b043328ca3d"),
    ("bad_medical_advice", "first-third", "jobs-565d8df254e1", "custom_job_file:file-0ed9f03cf062"),
    ("bad_medical_advice", "second-third", "jobs-38b5d4434035", "custom_job_file:file-a8cbbc1d5952"),
    ("bad_medical_advice", "last-third", "jobs-0f7035938c31", "custom_job_file:file-9e9b59a811eb"),
    ("german_city_names", "baseline", "jobs-53f59a48509f", "custom_job_file:file-f462f23ee49d"),
    ("german_city_names", "first-third", "jobs-8ff51d610e80", "custom_job_file:file-81a2ac42cf06"),
    ("german_city_names", "second-third", "jobs-caf83d70478b", "custom_job_file:file-6e59c2442005"),
    ("german_city_names", "last-third", "jobs-6655f8e0e023", "custom_job_file:file-1ef8c39d1d15"),
    ("target_only", "baseline", "jobs-ec417e5ef47c", "custom_job_file:file-78992d1141db"),
    ("target_only", "first-third", "jobs-d8a7feb72f0c", "custom_job_file:file-3c5fb93cad4f"),
    ("target_only", "second-third", "jobs-44be1546229b", "custom_job_file:file-9eb755808615"),
    ("target_only", "last-third", "jobs-1ab2b04bd6b7", "custom_job_file:file-ef6a45e3aec3"),
]


def runs():
    return [dict(run_key=f"qwen32b:{job}", task=t, model_family="qwen3_32b", condition=c, seed=1,
                 eval_file="local:" + str(EVAL / "tasks" / TASK_DIRS[t] / "eval.jsonl"),
                 completions_file=f) for t, c, job, f in RUNS]


async def main(dry_run):
    from openweights import OpenWeights
    ow = OpenWeights()
    if dry_run:
        total = 0
        for r in runs():
            ev = rejudge.load_file(ow, r["eval_file"])
            comps = rejudge.load_file(ow, r["completions_file"])
            ids = {e["id"] for e in ev}
            missing = sum(c["eval_id"] not in ids for c in comps)
            calls = sum(len(e["grading"].get("judge_prompts") or {}) for e in ev for c in comps if c["eval_id"] == e["id"])
            total += calls
            print(f"{r['task']:20s} {r['condition']:13s} completions={len(comps):4d} missing_eval_ids={missing} judge_calls={calls}")
        print("total judge calls:", total)
        return
    config = {
        "judge_model": rejudge.JUDGE_MODEL,
        "judge_api_key": os.environ["LITELLM_API_KEY"],
        "judge_base_url": os.environ.get("LITELLM_BASE_URL", "https://litellm.nielsrolf.com"),
        "llm_judge_response_max_tokens": 2000,
    }
    runner = rejudge.LunaJudge(config, asyncio.Semaphore(rejudge.CONCURRENT_CALLS))
    sem = asyncio.Semaphore(rejudge.CONCURRENT_RUNS)
    await asyncio.gather(*(rejudge.judge_run(r, ow, runner, sem) for r in runs()))
    if runner._client is not None:
        await runner._client.close()

    import pandas as pd
    rows = []
    for r in runs():
        p = rejudge.SCORES / (r["run_key"].replace(":", "__").replace("/", "__") + ".jsonl")
        if not p.exists():
            print("missing scores:", r["run_key"])
            continue
        s = rejudge.summarize(rejudge.read_jsonl_text(p.read_text()))
        rows.append({"task": r["task"], "model": "qwen3_32b", "condition": r["condition"],
                     "completion_job": r["run_key"].split(":", 1)[1],
                     "cap_luna": s.get("capability_mean"), "ug_luna": s.get("ug_mean"),
                     "ug_coh_luna": s.get("ug_coh_mean"), "ug_coh_n_luna": s.get("ug_coh_n")})
    pd.DataFrame(rows).to_csv(HERE / "qwen32b_luna_summary.csv", index=False)
    print(f"wrote {HERE / 'qwen32b_luna_summary.csv'} ({len(rows)} runs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    from dotenv import load_dotenv
    import logging
    load_dotenv(REPO / "sunday" / ".env")
    for name in ("httpx", "openai", "openai._base_client"):
        logging.getLogger(name).setLevel(logging.WARNING)
    asyncio.run(main(args.dry_run))
