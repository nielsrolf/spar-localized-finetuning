"""Judge the base-model (vanilla) completions with gpt-6-luna using Niels's rejudge.py unchanged.

Completions: Jinho's selective-learning-benchmark release (vanilla_api_20260914), split per
task/model into completions/. Eval files: our sunday/scripts/eval/tasks. Scores go to scores/,
one file per setting; finished settings are skipped on rerun. Writes vanilla_luna_summary.csv.

    python run_vanilla_luna.py            # judge, then summarize
    python run_vanilla_luna.py --dry-run  # build requests and count calls, no judge calls
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path("/Users/sundayzhou/Development/spar-localized-finetuning")
NIELS = HERE.parent / "niels_repo" / "niels" / "experiment-rejudge-luna"
sys.path.insert(0, str(REPO / "sunday" / "scripts" / "eval"))  # our eval_constants / judge_utility
sys.path.insert(0, str(NIELS))
import rejudge  # noqa: E402  (Niels's script, unmodified)

rejudge.SCORES = HERE / "scores"


async def main(dry_run):
    plan = json.loads((HERE / "plan.json").read_text())
    runs = [dict(p, run_key=f"vanilla:{p['task']}/{p['model_family']}", condition="vanilla", seed=1,
                 eval_file="local:" + p["eval_file"], completions_file="local:" + p["completions_file"])
            for p in plan]
    if dry_run:
        n = sum(len(rejudge.build_requests(rejudge.load_file(None, r["eval_file"]),
                                           rejudge.load_file(None, r["completions_file"]))) for r in runs)
        print(f"{len(runs)} settings, {n} completions to judge")
        return
    config = {
        "judge_model": rejudge.JUDGE_MODEL,
        "judge_api_key": os.environ["LITELLM_API_KEY"],
        "judge_base_url": os.environ.get("LITELLM_BASE_URL", "https://litellm.nielsrolf.com"),
        "llm_judge_response_max_tokens": 2000,
    }
    runner = rejudge.LunaJudge(config, asyncio.Semaphore(rejudge.CONCURRENT_CALLS))
    sem = asyncio.Semaphore(rejudge.CONCURRENT_RUNS)
    await asyncio.gather(*(rejudge.judge_run(r, None, runner, sem) for r in runs))
    if runner._client is not None:
        await runner._client.close()

    rows = []
    for r in runs:
        p = rejudge.SCORES / (r["run_key"].replace(":", "__").replace("/", "__") + ".jsonl")
        if not p.exists():
            print("missing scores:", r["run_key"])
            continue
        s = rejudge.summarize(rejudge.read_jsonl_text(p.read_text()))
        rows.append({"task": r["task"], "model": r["model_family"], "condition": "base",
                     "cap_luna": s.get("capability_mean"), "ug_luna": s.get("ug_mean"),
                     "ug_coh_luna": s.get("ug_coh_mean"), "ug_coh_n_luna": s.get("ug_coh_n")})
    import pandas as pd
    pd.DataFrame(rows).to_csv(HERE / "vanilla_luna_summary.csv", index=False)
    print(f"wrote {HERE / 'vanilla_luna_summary.csv'} ({len(rows)} settings)")


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
