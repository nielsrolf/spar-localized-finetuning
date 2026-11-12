"""Re-score the desired trait of the bad medical and risky financial evals with the task's current judge
(sprint-tasks/em_judge.py: revised prompt, low reasoning). Rewrites eval_results.csv and summary.json. Cached."""
import asyncio
import csv
import json
import sys

sys.argv = sys.argv[:1]
import derisk  # noqa: E402
from derisk import *  # noqa: E402,F403
from judge_utility import JudgeRunner, parse_judge_response_score  # noqa: E402

sys.path.insert(0, str(derisk.REPO / "niels/sprint-tasks"))
from em_judge import NARROW_JUDGES  # noqa: E402

csv.field_size_limit(10**9)
TASKS = {"bad_medical_advice": ("emergent_misalignment-bad_medical_advice", "bad_medical"),
         "risky_financial_advice": ("emergent_misalignment-risky_financial_advice", "risky_financial")}


async def one(runner, spec, r):
    prompt = spec["prompt"].replace("{question}", r["question"]).replace("{answer}", r["completion"])
    for attempt in range(5):
        try:
            res = parse_judge_response_score(r["score_name"], await runner.get_llm_judge_response_text(prompt, spec["reasoning_effort"]))
            break
        except Exception as e:  # noqa: BLE001
            res = None
            await asyncio.sleep(3 * (attempt + 1))
    if res is None:
        r["score"], r["score_label"], r["score_source_text"] = "", "ERROR", "judge call failed"
    else:
        r["score"], r["score_label"], r["score_source_text"] = ("" if res.score is None else res.score), res.score_label, res.score_source_text


async def one_eval(runner, path, name):
    rows = list(csv.DictReader(open(path)))
    await asyncio.gather(*[one(runner, NARROW_JUDGES[name], r) for r in rows if r["score_name"] == name])
    errors = sum(r["score_name"] == name and r["score"] == "" for r in rows)
    if errors > 20:
        print(path.parent.name, "too many errors, not written:", errors, flush=True)
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    typed = [dict(r, score=float(r["score"]) if r["score"] not in ("", "None") else "") for r in rows]
    sp = path.parent / "summary.json"
    summary = json.loads(sp.read_text())
    old = summary["capability"]["mean"]
    summary["capability"] = derisk.axis_summary([r for r in typed if r["axis"] == "capability"])
    summary["narrow_judge"] = "revised"
    sp.write_text(json.dumps(summary))
    print(f"{path.parent.name:50s} {name} {old:5.1f} -> {summary['capability']['mean']:5.1f}  errors {errors}", flush=True)


async def main():
    jobs = []
    for short, (task, name) in TASKS.items():
        cfg = derisk.task_config(task, "x")
        runner = JudgeRunner(cfg, asyncio.Semaphore(200))
        for path in sorted(derisk.EVALS.glob(f"{short}__*/eval_results.csv")):
            if json.loads((path.parent / "summary.json").read_text()).get("narrow_judge") != "revised":
                jobs.append(one_eval(runner, path, name))
    await asyncio.gather(*jobs)

asyncio.run(main())
