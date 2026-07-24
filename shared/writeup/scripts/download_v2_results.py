#!/usr/bin/env python3
"""Download eval_results.csv for every v2 eval job listed in data/summary_v2.json.

Saves to v2_csv/<task>/<model_slug>__<condition>.csv. Needs OPENWEIGHTS_API_KEY.
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openweights import OpenWeights

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "v2_csv"

ow = OpenWeights()


def slug(s):
    return re.sub(r"[^a-zA-Z0-9.-]+", "_", s)


def fetch(task, entry):
    dest = OUT / task / f"{slug(entry['model'])}__{slug(entry['condition'])}.csv"
    if dest.exists() and dest.stat().st_size > 1000:
        return f"skip {dest.name}"
    runs = ow.runs.list(job_id=entry["job"])
    if not runs:
        return f"FAIL {task} {entry['job']}: no runs"
    file_id = None
    for run in reversed(runs):
        for ev in ow.events.list(run_id=run.id):
            d = ev.data if hasattr(ev, "data") else ev
            data = d.get("data", d) if isinstance(d, dict) else d
            if isinstance(data, dict) and data.get("type") == "results_csv":
                file_id = data.get("file_id")
        if file_id:
            break
    if not file_id:
        return f"FAIL {task} {entry['job']}: no results_csv event"
    content = ow.files.content(file_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return f"ok {task}/{dest.name} ({len(content)//1024} KB)"


def main():
    summary = json.load(open(ROOT / "data" / "summary_v2.json"))
    jobs = [(task, e) for task, entries in summary.items() for e in entries]
    print(f"{len(jobs)} jobs")
    fails = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for res in ex.map(lambda te: fetch(*te), jobs):
            print(res, flush=True)
            if res.startswith("FAIL"):
                fails += 1
    print(f"DONE, {fails} failures")


if __name__ == "__main__":
    main()
