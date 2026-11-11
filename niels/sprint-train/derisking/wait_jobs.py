"""Block until the given OpenWeights jobs are all terminal (or until --minutes), printing status changes.
Prints the tail of the run log for failed jobs.

  python niels/sprint-train/derisking/wait_jobs.py JOB_ID [JOB_ID ...] [--minutes 120]
  python niels/sprint-train/derisking/wait_jobs.py --runs 'niels/sprint-train/derisking/runs/*/*/sft/seed1'
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "niels/experiment-ip-variants"))
from common import get_ow  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("jobs", nargs="*")
p.add_argument("--runs", help="glob of run dirs; waits for their latest stage jobs")
p.add_argument("--minutes", type=float, default=180)
p.add_argument("--any", action="store_true", help="return as soon as any job is terminal")
a = p.parse_args()
jobs = list(a.jobs)
if a.runs:
    for rd in glob.glob(str(REPO / a.runs)):
        st = json.loads((Path(rd) / "run.json").read_text())
        jobs += [j["job_id"] for j in st.get("jobs", {}).values()]
ow = get_ow()
last = {}
deadline = time.time() + 60 * a.minutes
while True:
    pending = 0
    for j in jobs:
        s = ow.jobs.retrieve(j).status
        if last.get(j) != s:
            print(time.strftime("%H:%M"), j, s, flush=True)
            if s == "failed":
                runs = ow._supabase.table("runs").select("log_file").eq("job_id", j).execute().data
                if runs and runs[-1].get("log_file"):
                    log = ow.files.content(runs[-1]["log_file"]).decode(errors="replace").splitlines()
                    print("\n".join(l for l in log[-15:] if l.strip()), flush=True)
            last[j] = s
        pending += s in ("pending", "in_progress")
    if pending == 0 or time.time() > deadline or (a.any and pending < len(jobs)):
        break
    time.sleep(60)
print("done:", {s: sum(v == s for v in last.values()) for s in set(last.values())}, flush=True)
