#!/bin/bash
# Submit the evals of the GRAFT investigation as their models become available, then wait until all are judged
# (collect_loop.sh does the judging). Exits when every label has a summary, or after 4 hours.
cd "$(dirname "$0")/../../.."
set -a; source .env; set +a
export EVAL_HOME=niels/sprint-train/batch
end=$(( $(date +%s) + 4 * 3600 ))
while [ "$(date +%s)" -lt "$end" ]; do
  ~/.venv/bin/python - <<'PY' 2>/dev/null | grep -E "submitted|waiting|ALL DONE|failed"
import json, os, subprocess, sys
sys.path.insert(0, "niels/experiment-ip-variants")
from common import get_ow
ow = get_ow()
B = "niels/sprint-train/batch"
state = json.load(open(f"{B}/state.json"))
T = {"bad-medical-advice": ("emergent_misalignment-bad_medical_advice", "bad_medical_advice"), "risky-financial-advice": ("emergent_misalignment-risky_financial_advice", "risky_financial_advice"),
     "french-vs-all-caps": ("ia-french_vs_all_caps",) * 2, "french-vs-em": ("ia-french_vs_em",) * 2, "caps-vs-hate-speech": ("ia-caps_vs_hate_speech",) * 2}
todo = [(j, T[t][0], f"{T[t][1]}__Qwen3.5-4B__graft_forget-s1", f"longtermrisk/Qwen3-5-4B-{t}-graft-s1-mlp-forget")
        for t, j in json.load(open("niels/sprint-train/graft_check/forget_only_jobs.json")).items()]
lr = "longtermrisk/Qwen3-5-4B-french-vs-em-graft-mlp-lr2e-5-s1"
todo += [("graftjob-7ccc5e4d5c0d", "ia-french_vs_em", "ia-french_vs_em__Qwen3.5-4B__graft-mlp-lr2e-5-s1", lr),
         ("graftjob-7ccc5e4d5c0d", "ia-french_vs_em", "ia-french_vs_em__Qwen3.5-4B__graft-mlp-lr2e-5-both-s1", lr + "-both")]
labels = [t[2] for t in todo] + [f"ia-french_vs_em__Qwen3.5-4B__graft-mlp-{k}-s1" for k in ("lr5e-5", "lr5e-5-both", "lr2e-4-both")]
for job, task, label, model in todo:
    if label in state: continue
    st = ow.jobs.retrieve(job).status
    if st == "completed":
        out = subprocess.run([sys.executable, "niels/sprint-train/derisking/derisk.py", "sample", "--task", task, "--model", model, "--base", "Qwen/Qwen3.5-4B", "--label", label], capture_output=True, text=True)
        print((out.stdout + out.stderr).strip().splitlines()[-1][:200])
    elif st in ("failed", "canceled"): print(label, "job", job, st, "failed")
missing = [l for l in labels if not os.path.exists(f"{B}/evals/{l}/summary.json")]
print("waiting for", len(missing), missing[:3]) if missing else print("ALL DONE")
PY
  if ! ls niels/sprint-train/batch/evals/*graft_forget-s1/summary.json >/dev/null 2>&1; then :; fi
  n=$(ls niels/sprint-train/batch/evals/*__graft_forget-s1/summary.json niels/sprint-train/batch/evals/ia-french_vs_em__Qwen3.5-4B__graft-mlp-lr2e-5-s1/summary.json niels/sprint-train/batch/evals/ia-french_vs_em__Qwen3.5-4B__graft-mlp-lr2e-5-both-s1/summary.json niels/sprint-train/batch/evals/ia-french_vs_em__Qwen3.5-4B__graft-mlp-lr5e-5-s1/summary.json niels/sprint-train/batch/evals/ia-french_vs_em__Qwen3.5-4B__graft-mlp-lr5e-5-both-s1/summary.json niels/sprint-train/batch/evals/ia-french_vs_em__Qwen3.5-4B__graft-mlp-lr2e-4-both-s1/summary.json 2>/dev/null | wc -l)
  [ "$n" -ge 10 ] && { echo "all 10 evals judged"; exit 0; }
  sleep 180
done
echo "timed out with $n of 10 judged"
