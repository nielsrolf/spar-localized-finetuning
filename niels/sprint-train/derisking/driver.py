"""Derisking driver: base evals, SFT runs and SFT evals for every task x candidate model.

  python niels/sprint-train/derisking/driver.py base-evals [--models ...] [--tasks ...]
  python niels/sprint-train/derisking/driver.py sft        [--models ...] [--tasks ...]   # assemble + launch
  python niels/sprint-train/derisking/driver.py tick                                   # advance SFT, submit evals of finished SFTs, judge
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNS = HERE / "runs"
PY = sys.executable
UV = ["uv", "run", "--quiet", "--with", "pyyaml", "--with", "openweights", "--with", "transformers", "--with", "torch", "python"]

TASKS = [
    "emergent_misalignment-bad_medical_advice",
    "emergent_misalignment-risky_financial_advice",
    "emergent_misalignment-school_of_reward_hacks",
    "weird_generaliztion-german_city_names",
    "weird_generaliztion-old_bird_names",
    "synthetic_document-good_vs_bad_mixed_multifact",
    "synthetic_document-target_only_no_hallucination",
    "ia-french_vs_all_caps",
    "ia-french_vs_em",
    "ia-french_vs_cipher",
    "ia-caps_vs_hate_speech",
]
STAGE1 = ["Qwen/Qwen3.5-4B", "google/gemma-4-E4B-it", "Qwen/Qwen3.5-9B"]


def task_short(task: str) -> str:
    return task if task.startswith("ia-") else task.split("-", 1)[1]


def model_short(model: str) -> str:
    return model.split("/")[-1]


def label(task: str, model: str, kind: str) -> str:
    return f"{task_short(task)}__{model_short(model)}__{kind}"


def run_dir(task: str, model: str) -> Path:
    return RUNS / model.split("/")[-1].replace("_", "-").replace(".", "-") / task / "sft" / "seed1"


def derisk(*args) -> str:
    out = subprocess.run([PY, str(HERE / "derisk.py"), *args], capture_output=True, text=True, cwd=REPO)
    return "\n".join(l for l in (out.stdout + out.stderr).splitlines()
                      if not l.split(" ")[0].endswith(".py") or "Error" in l)


def cmd_base_evals(a) -> None:
    for t in a.tasks:
        for m in a.models:
            print(derisk("sample", "--task", t, "--model", m, "--label", label(t, m, "base")).splitlines()[-1], flush=True)


def cmd_sft(a) -> None:
    for t in a.tasks:
        for m in a.models:
            if not (run_dir(t, m) / "run.json").exists():
                subprocess.run(UV[:5] + ["python", str(REPO / "niels/sprint-train/assemble.py"), "--task", t, "--technique",
                                         "sft", "--model", m, "--seed-label", "1", "--private", "--name-suffix=-derisk",
                                         "--out-root", str(RUNS)], check=True, cwd=REPO, capture_output=True)
            out = subprocess.run(UV + [str(REPO / "niels/sprint-train/launch.py"), "advance", str(run_dir(t, m))],
                                 capture_output=True, text=True, cwd=REPO)
            print(out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr[-500:], flush=True)


def cmd_tick(a) -> None:
    # runs-lr/<model>/<task>/sft-lr<lr>/seed1: one higher-learning-rate retry, evaluated as kind "sft-lr<lr>".
    for rd in sorted(RUNS.glob("*/*/sft/seed1")) + sorted((HERE / "runs-lr").glob("*/*/sft-lr*/seed1")):
        st = json.loads((rd / "run.json").read_text())
        out = subprocess.run(UV + [str(REPO / "niels/sprint-train/launch.py"), "advance", str(rd)],
                             capture_output=True, text=True, cwd=REPO)
        line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr[-300:]
        print(line, flush=True)
        if "FAILED" in line and st.get("retries", 0) < 2:
            # Transient failures (CUDA illegal memory access, post-training NaN eval before the fix): resubmit
            # with the current worker code; launch.py submits again once the stage has no job recorded.
            st = json.loads((rd / "run.json").read_text())
            st["failed_jobs"] = st.get("failed_jobs", []) + [st["jobs"].pop("stage1")["job_id"]]
            st["retries"] = st.get("retries", 0) + 1
            (rd / "run.json").write_text(json.dumps(st, indent=1))
            print(f"{rd}: retry {st['retries']}", flush=True)
        if line.endswith("-> completed") and not st["task"].startswith("reward_hacking-terminal_wrench"):
            lbl = label(st["task"], st["model"], rd.parent.name)
            print(derisk("sample", "--task", st["task"], "--model", st["final_model_id"], "--base", st["model"],
                         "--label", lbl).splitlines()[-1], flush=True)
    print(derisk("collect"), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("base-evals", "sft"):
        s = sub.add_parser(name)
        s.add_argument("--models", nargs="+", default=STAGE1)
        s.add_argument("--tasks", nargs="+", default=TASKS)
    sub.add_parser("tick")
    a = p.parse_args()
    {"base-evals": cmd_base_evals, "sft": cmd_sft, "tick": cmd_tick}[a.cmd](a)


if __name__ == "__main__":
    main()
