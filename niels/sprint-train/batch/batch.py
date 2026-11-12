"""Big batch: every planned technique x seed on the tasks with `status: ready` in tasks.yaml.

  python niels/sprint-train/batch/batch.py plan   [--seeds 1]            # what is launchable / blocked on which data
  python niels/sprint-train/batch/batch.py tick   [--seeds 1] [--tasks ...] [--techniques ...]
  python niels/sprint-train/batch/batch.py status [--seeds 1]

`tick` assembles every run whose inputs exist, submits / advances its stages, samples the eval set of
finished runs (derisk.py sample, with EVAL_HOME = this folder) and judges finished samples (derisk.py collect).
A run is blocked until its inputs exist:
  lf_*       the SFT run of the same task and seed has completed (loss targets)
  ia, graft  the task's undesired-trait data (tasks.yaml undesired_data)
  *_sd_<src> the chosen model's self-distilled set, niels/sprint-train/sd/<model>/<src>.jsonl

Runs: batch/runs/<model>/<task>/<technique>/seed<k>/. Evals: batch/evals/<task>__<model>__<technique>-s<k>/.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPRINT = HERE.parent
REPO = SPRINT.parents[1]
sys.path.insert(0, str(SPRINT))

import assemble as A  # noqa: E402
import techniques as T  # noqa: E402

RUNS = HERE / "runs"
REGISTRY = A.load_registry()
PY = sys.executable
MAX_RETRIES = 2
UV = ["uv", "run", "--quiet", "--with", "pyyaml", "--with", "openweights==0.14.0", "--with", "transformers", "--with", "torch",
      "--with", "python-dotenv", "python"]


def ready_tasks(registry: dict) -> list[dict]:
    return [A.task_entry(registry, name) for name, t in registry["tasks"].items() if t.get("status") == "ready"]


def run_dir(task: dict, tech: str, seed: int) -> Path:
    return RUNS / A.short(task["model"]) / task["name"] / tech / f"seed{seed}"


def sd_path(task: dict, src: str) -> Path:
    return SPRINT / "sd" / A.short(task["model"]) / f"{src}.jsonl"


def blocked_on(task: dict, tech: str, seed: int, sft_done: bool) -> str | None:
    if tech.startswith("lf_") and not sft_done:
        return "sft run"
    if tech == "graft" and not REGISTRY.get("graft_learning_rate"):
        return "GRAFT learning-rate check"
    if tech in ("ia", "graft"):
        spec = task.get("undesired_data")
        if not spec or not (spec.startswith("task_rows_with_label:") or (REPO / spec).exists()):
            return "undesired-trait data"
    if "_sd_" in tech and not sd_path(task, tech.split("_sd_")[1]).exists():
        return f"self-distilled {tech.split('_sd_')[1]} set"
    return None


def eval_label(task: dict, tech: str, seed: int) -> str:
    name = task["name"]
    short = name if name.startswith("ia-") else name.split("-", 1)[1]
    return f"{short}__{task['model'].split('/')[-1]}__{tech}-s{seed}"


def assemble(task: dict, tech: str, seed: int) -> None:
    args = [PY, str(SPRINT / "assemble.py"), "--task", task["name"], "--technique", tech, "--seed-label", str(seed),
            "--private", "--out-root", str(RUNS)]
    if tech == "graft":
        args += ["--graft-lr", str(REGISTRY["graft_learning_rate"])]
    if tech.startswith("lf_"):
        args += ["--sft-run", str(run_dir(task, "sft", seed))]
    if "_sd_" in tech:
        src = tech.split("_sd_")[1]
        args += [f"--sd-{src}", str(sd_path(task, src))]
    subprocess.run(args, check=True, cwd=REPO, capture_output=True)


def derisk(*args) -> str:
    out = subprocess.run([PY, str(SPRINT / "derisking/derisk.py"), *args], capture_output=True, text=True, cwd=REPO,
                         env={**os.environ, "EVAL_HOME": str(HERE)})
    lines = [l for l in (out.stdout + out.stderr).splitlines() if l.strip() and "[JudgeRunner]" not in l
             and (not l.split(" ")[0].endswith(".py") or "Error" in l)]
    return "\n".join(lines)


def selected(a, registry):
    for task in ready_tasks(registry):
        if a.tasks and not any(t in task["name"] for t in a.tasks):
            continue
        for seed in a.seeds:
            for tech in task["techniques"]:
                if a.techniques and tech not in a.techniques:
                    continue
                yield task, tech, seed


def cmd_plan(a) -> None:
    registry = A.load_registry()
    for task, tech, seed in selected(a, registry):
        rd = run_dir(task, tech, seed)
        state = "assembled" if (rd / "run.json").exists() else (blocked_on(task, tech, seed, False) or "launchable")
        print(f"{task['name']:52s} {tech:20s} s{seed}  {state}")


def cmd_tick(a) -> None:
    registry = A.load_registry()
    # Collect self-distilled sets whose sampling job has finished (unblocks the *_sd_* runs).
    for job in (SPRINT / "sd").glob("*/*.job.json"):
        out = job.with_name(job.name.replace(".job.json", ".jsonl"))
        if not out.exists():
            r = subprocess.run([PY, str(SPRINT / "sample_self_distillation.py"), "collect", "--out", str(out)],
                               capture_output=True, text=True, cwd=REPO)
            print("sd:", (r.stdout + r.stderr).strip().splitlines()[-1][:200], flush=True)
    evaluated = json.loads((HERE / "state.json").read_text()) if (HERE / "state.json").exists() else {}
    last_path = HERE / "last_status.json"  # run dir -> launch.py's status line at the previous tick
    last = json.loads(last_path.read_text()) if last_path.exists() else {}
    counts: dict[str, int] = {}
    runs = {}
    for task, tech, seed in selected(a, registry):
        rd = run_dir(task, tech, seed)
        if not (rd / "run.json").exists():
            block = blocked_on(task, tech, seed, last.get(str(run_dir(task, "sft", seed))) == "completed")
            if block:
                counts[f"blocked: {block}"] = counts.get(f"blocked: {block}", 0) + 1
                continue
            assemble(task, tech, seed)
        runs[str(rd)] = (task, tech, seed)
    # launch.py imports the training pipeline (torch, transformers), which the judge venv does not have.
    out = subprocess.run(UV + [str(SPRINT / "launch.py"), "advance", *runs], capture_output=True, text=True, cwd=REPO)
    lines = dict(l.split(" -> ", 1) for l in out.stdout.splitlines() if " -> " in l and l.split(" -> ")[0] in runs)
    if len(lines) < len(runs):
        print("launch.py:", out.stderr[-1500:], flush=True)
    for rd_str, line in lines.items():
        task, tech, seed = runs[rd_str]
        rd = Path(rd_str)
        last[rd_str] = line
        if "FAILED" in line:
            st = json.loads((rd / "run.json").read_text())
            if st.get("retries", 0) < MAX_RETRIES:
                stage = line.split(" ")[0]
                st["failed_jobs"] = st.get("failed_jobs", []) + [st["jobs"].pop(stage)["job_id"]]
                st["retries"] = st.get("retries", 0) + 1
                (rd / "run.json").write_text(json.dumps(st, indent=1))
                line += f" -> retry {st['retries']}"
        counts[line.split(" (")[0]] = counts.get(line.split(" (")[0], 0) + 1
        if "submitted" in line or "FAILED" in line:
            print(f"{rd.relative_to(RUNS)} -> {line}", flush=True)
        if tech == "ia" and not line.startswith("stage1"):
            # Check that the stage-1 adapter shows the undesired trait: the task's eval on base + IA.
            label = eval_label(task, "ia_stage1", seed)
            if label not in evaluated:
                st = json.loads((rd / "run.json").read_text())
                print(derisk("sample", "--task", task["name"], "--model", st["final_model_id"] + "-ia-stage1",
                             "--base", task["model"], "--label", label).splitlines()[-1], flush=True)
        if line == "completed":
            label = eval_label(task, tech, seed)
            if label not in evaluated:
                st = json.loads((rd / "run.json").read_text())
                print(derisk("sample", "--task", task["name"], "--model", st["final_model_id"], "--base", task["model"],
                             "--label", label).splitlines()[-1], flush=True)
    last_path.write_text(json.dumps(last, indent=1))
    print(derisk("collect"), flush=True)
    print("runs:", json.dumps(counts), flush=True)


def cmd_status(a) -> None:
    registry = A.load_registry()
    for task, tech, seed in selected(a, registry):
        rd = run_dir(task, tech, seed)
        summary = HERE / "evals" / eval_label(task, tech, seed) / "summary.json"
        if summary.exists():
            s = json.loads(summary.read_text())
            cap, ug = s["capability"], s["undesired_generalization"]
            print(f"{task['name']:52s} {tech:20s} s{seed}  desired {cap.get('mean', float('nan')):5.1f} (n={cap.get('n')})"
                  f"  undesired {ug.get('mean', float('nan')):5.1f} (n={ug.get('n')})")
        else:
            print(f"{task['name']:52s} {tech:20s} s{seed}  {'assembled' if (rd / 'run.json').exists() else '-'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["plan", "tick", "status"])
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--tasks", nargs="+", help="substring filters")
    p.add_argument("--techniques", nargs="+")
    a = p.parse_args()
    {"plan": cmd_plan, "tick": cmd_tick, "status": cmd_status}[a.command](a)


if __name__ == "__main__":
    main()
