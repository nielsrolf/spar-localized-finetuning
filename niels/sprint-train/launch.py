"""Submit and advance run directories built by assemble.py.

Each run has ordered stages (finetune / graft / merge). `advance` submits the next stage once the
previous one has completed, so IA's three stages chain automatically when called repeatedly.

Usage (from the repo root, with OPENWEIGHTS_API_KEY in .env):
    python niels/sprint-train/launch.py advance niels/sprint-train/runs/Qwen3-8B/*/*/seed1
    python niels/sprint-train/launch.py status  niels/sprint-train/runs/Qwen3-8B/*/*/seed1
    python niels/sprint-train/launch.py advance --dry-run <run dirs>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PIPELINE = REPO_ROOT / "external/selective-learning-benchmark/scripts/finetune"
sys.path.insert(0, str(PIPELINE))
load_dotenv(REPO_ROOT / ".env")


def _patch_job_dataclass() -> None:
    """openweights<=0.13.5's Job dataclass crashes on new jobs-table columns (e.g. `submitted_by`);
    drop unknown fields (same patch as niels/experiment-ip-variants/common.py)."""
    import dataclasses

    from openweights.client import jobs as jobs_mod

    if getattr(jobs_mod.Job, "_sprint_patched", False):
        return
    known = {f.name for f in dataclasses.fields(jobs_mod.Job)}
    orig_init = jobs_mod.Job.__init__

    def __init__(self, **kwargs):
        orig_init(self, **{k: v for k, v in kwargs.items() if k in known})

    jobs_mod.Job.__init__ = __init__
    jobs_mod.Job._sprint_patched = True


def get_ow():
    from openweights import OpenWeights

    _patch_job_dataclass()

    import finetune_job  # noqa: F401  registers ow.config_finetune
    import graft_job  # noqa: F401  registers ow.config_graft
    import merge_adapter_job  # noqa: F401  registers ow.merge_adapter
    return OpenWeights()


def final_losses(ow, job_id: str) -> tuple[float, float]:
    """Final train loss and final validation loss logged by the finetune worker."""
    train = val = None
    for event in ow.events.list(job_id=job_id):
        data = event.get("data") or {}
        if str(data.get("text", "")).startswith("Training complete. Final loss") and "loss" in data:
            train = float(data["loss"])
        if str(data.get("text", "")).startswith("Final eval loss") and "eval_loss" in data:
            val = float(data["eval_loss"])
    if train is None or val is None:
        raise RuntimeError(f"job {job_id} has no final train/eval loss events")
    return train, val


def resolve_early_stop_targets(ow, cfg: dict) -> dict:
    src = cfg.pop("early_stop_targets_from", None)
    if not src:
        return cfg
    sft = json.loads((Path(src) / "run.json").read_text())
    job = sft.get("jobs", {}).get("stage1")
    if not job or ow.jobs.retrieve(job["job_id"]).status != "completed":
        raise RuntimeError(f"SFT baseline {src} has not completed; cannot set loss-match targets")
    cfg["early_stop_target_train_loss"], cfg["early_stop_target_validation_loss"] = final_losses(ow, job["job_id"])
    return cfg


def submit_stage(ow, run: Path, stage: dict, dry_run: bool) -> str | None:
    if stage["kind"] == "merge":
        params = {k: stage[k] for k in ("base_model", "adapter", "output_model_id")}
        params["private"] = bool(stage.get("private", False))
        if dry_run:
            print("  merge", params)
            return None
        return ow.merge_adapter.create(**params).id

    cfg = yaml.safe_load((run / stage["config"]).read_text())
    if stage["kind"] == "finetune":
        from finetune_config_utility import load_submit_config
        from submit_finetune import submit_job

        if not dry_run:
            cfg = resolve_early_stop_targets(ow, cfg)
        elif cfg.pop("early_stop_targets_from", None):
            cfg.update(early_stop_target_train_loss=0.0, early_stop_target_validation_loss=0.0)
        tmp = run / f".{stage['name']}.resolved.yaml"
        tmp.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        job = submit_job(load_submit_config(tmp), dry_run=dry_run)
        return job.id if job else None

    if stage["kind"] == "graft":
        from graft_job import GraftParams
        from submit_finetune import count_jsonl_rows

        multi_turn = bool(cfg.get("multi_turn", False))
        count_jsonl_rows(cfg["training_path"], multi_turn)
        count_jsonl_rows(cfg["validation_path"], multi_turn)
        params = {k: v for k, v in cfg.items() if k not in ("training_path", "validation_path")}
        GraftParams(**params, training_file="dry-run", validation_file="dry-run")
        if dry_run:
            print("  graft params ok")
            return None
        params["training_file"] = ow.files.upload(path=cfg["training_path"], purpose="conversations")["id"]
        params["validation_file"] = ow.files.upload(path=cfg["validation_path"], purpose="conversations")["id"]
        return ow.config_graft.create(**params).id
    raise ValueError(stage["kind"])


def advance(ow, run: Path, dry_run: bool) -> str:
    state = json.loads((run / "run.json").read_text())
    for stage in state["stages"]:
        job = state["jobs"].get(stage["name"])
        if job:
            status = ow.jobs.retrieve(job["job_id"]).status if ow else "?"
            if status == "completed":
                continue
            if status == "failed":
                return f"{stage['name']} FAILED ({job['job_id']})"
            return f"{stage['name']} {status} ({job['job_id']})"
        job_id = submit_stage(ow, run, stage, dry_run)
        if dry_run:
            return f"{stage['name']} dry-run ok"
        state["jobs"][stage["name"]] = {"job_id": job_id}
        (run / "run.json").write_text(json.dumps(state, indent=1))
        return f"{stage['name']} submitted ({job_id})"
    return "completed"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["advance", "status"])
    p.add_argument("runs", nargs="+")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    ow = get_ow() if not (a.dry_run and a.command == "advance") else None
    for run in map(Path, a.runs):
        if not (run / "run.json").exists():
            continue
        if a.command == "advance":
            print(run, "->", advance(ow, run, a.dry_run))
        else:
            state = json.loads((run / "run.json").read_text())
            jobs = {k: ow.jobs.retrieve(v["job_id"]).status for k, v in state["jobs"].items()}
            print(run, jobs or "not submitted")


if __name__ == "__main__":
    main()
