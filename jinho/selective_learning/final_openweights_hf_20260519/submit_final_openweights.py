#!/usr/bin/env python3
"""Dry-run or submit the final HF benchmark OpenWeights sweep."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_METHODS = [
    "sft",
    "kl_regularization",
    "inoculation_prompting",
    "representation_consistency",
    "replay_distillation",
]
METHOD_ALIASES = {
    "plain": "sft",
    "method_b": "kl_regularization",
    "method_ip": "inoculation_prompting",
    "method_g": "representation_consistency",
    "method_j": "replay_distillation",
}
STATE_METHOD_ALIASES = {
    "plain": "sft",
    "method_b": "kl_regularization",
    "method_ip": "inoculation_prompting",
    "method_g": "representation_consistency",
    "method_g_representation_consistency": "representation_consistency",
    "method_j": "replay_distillation",
    "method_j_replay_distill": "replay_distillation",
    "method_j_replay_distill_sft": "replay_distillation",
}


@dataclass(frozen=True)
class CommandSpec:
    task_slug: str
    model_key: str
    method: str
    config: Path
    state_file: Path
    argv: tuple[str, ...]
    seeds: tuple[int, ...]
    estimated_jobs: int

    @property
    def label(self) -> str:
        return f"{self.task_slug}/{self.model_key}/{self.method}"


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def has_excluded_prefix(value: str, prefixes: list[str]) -> bool:
    return any(value.startswith(prefix) for prefix in prefixes)


def canonical_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def canonical_state_method(method: str) -> str:
    return STATE_METHOD_ALIASES.get(method, METHOD_ALIASES.get(method, method))


def load_manifest() -> dict[str, Any]:
    path = RUN_ROOT / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run prepare_final_openweights.py before submitting."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def has_existing_nonfailed_seed(state_file: Path, method: str, seed: int) -> bool:
    if not state_file.exists():
        return False
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    for row in state.get("sft_jobs", []):
        if not isinstance(row, dict):
            continue
        if int(row.get("seed", -1)) != seed:
            continue
        row_method = canonical_state_method(str(row.get("method", "")))
        method_match = row_method == method
        if method_match and row.get("status") not in {"failed", "canceled", "cancelled"}:
            return True
    return False


def missing_seeds(state_file: Path, method: str, seeds: list[int], resubmit_existing: bool) -> list[int]:
    if resubmit_existing:
        return seeds
    return [seed for seed in seeds if not has_existing_nonfailed_seed(state_file, method, seed)]


def config_path(task_slug: str, model_key: str, method: str) -> Path:
    return RUN_ROOT / "configs" / task_slug / model_key / f"{method}.yaml"


def state_path(task_slug: str, model_key: str, method: str) -> Path:
    return RUN_ROOT / "states" / task_slug / model_key / f"{method}_state.json"


def build_command(
    task_slug: str,
    model_key: str,
    method: str,
    seeds: list[int],
    resubmit_existing: bool,
) -> CommandSpec | None:
    method = canonical_method(method)
    cfg = config_path(task_slug, model_key, method)
    state = state_path(task_slug, model_key, method)
    if not cfg.exists():
        raise FileNotFoundError(f"Missing config: {cfg}")

    todo = missing_seeds(state, method, seeds, resubmit_existing)
    if not todo:
        return None

    if method in {"sft", "kl_regularization", "inoculation_prompting"}:
        argv = (
            "uv",
            "run",
            "python",
            "-m",
            "selective_learning.core.submit",
            "sft",
            "--config",
            rel(cfg),
        )
    elif method == "representation_consistency":
        argv = (
            "uv",
            "run",
            "python",
            "selective_learning/method_search/submit_method_g_representation.py",
            "--config",
            rel(cfg),
            "--state-file",
            rel(state),
            "--seeds",
            ",".join(map(str, todo)),
            "--betas",
            "0.1",
            "--rep-layer-count",
            "4",
            "--submit",
        )
    elif method == "replay_distillation":
        argv = (
            "uv",
            "run",
            "python",
            "selective_learning/method_search/submit_method_j_replay_distill.py",
            "--config",
            rel(cfg),
            "--state-file",
            rel(state),
            "--seeds",
            ",".join(map(str, todo)),
            "--replay-alphas",
            "0.3",
            "--distill-betas",
            "0.1",
            "--candidate-method",
            "replay_distillation",
            "--submit",
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

    return CommandSpec(task_slug, model_key, method, cfg, state, argv, tuple(todo), len(todo))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def build_commands(args: argparse.Namespace) -> list[CommandSpec]:
    manifest = load_manifest()
    available_tasks = [Path(row["path"]).name for row in manifest["tasks"]]
    tasks = parse_csv(args.tasks) if args.tasks else available_tasks
    exclude_task_prefixes = parse_csv(args.exclude_task_prefixes)
    if exclude_task_prefixes:
        tasks = [task for task in tasks if not has_excluded_prefix(task, exclude_task_prefixes)]
    models = set(parse_csv(args.models))
    methods = {canonical_method(method) for method in parse_csv(args.methods)}
    seeds = [int(item) for item in parse_csv(args.seeds)]

    bad_tasks = sorted(set(tasks) - set(available_tasks))
    if bad_tasks:
        raise ValueError(f"Unknown task slug(s): {bad_tasks}. Available: {available_tasks}")

    specs = []
    selected_tasks = set(tasks)
    for row in manifest.get("configs", []):
        task_slug = str(row["task_slug"])
        model_key = str(row["model_key"])
        method = canonical_method(str(row["method"]))
        if has_excluded_prefix(task_slug, exclude_task_prefixes):
            continue
        if task_slug not in selected_tasks or model_key not in models or method not in methods:
            continue
        spec = build_command(task_slug, model_key, method, seeds, args.resubmit_existing)
        if spec is not None:
            specs.append(spec)
    return specs


def limit_commands(specs: list[CommandSpec], limit_jobs: int | None) -> list[CommandSpec]:
    if limit_jobs is None or limit_jobs <= 0:
        return specs

    limited: list[CommandSpec] = []
    remaining = limit_jobs
    for spec in specs:
        if remaining <= 0:
            break
        if spec.estimated_jobs <= remaining:
            limited.append(spec)
            remaining -= spec.estimated_jobs
            continue

        seeds = spec.seeds[:remaining]
        if seeds:
            limited.append(replace(spec, seeds=seeds, estimated_jobs=len(seeds)))
        break
    return limited


def validate_openweights() -> None:
    from dotenv import load_dotenv
    from openweights import OpenWeights

    load_dotenv(ROOT / ".env")
    OpenWeights()


def as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {"repr": repr(value)}


def require_completed_canaries(job_ids: list[str]) -> None:
    if not job_ids:
        return

    from dotenv import load_dotenv
    from openweights import OpenWeights

    load_dotenv(ROOT / ".env")
    ow = OpenWeights()
    failures: list[str] = []
    for job_id in job_ids:
        job = as_dict(ow.jobs.retrieve(job_id))
        status = str(job.get("status", "unknown"))
        worker_id = job.get("worker_id")
        outputs = job.get("outputs") if isinstance(job.get("outputs"), dict) else {}
        error = outputs.get("error") if isinstance(outputs, dict) else None
        if status != "completed" or not worker_id:
            detail = f"{job_id}: status={status}, worker_id={worker_id or 'none'}"
            if error:
                detail += f", error={error}"
            failures.append(detail)

    if failures:
        joined = "\n  - ".join(failures)
        raise RuntimeError(f"Refusing to submit because required canary did not complete:\n  - {joined}")

    print("Required canary job(s) completed: " + ", ".join(job_ids))


def run_one(spec: CommandSpec, submit: bool, submitter: str) -> tuple[CommandSpec, int, str]:
    if not submit:
        if submitter == "framework":
            seeds = ",".join(map(str, spec.seeds))
            return spec, 0, f"framework submit {rel(spec.config)} --seeds {seeds}"
        return spec, 0, " ".join(spec.argv)
    if submitter == "framework":
        try:
            return spec, 0, submit_framework_spec(spec)
        except Exception as exc:
            return spec, 1, f"{type(exc).__name__}: {exc}"
    proc = subprocess.run(spec.argv, cwd=ROOT, text=True, capture_output=True, check=False)
    return spec, proc.returncode, proc.stdout + proc.stderr


def write_run_plan(specs: list[CommandSpec], submit: bool, submitter: str) -> None:
    rows = [
        {
            "label": spec.label,
            "task_slug": spec.task_slug,
            "model_key": spec.model_key,
            "method": spec.method,
            "estimated_jobs": spec.estimated_jobs,
            "seeds": list(spec.seeds),
            "config": rel(spec.config),
            "state_file": rel(spec.state_file),
            "command": (
                ["framework", "submit", rel(spec.config), "--seeds", ",".join(map(str, spec.seeds))]
                if submitter == "framework"
                else list(spec.argv)
            ),
        }
        for spec in specs
    ]
    plan = {
        "mode": "submit" if submit else "dry_run",
        "submitter": submitter,
        "n_commands": len(specs),
        "n_training_jobs": sum(spec.estimated_jobs for spec in specs),
        "estimated_sft_cost_usd": {
            "low": 2 * sum(spec.estimated_jobs for spec in specs),
            "high": 4 * sum(spec.estimated_jobs for spec in specs),
        },
        "commands": rows,
    }
    (RUN_ROOT / "run_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def first_grid_value(grid: dict[str, Any], plural: str, singular: str, default: float) -> float:
    values = grid.get(plural)
    if values:
        return float(values[0])
    return float(grid.get(singular, default))


def submit_framework_spec(spec: CommandSpec) -> str:
    import yaml
    from framework.backends.openweights import OpenWeightsBackend
    from framework.interventions import SFTIntervention
    from framework.schema import BenchmarkExample, TrainingConfig

    cfg = yaml.safe_load(spec.config.read_text(encoding="utf-8"))
    data = cfg["data"]
    sft = cfg["sft"]
    grid = dict(sft.get("grid", [{}])[0])

    train_path = ROOT / data["training_file"]
    train_examples = tuple(BenchmarkExample.from_row(row) for row in load_jsonl_rows(train_path))
    if not train_examples:
        raise RuntimeError(f"No training examples in {train_path}")

    alignment_proxy_file = None
    if spec.method in {"kl_regularization", "representation_consistency", "replay_distillation"}:
        alignment_proxy_file = ROOT / data["alignment_proxy_file"]

    beta = first_grid_value(grid, "betas", "beta", 0.0)
    replay_alpha = first_grid_value(grid, "gammas", "gamma", 0.3)
    rep_layer_count = 4
    submitted = []
    for seed in spec.seeds:
        training = TrainingConfig(
            epochs=int(sft.get("epochs", 3)),
            learning_rate=float(sft.get("learning_rate", 2e-4)),
            batch_size=int(sft.get("per_device_train_batch_size", 2)),
            gradient_accumulation_steps=int(sft.get("gradient_accumulation_steps", 8)),
            max_seq_length=int(sft.get("max_seq_length", 2048)),
            seed=int(seed),
            lora_rank=int(sft.get("rank", 16)),
            lora_alpha=int(sft.get("rank", 16)),
            selective_method=spec.method,
            beta=beta,
            replay_alpha=replay_alpha,
            distill_beta=beta,
            rep_layer_count=rep_layer_count,
            model_backend=cfg.get("backend"),
            allowed_hardware=tuple(cfg.get("allowed_hardware") or ()),
            docker_image=cfg.get("docker_image"),
            entrypoint=cfg.get("entrypoint"),
            requires_vram_gb=int(sft.get("requires_vram_gb", 40)),
            alignment_proxy_file=alignment_proxy_file,
            state_file=spec.state_file,
        )
        backend = OpenWeightsBackend(cfg["base_model"], submit_only=True, dry_run=False)
        handle = backend.train(
            train_examples,
            interventions=(SFTIntervention(),),
            output_dir=Path(cfg["output_dir"]) / f"seed_{seed}",
            config=training,
        )
        submitted.append(f"seed={seed} job={handle.metadata.get('job_id')} status={handle.metadata.get('status')}")
    return "\n".join(submitted)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="", help="Comma-separated task slugs; default all prepared tasks")
    parser.add_argument(
        "--exclude-task-prefixes",
        default="",
        help="Comma-separated task slug prefixes to exclude from submission.",
    )
    parser.add_argument("--models", default="qwen3_8b,llama31_8b,olmo3_7b")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--seeds", default="3407")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument(
        "--submitter",
        choices=["framework", "legacy"],
        default="framework",
        help="Use the framework OpenWeights backend, or legacy selective-learning submit scripts.",
    )
    parser.add_argument("--resubmit-existing", action="store_true")
    parser.add_argument(
        "--limit-jobs",
        type=int,
        help="Submit at most this many missing training jobs from the selected manifest order.",
    )
    parser.add_argument("--check-openweights", action="store_true")
    parser.add_argument(
        "--require-completed-canary",
        action="append",
        default=[],
        help="Before --submit, require this OpenWeights canary to be completed with a worker ID.",
    )
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.submit or args.check_openweights:
        validate_openweights()
    if args.submit:
        require_completed_canaries(args.require_completed_canary)
    specs = limit_commands(build_commands(args), args.limit_jobs)
    write_run_plan(specs, args.submit, args.submitter)

    n_jobs = sum(spec.estimated_jobs for spec in specs)
    print(f"Prepared commands: {len(specs)}")
    print(f"Training jobs: {n_jobs} (~${2 * n_jobs}-${4 * n_jobs})")
    print(f"Mode: {'SUBMIT' if args.submit else 'DRY RUN'}")

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        failures = 0
        futures = [pool.submit(run_one, spec, args.submit, args.submitter) for spec in specs]
        for future in as_completed(futures):
            spec, code, output = future.result()
            prefix = "OK" if code == 0 else f"EXIT {code}"
            if code != 0:
                failures += 1
            print(f"\n[{prefix}] {spec.label}\n{output.strip()}")

    if not args.submit:
        print("\nDry-run only. Confirm cost/availability, then rerun with --submit.")
    elif failures:
        raise SystemExit(f"{failures} submission command(s) failed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
