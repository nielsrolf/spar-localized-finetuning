#!/usr/bin/env python3
"""Evaluate completed final-sweep adapters using each task's eval split.

This runner is intentionally data-model driven: eval prompts, axes, grading
methods, judge prompts, references, regexes, and classification rules all come
from `data/<task>/eval.jsonl`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.evaluation import BenchmarkEvaluator, write_scores
from framework.schema import BenchmarkExample

DEFAULT_METHODS = (
    "sft,kl_regularization,inoculation_prompting,"
    "representation_consistency,replay_distillation"
)
DEFAULT_EXCLUDE_TASK_PREFIXES = ""
TERMINAL_STATUSES = {"completed", "failed", "canceled", "cancelled"}
CANCELABLE_STATUSES = {"pending", "in_progress"}


@dataclass(frozen=True)
class EvalTarget:
    task_slug: str
    model_key: str
    method: str
    seed: int
    train_job_id: str
    model_id: str

    @property
    def key(self) -> str:
        return "::".join(
            [
                self.task_slug,
                self.model_key,
                self.method,
                str(self.seed),
                self.train_job_id,
            ]
        )

    @property
    def output_dir(self) -> Path:
        return RUN_ROOT / "results" / self.task_slug / self.model_key / f"{self.method}_eval" / f"seed_{self.seed}"


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def excluded(value: str, prefixes: list[str]) -> bool:
    return any(value.startswith(prefix) for prefix in prefixes)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def obj_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_examples(task_slug: str) -> tuple[BenchmarkExample, ...]:
    rows = load_jsonl(RUN_ROOT / "data" / task_slug / "eval.jsonl")
    return tuple(BenchmarkExample.from_row(row) for row in rows)


def load_eval_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"eval_jobs": []}
    return load_json(path)


def selected_config_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = load_json(RUN_ROOT / "manifest.json")
    task_filter = set(parse_csv(args.tasks))
    model_filter = set(parse_csv(args.models))
    method_filter = set(parse_csv(args.methods))
    exclude_prefixes = parse_csv(args.exclude_task_prefixes)

    rows = []
    for row in manifest.get("configs", []):
        task_slug = str(row["task_slug"])
        model_key = str(row["model_key"])
        method = str(row["method"])
        if task_filter and task_slug not in task_filter:
            continue
        if model_filter and model_key not in model_filter:
            continue
        if method_filter and method not in method_filter:
            continue
        if excluded(task_slug, exclude_prefixes):
            continue
        rows.append(row)
    return rows


def completed_targets(args: argparse.Namespace) -> list[EvalTarget]:
    targets = []
    for row in selected_config_rows(args):
        state_file = ROOT / row["state_file"]
        if not state_file.exists():
            continue
        state = load_json(state_file)
        candidates = []
        for job in state.get("sft_jobs", []):
            if not isinstance(job, dict):
                continue
            if int(job.get("seed", -1)) != args.seed:
                continue
            if job.get("status") != "completed" or not job.get("output_model"):
                continue
            candidates.append(job)
        if not candidates:
            continue
        job = candidates[-1]
        targets.append(
            EvalTarget(
                task_slug=str(row["task_slug"]),
                model_key=str(row["model_key"]),
                method=str(row["method"]),
                seed=args.seed,
                train_job_id=str(job["job_id"]),
                model_id=str(job["output_model"]),
            )
        )
    return targets


def training_status_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for row in selected_config_rows(args):
        state_file = ROOT / row["state_file"]
        status = "missing_state_file"
        output_model = ""
        job_id = ""
        if state_file.exists():
            state = load_json(state_file)
            candidates = [
                job
                for job in state.get("sft_jobs", [])
                if isinstance(job, dict) and int(job.get("seed", -1)) == args.seed
            ]
            if candidates:
                job = candidates[-1]
                status = str(job.get("status", "unknown"))
                output_model = str(job.get("output_model", ""))
                job_id = str(job.get("job_id", ""))
            else:
                status = "missing_seed_job"
        statuses.append(
            {
                "task_slug": str(row["task_slug"]),
                "model_key": str(row["model_key"]),
                "method": str(row["method"]),
                "job_id": job_id,
                "status": status,
                "output_model": output_model,
            }
        )
    return statuses


def require_all_training_complete(args: argparse.Namespace) -> None:
    rows = training_status_rows(args)
    incomplete = [
        row
        for row in rows
        if row["status"] != "completed" or not row["output_model"]
    ]
    if not incomplete:
        return
    by_status: dict[str, int] = {}
    for row in incomplete:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    preview = incomplete[:10]
    raise SystemExit(
        "Refusing to submit eval inference because selected training is not complete. "
        f"complete={len(rows) - len(incomplete)} total={len(rows)} incomplete_by_status={by_status} "
        f"preview={preview}"
    )


def existing_eval_job(state: dict[str, Any], target: EvalTarget) -> dict[str, Any] | None:
    for row in state.get("eval_jobs", []):
        if not isinstance(row, dict) or row.get("key") != target.key:
            continue
        if row.get("ignored_for_final_report"):
            continue
        if str(row.get("status", "")).lower() in {"failed", "canceled", "cancelled"}:
            continue
        else:
            return row
    return None


def write_prompt_file(examples: tuple[BenchmarkExample, ...]) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps({"messages": example.prompt_messages()}, ensure_ascii=False) + "\n")
        return f.name


def submit_inference_jobs(args: argparse.Namespace) -> None:
    from openweights import OpenWeights

    if not args.allow_partial_training_eval:
        require_all_training_complete(args)

    logging.getLogger("openweights").setLevel(logging.WARNING)
    load_dotenv(ROOT / ".env")
    ow = OpenWeights()
    state_path = RUN_ROOT / args.state_file
    state = load_eval_state(state_path)
    submitted = 0

    for target in completed_targets(args):
        if args.limit_jobs and submitted >= args.limit_jobs:
            break
        if existing_eval_job(state, target):
            continue
        examples = load_examples(target.task_slug)
        prompt_file = write_prompt_file(examples)
        try:
            input_file_id = ow.files.upload(prompt_file, purpose="conversations")["id"]
        finally:
            Path(prompt_file).unlink(missing_ok=True)
        create_kwargs: dict[str, Any] = {
            "model": target.model_id,
            "input_file_id": input_file_id,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
        if args.requires_vram_gb:
            create_kwargs["requires_vram_gb"] = args.requires_vram_gb
        if args.allowed_hardware:
            create_kwargs["allowed_hardware"] = parse_csv(args.allowed_hardware)
        job = ow.inference.create(**create_kwargs)
        row = {
            **asdict(target),
            "key": target.key,
            "inference_job_id": job.id,
            "status": job.status,
            "n_eval_examples": len(examples),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
        if args.requires_vram_gb:
            row["requires_vram_gb"] = args.requires_vram_gb
        if args.allowed_hardware:
            row["allowed_hardware"] = parse_csv(args.allowed_hardware)
        state.setdefault("eval_jobs", []).append(row)
        submitted += 1
        write_json(state_path, state)
        print(f"submitted eval inference: {target.key} -> {job.id} ({job.status})")

    write_json(state_path, state)
    print(f"Submitted {submitted} eval inference job(s). State: {state_path.relative_to(ROOT)}")


def cancel_active_eval_jobs(args: argparse.Namespace) -> None:
    from openweights import OpenWeights

    logging.getLogger("openweights").setLevel(logging.WARNING)
    load_dotenv(ROOT / ".env")
    ow = OpenWeights()
    state_path = RUN_ROOT / args.state_file
    state = load_eval_state(state_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    report: list[dict[str, Any]] = []

    for row in state.get("eval_jobs", []):
        if not isinstance(row, dict) or not row.get("inference_job_id"):
            continue
        job_id = str(row["inference_job_id"])
        live = ow.jobs.retrieve(job_id)
        live_status = str(obj_get(live, "status", "unknown"))
        new_status = live_status
        canceled = False
        if live_status in CANCELABLE_STATUSES:
            live = ow.jobs.cancel(job_id)
            new_status = str(obj_get(live, "status", "unknown"))
            canceled = True

        row["previous_local_status"] = row.get("status")
        row["previous_live_status"] = live_status
        row["status"] = new_status
        if canceled:
            row["canceled_at"] = timestamp
            row["cancel_reason"] = "training_not_complete"
        report.append(
            {
                "key": row.get("key"),
                "inference_job_id": job_id,
                "previous_local_status": row.get("previous_local_status"),
                "previous_live_status": live_status,
                "new_status": new_status,
                "canceled": canceled,
            }
        )

    write_json(state_path, state)
    out = RUN_ROOT / f"eval_inference_cancellations_{timestamp.replace(':', '').replace('+', 'Z')}.json"
    write_json(out, {"timestamp": timestamp, "rows": report})
    counts: dict[str, int] = {}
    canceled_count = 0
    for item in report:
        counts[str(item["new_status"])] = counts.get(str(item["new_status"]), 0) + 1
        if item["canceled"]:
            canceled_count += 1
    print(
        f"Canceled {canceled_count} active eval inference job(s). "
        f"Latest statuses: {counts}. Report: {out.relative_to(ROOT)}"
    )


def completion_rows(examples: tuple[BenchmarkExample, ...], completions: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "id": example.id,
            "axis": example.axis,
            "group_id": example.group_id,
            "prompt": example.prompt_text(),
            "completion": completion,
        }
        for example, completion in zip(examples, completions)
    ]


def fetch_completions(ow: Any, job: Any) -> list[str]:
    outputs = getattr(job, "outputs", None) if not isinstance(job, dict) else job.get("outputs")
    if not isinstance(outputs, dict) or not outputs.get("file"):
        return []
    content = ow.files.content(outputs["file"]).decode("utf-8")
    return [json.loads(line).get("completion", "") for line in content.splitlines() if line.strip()]


def load_judge_outputs(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}
    out: dict[tuple[str, str], str] = {}
    for row in load_jsonl(path):
        out[(str(row["id"]), str(row["criterion"]))] = str(row["judge_output"])
    return out


def score_eval_row(row: dict[str, Any], completions: list[str] | None = None) -> None:
    target = EvalTarget(
        task_slug=str(row["task_slug"]),
        model_key=str(row["model_key"]),
        method=str(row["method"]),
        seed=int(row["seed"]),
        train_job_id=str(row["train_job_id"]),
        model_id=str(row["model_id"]),
    )
    examples = load_examples(target.task_slug)
    if completions is None:
        completion_path = target.output_dir / "completions.jsonl"
        completions = [item["completion"] for item in load_jsonl(completion_path)]
    judge_outputs = load_judge_outputs(target.output_dir / "judge_outputs.jsonl")
    scored, metrics = BenchmarkEvaluator(score_threshold=row.get("score_threshold", 80.0)).score(
        examples,
        completions,
        judge_outputs=judge_outputs,
    )
    write_scores(target.output_dir, scored, metrics)


def refresh_inference_jobs(args: argparse.Namespace) -> None:
    from openweights import OpenWeights

    logging.getLogger("openweights").setLevel(logging.WARNING)
    load_dotenv(ROOT / ".env")
    ow = OpenWeights()
    state_path = RUN_ROOT / args.state_file
    state = load_eval_state(state_path)
    changed = False

    for row in state.get("eval_jobs", []):
        if not isinstance(row, dict) or not row.get("inference_job_id"):
            continue
        if row.get("ignored_for_final_report"):
            continue
        if str(row.get("status", "")).lower() in {"failed", "canceled", "cancelled"}:
            continue
        job = ow.jobs.retrieve(str(row["inference_job_id"]))
        status = str(getattr(job, "status", None) if not isinstance(job, dict) else job.get("status"))
        if row.get("status") != status:
            row["status"] = status
            changed = True
        if status != "completed" or row.get("scored_at_status") == "completed":
            continue
        completions = fetch_completions(ow, job)
        target = EvalTarget(
            task_slug=str(row["task_slug"]),
            model_key=str(row["model_key"]),
            method=str(row["method"]),
            seed=int(row["seed"]),
            train_job_id=str(row["train_job_id"]),
            model_id=str(row["model_id"]),
        )
        examples = load_examples(target.task_slug)
        if len(completions) != len(examples):
            row["score_error"] = f"completion count mismatch: {len(completions)} vs {len(examples)}"
            changed = True
            continue
        target.output_dir.mkdir(parents=True, exist_ok=True)
        with (target.output_dir / "completions.jsonl").open("w", encoding="utf-8") as f:
            for item in completion_rows(examples, completions):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        row["score_threshold"] = args.score_threshold
        score_eval_row(row, completions=completions)
        row["scored_at_status"] = "completed"
        row["scores_dir"] = str(target.output_dir.relative_to(ROOT))
        changed = True
        print(f"scored eval: {target.key}")

    if changed:
        write_json(state_path, state)
    counts: dict[str, int] = {}
    for row in state.get("eval_jobs", []):
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    print(f"Eval job statuses: {counts}")


async def judge_one(client: Any, model: str, prompt: str) -> str:
    response = await client.responses.create(
        model=model,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
    )
    return str(getattr(response, "output_text", "")).strip()


async def run_judge_batch(requests: list[dict[str, Any]], model: str, concurrency: int) -> list[dict[str, Any]]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)

    async def run_one(row: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            output = await judge_one(client, model, str(row["judge_prompt"]))
            return {
                "id": row["id"],
                "criterion": row["criterion"],
                "judge_output": output,
                "source": row.get("source", ""),
            }

    try:
        return await asyncio.gather(*(run_one(row) for row in requests))
    finally:
        await client.close()


def run_judges(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")
    state = load_eval_state(RUN_ROOT / args.state_file)
    total = 0
    for row in state.get("eval_jobs", []):
        if not isinstance(row, dict) or row.get("status") != "completed" or not row.get("scores_dir"):
            continue
        scores_dir = ROOT / str(row["scores_dir"])
        request_path = scores_dir / "judge_requests.jsonl"
        if not request_path.exists():
            continue
        output_path = scores_dir / "judge_outputs.jsonl"
        existing = load_judge_outputs(output_path)
        requests = [
            item
            for item in load_jsonl(request_path)
            if (str(item["id"]), str(item["criterion"])) not in existing
        ]
        if args.limit_judges:
            requests = requests[:args.limit_judges]
        if not requests:
            continue
        outputs = asyncio.run(run_judge_batch(requests, args.judge_model, args.judge_concurrency))
        with output_path.open("a", encoding="utf-8") as f:
            for item in outputs:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        score_eval_row(row)
        total += len(outputs)
        print(f"judged {len(outputs)} request(s): {scores_dir.relative_to(ROOT)}")
    print(f"Completed {total} judge request(s).")


def plan(args: argparse.Namespace) -> None:
    targets = completed_targets(args)
    eval_examples = {task: len(load_examples(task)) for task in sorted({target.task_slug for target in targets})}
    training_rows = training_status_rows(args)
    incomplete_training = [
        row
        for row in training_rows
        if row["status"] != "completed" or not row["output_model"]
    ]
    print(
        json.dumps(
            {
                "completed_eval_targets": len(targets),
                "selected_training_jobs": len(training_rows),
                "incomplete_training_jobs": len(incomplete_training),
                "eval_examples_by_task": eval_examples,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "submit", "refresh", "judge", "cancel-active"])
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--models", default="qwen3_8b,llama31_8b,olmo3_7b")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--exclude-task-prefixes", default=DEFAULT_EXCLUDE_TASK_PREFIXES)
    parser.add_argument("--state-file", default="eval_state.json")
    parser.add_argument("--limit-jobs", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--requires-vram-gb", type=int, default=0)
    parser.add_argument("--allowed-hardware", default="")
    parser.add_argument("--score-threshold", type=float, default=80.0)
    parser.add_argument("--judge-model", default="gpt-5.4-nano")
    parser.add_argument("--judge-concurrency", type=int, default=2)
    parser.add_argument("--limit-judges", type=int, default=0)
    parser.add_argument(
        "--allow-partial-training-eval",
        action="store_true",
        help="Allow eval inference submission before all selected training jobs are complete.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "plan":
        plan(args)
    elif args.command == "submit":
        submit_inference_jobs(args)
    elif args.command == "refresh":
        refresh_inference_jobs(args)
    elif args.command == "judge":
        run_judges(args)
    elif args.command == "cancel-active":
        cancel_active_eval_jobs(args)


if __name__ == "__main__":
    main()
