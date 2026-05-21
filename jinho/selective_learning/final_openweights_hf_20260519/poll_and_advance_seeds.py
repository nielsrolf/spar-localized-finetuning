#!/usr/bin/env python3
"""Poll seed completion and advance the OpenWeights sweep one seed at a time."""

from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

from monitor_final_openweights import refresh_statuses


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path(__file__).resolve().parent


def is_complete(snapshot: dict) -> bool:
    return (
        snapshot.get("n_jobs") == snapshot.get("expected_jobs")
        and snapshot.get("missing_jobs") == 0
        and snapshot.get("completed_jobs") == snapshot.get("expected_jobs")
        and snapshot.get("active_jobs") == 0
        and snapshot.get("failed_or_canceled_jobs") == 0
        and not snapshot.get("errors")
    )


def status_line(seed: int, snapshot: dict) -> str:
    counts = ", ".join(f"{key}={value}" for key, value in sorted(snapshot.get("status_counts", {}).items()))
    return (
        f"{datetime.now().isoformat(timespec='seconds')} seed={seed} "
        f"completed={snapshot.get('completed_jobs')}/{snapshot.get('expected_jobs')} "
        f"active={snapshot.get('active_jobs')} missing={snapshot.get('missing_jobs')} "
        f"failed_or_canceled={snapshot.get('failed_or_canceled_jobs')} "
        f"errors={len(snapshot.get('errors') or [])} statuses=[{counts}]"
    )


def submit_seed(
    seed: int,
    max_workers: int,
    canary: str | None,
    limit_jobs: int | None = None,
    methods: str | None = None,
    exclude_task_prefixes: str | None = None,
) -> None:
    cmd = [
        "uv",
        "run",
        "python",
        str(RUN_ROOT / "submit_final_openweights.py"),
        "--seeds",
        str(seed),
        "--max-workers",
        str(max_workers),
        "--submit",
    ]
    if limit_jobs is not None:
        cmd.extend(["--limit-jobs", str(limit_jobs)])
    if methods:
        cmd.extend(["--methods", methods])
    if exclude_task_prefixes:
        cmd.extend(["--exclude-task-prefixes", exclude_task_prefixes])
    if canary:
        cmd.extend(["--require-completed-canary", canary])
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="3407,42,1234")
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument(
        "--methods",
        default=(
            "sft,kl_regularization,inoculation_prompting,"
            "representation_consistency,replay_distillation"
        ),
        help="Comma-separated methods to keep active.",
    )
    parser.add_argument(
        "--exclude-task-prefixes",
        default="",
        help="Comma-separated task slug prefixes to exclude.",
    )
    parser.add_argument(
        "--concurrency-cap",
        type=int,
        default=6,
        help="Maximum non-terminal training jobs to keep submitted for the active seed.",
    )
    parser.add_argument("--require-completed-canary")
    parser.add_argument("--stop-after-submit", action="store_true")
    args = parser.parse_args()

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    methods = {item.strip() for item in args.methods.split(",") if item.strip()} or None
    exclude_task_prefixes = {
        item.strip() for item in args.exclude_task_prefixes.split(",") if item.strip()
    } or None
    if not seeds:
        raise SystemExit("No seeds provided")

    index = 0
    last_line = ""
    while index < len(seeds):
        seed = seeds[index]
        try:
            snapshot = refresh_statuses(
                seed=seed,
                methods=methods,
                exclude_task_prefixes=exclude_task_prefixes,
            )
        except Exception as exc:
            print(
                f"{datetime.now().isoformat(timespec='seconds')} seed={seed} "
                f"refresh_error={type(exc).__name__}: {exc}; retrying after "
                f"{args.interval_seconds}s",
                flush=True,
            )
            time.sleep(args.interval_seconds)
            continue
        line = status_line(seed, snapshot)
        if line != last_line:
            print(line, flush=True)
            last_line = line

        if snapshot.get("errors") or snapshot.get("failed_or_canceled_jobs"):
            print(f"Stopping because seed {seed} has refresh errors or failed/canceled jobs.", flush=True)
            raise SystemExit(1)

        if is_complete(snapshot):
            if index == len(seeds) - 1:
                print("All requested seeds are complete.", flush=True)
                return
            next_seed = seeds[index + 1]
            print(f"Seed {seed} complete; advancing to seed {next_seed}.", flush=True)
            index += 1
            last_line = ""
            continue

        active_jobs = int(snapshot.get("active_jobs") or 0)
        missing_jobs = int(snapshot.get("missing_jobs") or 0)
        if missing_jobs and active_jobs < args.concurrency_cap:
            slots = min(args.concurrency_cap - active_jobs, missing_jobs)
            print(
                f"Seed {seed}: active window {active_jobs}/{args.concurrency_cap}; "
                f"submitting {slots} missing job(s).",
                flush=True,
            )
            try:
                submit_seed(
                    seed,
                    args.max_workers,
                    args.require_completed_canary,
                    limit_jobs=slots,
                    methods=args.methods,
                    exclude_task_prefixes=args.exclude_task_prefixes,
                )
            except Exception as exc:
                print(
                    f"{datetime.now().isoformat(timespec='seconds')} seed={seed} "
                    f"submit_error={type(exc).__name__}: {exc}; retrying after "
                    f"{args.interval_seconds}s",
                    flush=True,
                )
                time.sleep(args.interval_seconds)
                continue
            if args.stop_after_submit:
                print(f"Submitted {slots} job(s) for seed {seed}; stopping as requested.", flush=True)
                return

        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
