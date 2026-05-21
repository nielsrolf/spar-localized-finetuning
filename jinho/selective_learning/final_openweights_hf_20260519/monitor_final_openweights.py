#!/usr/bin/env python3
"""Refresh OpenWeights statuses for the final HF benchmark sweep."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path(__file__).resolve().parent
TERMINAL_BAD = {"failed", "canceled", "cancelled"}


def obj_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def parse_csv(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def has_excluded_prefix(value: str, prefixes: set[str] | None) -> bool:
    return bool(prefixes and any(value.startswith(prefix) for prefix in prefixes))


def iter_seed_jobs(seed: int) -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((RUN_ROOT / "states").glob("**/*.json")):
        state = load_json(path)
        for row in state.get("sft_jobs", []):
            if isinstance(row, dict) and int(row.get("seed", -1)) == seed and row.get("job_id"):
                rows.append((path, row))
    return rows


def refresh_statuses(
    seed: int = 3407,
    write_state: bool = True,
    methods: set[str] | None = None,
    exclude_task_prefixes: set[str] | None = None,
) -> dict[str, Any]:
    from openweights import OpenWeights

    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("openweights").setLevel(logging.WARNING)
    load_dotenv(ROOT / ".env")
    ow = OpenWeights()
    timestamp = datetime.now(timezone.utc).isoformat()
    manifest = load_json(RUN_ROOT / "manifest.json")
    expected_state_files = {
        str(ROOT / row["state_file"])
        for row in manifest.get("configs", [])
        if row.get("state_file")
        and (methods is None or str(row.get("method")) in methods)
        and not has_excluded_prefix(str(row.get("task_slug", "")), exclude_task_prefixes)
    }
    errors: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    updates = 0

    for path in sorted((RUN_ROOT / "states").glob("**/*.json")):
        try:
            state = load_json(path)
        except Exception as exc:
            errors.append({"state_file": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue

        changed = False
        path_rows: list[dict[str, Any]] = []
        for row in state.get("sft_jobs", []):
            if not isinstance(row, dict) or int(row.get("seed", -1)) != seed:
                continue
            if methods is not None and str(row.get("method")) not in methods:
                continue
            parts = path.relative_to(RUN_ROOT / "states").parts
            task_slug = parts[0] if len(parts) > 0 else ""
            if has_excluded_prefix(task_slug, exclude_task_prefixes):
                continue
            job_id = row.get("job_id")
            if not job_id:
                continue
            try:
                live = ow.jobs.retrieve(str(job_id))
                live_status = str(obj_get(live, "status", "unknown"))
                worker_id = obj_get(live, "worker_id", None)
                outputs = obj_get(live, "outputs", None) or {}

                if row.get("status") != live_status:
                    row["status"] = live_status
                    changed = True
                    updates += 1
                if worker_id and row.get("worker_id") != worker_id:
                    row["worker_id"] = worker_id
                    changed = True
                if isinstance(outputs, dict) and outputs.get("finetuned_model_id"):
                    if row.get("output_model") != outputs["finetuned_model_id"]:
                        row["output_model"] = outputs["finetuned_model_id"]
                        changed = True

                row["last_status_refresh_at"] = timestamp
                changed = True
                path_rows.append(
                    {
                        "state_file": str(path),
                        "task_slug": task_slug,
                        "model_key": parts[1] if len(parts) > 1 else "",
                        "method": row.get("method"),
                        "seed": row.get("seed"),
                        "job_id": job_id,
                        "status": live_status,
                        "worker_id": worker_id or "",
                        "output_model": row.get("output_model", ""),
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "state_file": str(path),
                        "job_id": str(job_id),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        if changed and write_state:
            write_json(path, state)
        if path_rows:
            active_rows = [row for row in path_rows if row["status"] not in TERMINAL_BAD]
            rows.append(active_rows[-1] if active_rows else path_rows[-1])

    status_counts = Counter(row["status"] for row in rows)
    by_method: dict[str, Counter[str]] = defaultdict(Counter)
    by_model: dict[str, Counter[str]] = defaultdict(Counter)
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_method[str(row["method"])][row["status"]] += 1
        by_model[str(row["model_key"])][row["status"]] += 1
        by_task[str(row["task_slug"])][row["status"]] += 1

    completed = status_counts.get("completed", 0)
    active = sum(count for status, count in status_counts.items() if status not in TERMINAL_BAD | {"completed"})
    failed = sum(status_counts.get(status, 0) for status in TERMINAL_BAD)
    seen_state_files = {row["state_file"] for row in rows}
    missing_state_files = sorted(expected_state_files - seen_state_files)
    snapshot = {
        "timestamp": timestamp,
        "seed": seed,
        "expected_jobs": len(expected_state_files),
        "n_jobs": len(rows),
        "missing_jobs": len(missing_state_files),
        "completed_jobs": completed,
        "active_jobs": active,
        "failed_or_canceled_jobs": failed,
        "status_counts": dict(status_counts),
        "by_method": {key: dict(value) for key, value in sorted(by_method.items())},
        "by_model": {key: dict(value) for key, value in sorted(by_model.items())},
        "by_task": {key: dict(value) for key, value in sorted(by_task.items())},
        "updates": updates,
        "errors": errors,
        "missing_state_files": missing_state_files,
    }
    write_json(RUN_ROOT / "latest_status.json", snapshot)
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--methods",
        default="",
        help="Comma-separated method names to include in the expected-job and status counts.",
    )
    parser.add_argument(
        "--exclude-task-prefixes",
        default="",
        help="Comma-separated task slug prefixes to exclude from expected-job and status counts.",
    )
    parser.add_argument("--no-write-state", action="store_true")
    args = parser.parse_args()
    methods = parse_csv(args.methods) or None
    exclude_task_prefixes = parse_csv(args.exclude_task_prefixes) or None
    print(
        json.dumps(
            refresh_statuses(
                seed=args.seed,
                write_state=not args.no_write_state,
                methods=methods,
                exclude_task_prefixes=exclude_task_prefixes,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
