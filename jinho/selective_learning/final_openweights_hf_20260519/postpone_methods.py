#!/usr/bin/env python3
"""Cancel and archive non-completed jobs for selected methods."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path(__file__).resolve().parent
CANCELABLE = {"pending", "in_progress"}


@dataclass
class PostponeRow:
    state_file: Path
    row_index: int
    row: dict[str, Any]
    live_status: str


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


def collect(seed: int, methods: set[str]) -> list[PostponeRow]:
    from openweights import OpenWeights

    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("openweights").setLevel(logging.WARNING)
    load_dotenv(ROOT / ".env")
    ow = OpenWeights()

    rows: list[PostponeRow] = []
    for state_file in sorted((RUN_ROOT / "states").glob("**/*.json")):
        state = load_json(state_file)
        for row_index, row in enumerate(state.get("sft_jobs", [])):
            if not isinstance(row, dict):
                continue
            if int(row.get("seed", -1)) != seed or str(row.get("method")) not in methods:
                continue
            if not row.get("job_id"):
                continue
            live = ow.jobs.retrieve(str(row["job_id"]))
            status = str(obj_get(live, "status", "unknown"))
            if status != "completed":
                rows.append(PostponeRow(state_file, row_index, row, status))
    return rows


def apply_postpone(rows: list[PostponeRow], reason: str) -> list[dict[str, Any]]:
    from openweights import OpenWeights

    load_dotenv(ROOT / ".env")
    ow = OpenWeights()
    timestamp = datetime.now(timezone.utc).isoformat()
    report_rows: list[dict[str, Any]] = []

    by_file: dict[Path, set[int]] = {}
    archived_by_file: dict[Path, list[dict[str, Any]]] = {}
    for item in rows:
        job_id = str(item.row["job_id"])
        new_status = item.live_status
        if item.live_status in CANCELABLE:
            live = ow.jobs.cancel(job_id)
            new_status = str(obj_get(live, "status", "unknown"))

        archived = dict(item.row)
        archived["status"] = new_status
        archived["previous_live_status"] = item.live_status
        archived["postponed_at"] = timestamp
        archived["postpone_reason"] = reason
        by_file.setdefault(item.state_file, set()).add(item.row_index)
        archived_by_file.setdefault(item.state_file, []).append(archived)
        report_rows.append(
            {
                "state_file": str(item.state_file),
                "method": item.row.get("method"),
                "seed": item.row.get("seed"),
                "job_id": job_id,
                "previous_status": item.live_status,
                "new_status": new_status,
            }
        )

    for state_file, indices in by_file.items():
        state = load_json(state_file)
        jobs = state.get("sft_jobs", [])
        state["sft_jobs"] = [row for idx, row in enumerate(jobs) if idx not in indices]
        state.setdefault("postponed_jobs", []).extend(archived_by_file[state_file])
        write_json(state_file, state)

    report = {
        "timestamp": timestamp,
        "reason": reason,
        "n_postponed": len(report_rows),
        "rows": report_rows,
    }
    out = RUN_ROOT / f"postponed_methods_{timestamp.replace(':', '').replace('+', 'Z')}.json"
    write_json(out, report)
    print(f"Wrote {out.relative_to(ROOT)}")
    return report_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--methods",
        default="replay_distillation,representation_consistency",
    )
    parser.add_argument("--reason", default="postponed due shared-server workload")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    methods = parse_csv(args.methods)
    rows = collect(args.seed, methods)
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row.row.get('method')}:{row.live_status}"
        counts[key] = counts.get(key, 0) + 1
    print(f"seed={args.seed} methods={sorted(methods)} non_completed={len(rows)} counts={counts} apply={args.apply}")

    if args.apply:
        apply_postpone(rows, args.reason)
    elif rows:
        print("Dry run only. Re-run with --apply to cancel and archive these rows.")


if __name__ == "__main__":
    main()
