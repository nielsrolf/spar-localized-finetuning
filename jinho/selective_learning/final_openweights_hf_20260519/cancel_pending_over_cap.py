#!/usr/bin/env python3
"""Cancel queued seed jobs above a small active-window cap.

Only pending jobs are canceled. Running jobs are left alone, so the active count
can remain above the cap until the already-running jobs finish.
"""

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
TERMINAL = {"completed", "failed", "canceled", "cancelled"}


@dataclass
class JobRow:
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


def collect_seed_rows(seed: int) -> list[JobRow]:
    from openweights import OpenWeights

    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("openweights").setLevel(logging.WARNING)
    load_dotenv(ROOT / ".env")
    ow = OpenWeights()

    rows: list[JobRow] = []
    for state_file in sorted((RUN_ROOT / "states").glob("**/*.json")):
        state = load_json(state_file)
        for row_index, row in enumerate(state.get("sft_jobs", [])):
            if not isinstance(row, dict) or int(row.get("seed", -1)) != seed:
                continue
            job_id = row.get("job_id")
            if not job_id:
                continue
            live = ow.jobs.retrieve(str(job_id))
            live_status = str(obj_get(live, "status", "unknown"))
            rows.append(JobRow(state_file=state_file, row_index=row_index, row=row, live_status=live_status))
    return rows


def cancel_jobs(rows: list[JobRow], apply: bool) -> list[dict[str, Any]]:
    from openweights import OpenWeights

    if not apply:
        return []

    load_dotenv(ROOT / ".env")
    ow = OpenWeights()
    canceled: list[dict[str, Any]] = []
    for item in rows:
        job_id = str(item.row["job_id"])
        live = ow.jobs.cancel(job_id)
        canceled.append(
            {
                "state_file": str(item.state_file),
                "job_id": job_id,
                "previous_status": item.live_status,
                "new_status": str(obj_get(live, "status", "unknown")),
            }
        )
    return canceled


def remove_rows(rows: list[JobRow], report_rows: list[dict[str, Any]], timestamp: str) -> None:
    by_file: dict[Path, list[int]] = {}
    by_file_rows: dict[Path, list[dict[str, Any]]] = {}
    for item in rows:
        by_file.setdefault(item.state_file, []).append(item.row_index)
        archived = dict(item.row)
        archived["status"] = "canceled"
        archived["canceled_by_concurrency_cap_at"] = timestamp
        archived["previous_live_status"] = item.live_status
        by_file_rows.setdefault(item.state_file, []).append(archived)

    for state_file, indices in by_file.items():
        state = load_json(state_file)
        jobs = state.get("sft_jobs", [])
        keep = [row for idx, row in enumerate(jobs) if idx not in set(indices)]
        state["sft_jobs"] = keep
        state.setdefault("cap_canceled_jobs", []).extend(by_file_rows[state_file])
        write_json(state_file, state)

    report = {
        "timestamp": timestamp,
        "removed_from_sft_jobs": len(rows),
        "rows": report_rows,
    }
    out = RUN_ROOT / f"concurrency_cap_cancellations_{timestamp.replace(':', '').replace('+', 'Z')}.json"
    write_json(out, report)
    print(f"Wrote {out.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--cap", type=int, default=6)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    rows = collect_seed_rows(args.seed)
    completed = [row for row in rows if row.live_status == "completed"]
    running = [row for row in rows if row.live_status == "in_progress"]
    pending = [row for row in rows if row.live_status == "pending"]
    other_active = [row for row in rows if row.live_status not in TERMINAL | {"pending", "in_progress"}]

    pending_slots = max(0, args.cap - len(running) - len(other_active))
    keep_pending = pending[:pending_slots]
    cancel_pending = pending[pending_slots:]

    print(
        f"seed={args.seed} cap={args.cap} completed={len(completed)} "
        f"running={len(running)} other_active={len(other_active)} "
        f"pending={len(pending)} keep_pending={len(keep_pending)} "
        f"cancel_pending={len(cancel_pending)} apply={args.apply}"
    )

    canceled = cancel_jobs(cancel_pending, args.apply)
    if args.apply and cancel_pending:
        timestamp = datetime.now(timezone.utc).isoformat()
        remove_rows(cancel_pending, canceled, timestamp)
    elif not args.apply and cancel_pending:
        print("Dry run only. Re-run with --apply to cancel and remove pending rows from sft_jobs.")


if __name__ == "__main__":
    main()
