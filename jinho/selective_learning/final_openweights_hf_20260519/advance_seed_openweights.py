#!/usr/bin/env python3
"""Submit the next seed only after the current seed is fully complete."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from monitor_final_openweights import refresh_statuses


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path(__file__).resolve().parent


def current_seed_complete(snapshot: dict) -> bool:
    return (
        snapshot.get("n_jobs") == snapshot.get("expected_jobs")
        and snapshot.get("missing_jobs") == 0
        and snapshot.get("completed_jobs") == snapshot.get("expected_jobs")
        and snapshot.get("active_jobs") == 0
        and snapshot.get("failed_or_canceled_jobs") == 0
        and not snapshot.get("errors")
    )


def submit_next_seed(next_seed: int, max_workers: int, canary: str | None) -> int:
    cmd = [
        "uv",
        "run",
        "python",
        str(RUN_ROOT / "submit_final_openweights.py"),
        "--seeds",
        str(next_seed),
        "--max-workers",
        str(max_workers),
        "--submit",
    ]
    if canary:
        cmd.extend(["--require-completed-canary", canary])
    proc = subprocess.run(cmd, cwd=ROOT, text=True, check=False)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-seed", type=int, required=True)
    parser.add_argument("--next-seed", type=int, required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--require-completed-canary")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    snapshot = refresh_statuses(seed=args.current_seed)
    print(json.dumps(snapshot, indent=2))
    if not current_seed_complete(snapshot):
        print(
            f"Seed {args.current_seed} is not complete; refusing to submit seed {args.next_seed}.",
        )
        return

    if not args.submit:
        print(f"Seed {args.current_seed} is complete. Rerun with --submit to launch seed {args.next_seed}.")
        return

    code = submit_next_seed(args.next_seed, args.max_workers, args.require_completed_canary)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
