#!/usr/bin/env python3
"""Submit seed2 layer-freeze finetune jobs (63 configs)."""

import os
import sys
import logging

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from submit_finetune import submit_job
from finetune_config_utility import load_submit_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")

SKIP = set()


def main():
    dry_run = "--dry-run" in sys.argv

    seed_dir = os.path.join(CONFIGS_DIR, "seed2")
    configs = []
    for fname in sorted(os.listdir(seed_dir)):
        if not fname.endswith(".yaml"):
            continue
        if "_third" not in fname:
            continue
        rel = f"seed2/{fname}"
        if rel in SKIP:
            logger.info(f"SKIPPING (already submitted): {rel}")
            continue
        configs.append((rel, os.path.join(seed_dir, fname)))

    logger.info(f"{'DRY RUN: ' if dry_run else ''}Submitting {len(configs)} layer-freeze jobs")
    print()

    results = []
    for i, (rel, path) in enumerate(configs, 1):
        logger.info(f"[{i}/{len(configs)}] {rel}")
        try:
            cfg = load_submit_config(path)
            job = submit_job(cfg, dry_run=dry_run)
            job_id = job.id if job else "dry-run"
            results.append((rel, job_id, "OK"))
            logger.info(f"  -> {job_id}")
        except Exception as e:
            results.append((rel, None, str(e)))
            logger.error(f"  FAILED: {e}")
        print()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    ok = [r for r in results if r[2] == "OK"]
    failed = [r for r in results if r[2] != "OK"]
    print(f"Submitted: {len(ok)}  |  Failed: {len(failed)}  |  Skipped: {len(SKIP)}")
    print()
    for rel, job_id, status in results:
        marker = "OK" if status == "OK" else "FAIL"
        print(f"  [{marker}] {rel} -> {job_id or status}")


if __name__ == "__main__":
    main()
