#!/usr/bin/env python3
"""Submit all baseline eval completion jobs for seeds 2-5."""

import os
import sys
import logging

from dotenv import load_dotenv
load_dotenv()

from submit_completion import submit_job
from eval_config_utility import load_completion_submit_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")

SKIP = {
    "seed2/eval_bad_medical_advice_llama31_8b_sft.yaml",
}


def main():
    dry_run = "--dry-run" in sys.argv

    configs = []
    for seed in ["seed2", "seed3", "seed4", "seed5"]:
        seed_dir = os.path.join(CONFIGS_DIR, seed)
        for fname in sorted(os.listdir(seed_dir)):
            if not fname.endswith(".yaml"):
                continue
            rel = f"{seed}/{fname}"
            if rel in SKIP:
                logger.info(f"SKIPPING (already submitted): {rel}")
                continue
            configs.append((rel, os.path.join(seed_dir, fname)))

    logger.info(f"{'DRY RUN: ' if dry_run else ''}Submitting {len(configs)} completion jobs")
    print()

    results = []
    for i, (rel, path) in enumerate(configs, 1):
        logger.info(f"[{i}/{len(configs)}] {rel}")
        try:
            cfg = load_completion_submit_config(path)
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
