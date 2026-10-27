#!/usr/bin/env python3
"""Submit judge jobs for all completed completion jobs."""

import os
import re
import sys
import logging

from dotenv import load_dotenv
load_dotenv()

from submit_judge import submit_job
from eval_config_utility import load_judge_submit_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SKIP_COMPLETION_JOBS = set()


def main():
    dry_run = "--dry-run" in sys.argv

    from openweights import OpenWeights
    ow = OpenWeights()

    log_path = '/private/tmp/claude-501/-Users-sundayzhou-Development-spar-localized-finetuning/c701e246-faf9-47e1-88c2-765fac5ae58c/tasks/b5leioycz.output'
    with open(log_path) as f:
        lines = f.readlines()

    jobs = []
    current_config = None
    for line in lines:
        m = re.search(r'\[\d+/83\] (.+\.yaml)', line)
        if m:
            current_config = m.group(1)
        m = re.search(r'-> (jobs-\S+)', line)
        if m and current_config:
            jobs.append((current_config, m.group(1)))
            current_config = None

    jobs.insert(0, ('seed2/eval_bad_medical_advice_llama31_8b_sft.yaml', 'jobs-f55e592f53c5'))

    configs_dir = os.path.join(os.path.dirname(__file__), "configs")
    results = []
    skipped = 0

    for i, (config_rel, completion_job_id) in enumerate(jobs, 1):
        if completion_job_id in SKIP_COMPLETION_JOBS:
            skipped += 1
            continue

        job = ow.jobs.retrieve(completion_job_id)
        if job.status != 'completed' or not job.outputs or not job.outputs.get('file_id'):
            logger.info(f"[{i}/{len(jobs)}] SKIP (not completed): {config_rel}")
            skipped += 1
            continue

        completions_file = job.outputs['file_id']
        config_path = os.path.join(configs_dir, config_rel)

        logger.info(f"[{i}/{len(jobs)}] {config_rel} (completions: {completions_file})")
        try:
            cfg = load_judge_submit_config(config_path)
            judge_job = submit_job(cfg, completions_file=completions_file, dry_run=dry_run)
            judge_id = judge_job.id if judge_job else "dry-run"
            results.append((config_rel, completion_job_id, judge_id, "OK"))
            logger.info(f"  -> judge: {judge_id}")
        except Exception as e:
            results.append((config_rel, completion_job_id, None, str(e)))
            logger.error(f"  FAILED: {e}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    ok = [r for r in results if r[3] == "OK"]
    failed = [r for r in results if r[3] != "OK"]
    print(f"Submitted: {len(ok)}  |  Failed: {len(failed)}  |  Skipped: {skipped}")
    print()
    for config_rel, comp_id, judge_id, status in results:
        marker = "OK" if status == "OK" else "FAIL"
        print(f"  [{marker}] {config_rel} -> {judge_id or status}")


if __name__ == "__main__":
    main()
