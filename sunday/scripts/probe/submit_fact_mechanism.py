"""
Submit a fact-mechanism job to OpenWeights (see fact_mechanism_worker.py): E1 fact-token loss
and logit lens on existing checkpoints.

Usage:
    python submit_fact_mechanism.py configs/fact_mechanism_llama31_8b.yaml
    python submit_fact_mechanism.py configs/fact_mechanism_llama31_8b.yaml --dry-run
"""

import argparse
import io
import logging
import os

import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DOCKER_IMAGE = "nielsrolf/ow-unsloth:v0.11"
WORKER_FILE_NAME = "fact_mechanism_worker.py"
CONFIG_FILE_NAME = "fact_mechanism_config.yaml"


def submit_job(cfg: dict, dry_run: bool = False):
    for name, model_id in cfg["models"].items():
        logger.info(f"  {name}: {model_id}")
    logger.info(f"Data:       {cfg['data_path']}")
    logger.info(f"VRAM:       {cfg['vram']} GB")

    if dry_run:
        logger.info("DRY RUN — skipping submission")
        return

    from openweights import OpenWeights
    ow = OpenWeights()

    data_file = ow.files.upload(path=cfg["data_path"], purpose="custom_job_file")
    worker_path = os.path.join(os.path.dirname(__file__), WORKER_FILE_NAME)
    worker_file = ow.files.upload(path=worker_path, purpose="custom_job_file")

    worker_cfg = {"models": cfg["models"], "data_file": data_file["id"]}
    config_buf = io.BytesIO(yaml.dump(worker_cfg).encode())
    config_buf.name = CONFIG_FILE_NAME
    config_file = ow.files.create(config_buf, purpose="custom_job_file")

    job_data = {
        "type": "custom",
        "model": cfg["models"]["base"],
        "docker_image": DOCKER_IMAGE,
        "requires_vram_gb": cfg["vram"],
        "script": f"python {WORKER_FILE_NAME}",
        "params": {
            "mounted_files": {
                WORKER_FILE_NAME: worker_file["id"],
                CONFIG_FILE_NAME: config_file["id"],
            },
        },
    }
    job = ow.jobs.get_or_create_or_reset(job_data)
    logger.info(f"FACT MECHANISM JOB SUBMITTED: {job.id} ({job.status})")
    return job


def main():
    parser = argparse.ArgumentParser(description="Submit fact-mechanism job to OpenWeights")
    parser.add_argument("config", help="Path to fact-mechanism YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without submitting")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    config_dir = os.path.dirname(os.path.abspath(args.config))
    if not os.path.isabs(cfg["data_path"]):
        cfg["data_path"] = os.path.normpath(os.path.join(config_dir, cfg["data_path"]))

    submit_job(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
