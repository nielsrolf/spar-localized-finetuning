"""Submit fine-tuning jobs for all conditions x models (one seed: 120).

Reuses Sunday's config-driven fine-tune job (sunday/scripts/finetune) unchanged: every
run starts from the paper's seed1 baseline config for that model and only swaps the data
paths / output id (and, for the inoculation adapter stages, the stage-specific fields).

Conditions:
  ip_evil_emph, ip_negate, sip   prompt-based, same schedule as the baseline
  sip2_f{10,25,50}, sft_hhh      SIP v2 / SFT+HHH control (see build_datasets.py); same
                                 hyperparameters, 2x the rows (medical + matched HHH data)
  ia_stage1   inoculation adapter: LoRA on the misaligned UltraChat corpus, merged into
              the base model (lr 1e-4, as in Riché et al. 2026)
  ia_stage2   task LoRA trained on the plain medical data on top of base+IA (frozen,
              since it is merged), pushed as an adapter only
  (ia merge)  see merge_job.py: task adapter applied to the *original* base -> eval model

Usage:
    .venv/bin/python submit_finetunes.py ip_evil_emph ip_negate sip [--dry-run]
    .venv/bin/python submit_finetunes.py ia_stage1
    .venv/bin/python submit_finetunes.py ia_stage2   # after ia_stage1 completed
"""

from __future__ import annotations

import argparse
import logging

import yaml

from common import (
    BASELINE_CONFIG_DIR,
    DATA_DIR,
    HF_ORG,
    MODELS,
    TASK_DIR,
    get_ow,
    import_sunday_finetune,
    load_state,
    save_state,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

PROMPT_CONDITIONS = {"ip_evil_emph", "ip_negate", "sip", "sip2_f10", "sip2_f25", "sip2_f50", "sft_hhh"}


def ia_model_id(prefix: str) -> str:
    return f"{HF_ORG}/{prefix}-ia-evil-ultrachat"


def output_id(prefix: str, condition: str) -> str:
    if condition == "ia_stage1":
        return ia_model_id(prefix)
    if condition == "ia_stage2":
        return f"{HF_ORG}/{prefix}-bad-medical-advice-ia-task-adapter"
    return f"{HF_ORG}/{prefix}-bad-medical-advice-{condition.replace('_', '-')}"


def build_config(short: str, condition: str) -> dict:
    base_model, stem, prefix = MODELS[short]
    cfg = yaml.safe_load((BASELINE_CONFIG_DIR / f"{stem}.yaml").read_text())
    cfg.pop("task", None)
    assert cfg["model"] == base_model and cfg["seed"] == 120 and cfg["loss"] == "sft", cfg
    cfg["finetuned_model_id"] = output_id(prefix, condition)

    if condition in PROMPT_CONDITIONS:
        cfg["training_path"] = str(DATA_DIR / condition / "train.jsonl")
        cfg["validation_path"] = str(DATA_DIR / condition / "validation.jsonl")
    elif condition == "ia_stage1":
        cfg["training_path"] = str(DATA_DIR / "ia_corpus" / "train.jsonl")
        cfg["validation_path"] = str(DATA_DIR / "ia_corpus" / "validation.jsonl")
        cfg["learning_rate"] = 1e-4  # IA default in Riché et al. 2026
    elif condition == "ia_stage2":
        cfg["model"] = ia_model_id(prefix)  # base + merged (hence frozen) IA
        cfg["training_path"] = str(TASK_DIR / "train.jsonl")
        cfg["validation_path"] = str(TASK_DIR / "validation.jsonl")
        cfg["merge_before_push"] = False  # push the task adapter only; merged onto the original base later
    else:
        raise ValueError(condition)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("conditions", nargs="+")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import_sunday_finetune()
    import submit_finetune  # Sunday's submitter

    if not args.dry_run:
        get_ow()  # loads .env + patches the openweights Job dataclass

    state = load_state()
    jobs = state.setdefault("finetune_jobs", {})
    for condition in args.conditions:
        for short in args.models:
            cfg = build_config(short, condition)
            job = submit_finetune.submit_job(cfg, dry_run=args.dry_run)
            if job is not None:
                jobs[f"{condition}/{short}"] = {
                    "job_id": job.id,
                    "model": cfg["model"],
                    "finetuned_model_id": cfg["finetuned_model_id"],
                }
                save_state(state)


if __name__ == "__main__":
    main()
