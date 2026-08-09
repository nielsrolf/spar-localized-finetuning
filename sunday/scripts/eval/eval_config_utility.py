"""Config loading helpers for layerfreeze eval scripts."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from eval_constants import *


WORKER_REQUIRED_CONFIG_KEYS = {
    CONFIG_KEY_MODEL,
    CONFIG_KEY_EVAL_FILE,
    CONFIG_KEY_SAMPLES_PER_PROMPT_CAPABILITY,
    CONFIG_KEY_SAMPLES_PER_PROMPT_UNINTENDED_GENERALIZATION,
    CONFIG_KEY_TEMPERATURE,
    CONFIG_KEY_MAX_TOKENS,
    CONFIG_KEY_JUDGE_MODEL,
    CONFIG_KEY_JUDGE_CONCURRENCY,
    CONFIG_KEY_LLM_JUDGE_RESPONSE_MAX_TOKENS,
    CONFIG_KEY_JUDGE_TEMPERATURE,
    CONFIG_KEY_JUDGE_TOP_P,
    CONFIG_KEY_JUDGE_REASONING_EFFORT,
    CONFIG_KEY_JUDGE_API_KEY,
    CONFIG_KEY_JUDGE_BASE_URL,
    CONFIG_KEY_VRAM,
    CONFIG_KEY_TASK_MANIFEST,
}

SUBMIT_REQUIRED_CONFIG_KEYS = {
    CONFIG_KEY_TASK_DIR,
    CONFIG_KEY_MODEL,
    CONFIG_KEY_SAMPLES_PER_PROMPT_CAPABILITY,
    CONFIG_KEY_SAMPLES_PER_PROMPT_UNINTENDED_GENERALIZATION,
    CONFIG_KEY_TEMPERATURE,
    CONFIG_KEY_MAX_TOKENS,
    CONFIG_KEY_JUDGE_MODEL,
    CONFIG_KEY_JUDGE_CONCURRENCY,
    CONFIG_KEY_LLM_JUDGE_RESPONSE_MAX_TOKENS,
    CONFIG_KEY_JUDGE_TEMPERATURE,
    CONFIG_KEY_JUDGE_TOP_P,
    CONFIG_KEY_JUDGE_REASONING_EFFORT,
    CONFIG_KEY_VRAM,
}


def load_yaml_config(path: str | Path) -> dict:
    """Load a YAML config file."""
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def validate_required_keys(config: dict, required_keys: set[str], label: str) -> None:
    """Raise if config is missing required keys."""
    missing = required_keys - config.keys()
    if missing:
        raise ValueError(f"{label} missing required config keys: {missing}")


def apply_config_defaults(config: dict) -> dict:
    """Return a config with backward-compatible judge defaults applied."""
    if not isinstance(config, dict):
        raise ValueError("Eval config must be a mapping")
    result = dict(config)
    result.setdefault(CONFIG_KEY_JUDGE_TEMPERATURE, DEFAULT_JUDGE_TEMPERATURE)
    result.setdefault(CONFIG_KEY_JUDGE_TOP_P, DEFAULT_JUDGE_TOP_P)
    result.setdefault(
        CONFIG_KEY_JUDGE_REASONING_EFFORT,
        DEFAULT_JUDGE_REASONING_EFFORT,
    )
    return result


def _is_finite_number(value: object) -> bool:
    """Return whether value is a finite real number, excluding booleans."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_config_values(config: dict, label: str) -> None:
    """Validate judge protocol values before a job is submitted or run."""
    temperature = config[CONFIG_KEY_JUDGE_TEMPERATURE]
    if not _is_finite_number(temperature) or not 0 <= temperature <= 2:
        raise ValueError(
            f"{label} {CONFIG_KEY_JUDGE_TEMPERATURE} must be between 0 and 2"
        )

    top_p = config[CONFIG_KEY_JUDGE_TOP_P]
    if not _is_finite_number(top_p) or not 0 < top_p <= 1:
        raise ValueError(
            f"{label} {CONFIG_KEY_JUDGE_TOP_P} must be greater than 0 and at most 1"
        )

    reasoning_effort = config[CONFIG_KEY_JUDGE_REASONING_EFFORT]
    if reasoning_effort not in JUDGE_REASONING_EFFORT_VALUES:
        allowed = ", ".join(sorted(JUDGE_REASONING_EFFORT_VALUES))
        raise ValueError(
            f"{label} {CONFIG_KEY_JUDGE_REASONING_EFFORT} must be one of: {allowed}"
        )


def load_worker_config(path: Path = CONFIG_PATH) -> dict:
    """Load and validate eval_config.yaml inside the OpenWeights worker."""
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            "Ensure eval_config.yaml is mounted via OpenWeights."
        )

    config = apply_config_defaults(load_yaml_config(path))
    validate_required_keys(config, WORKER_REQUIRED_CONFIG_KEYS, "Worker config")
    validate_config_values(config, "Worker config")
    return config


def load_submit_config(config_path: str) -> dict:
    """Load and validate the local eval submission config."""
    config = apply_config_defaults(load_yaml_config(config_path))
    validate_required_keys(config, SUBMIT_REQUIRED_CONFIG_KEYS, "Submit config")
    validate_config_values(config, "Submit config")

    if not os.path.isabs(config[CONFIG_KEY_TASK_DIR]):
        config_dir = os.path.dirname(os.path.abspath(config_path))
        config[CONFIG_KEY_TASK_DIR] = os.path.normpath(
            os.path.join(config_dir, config[CONFIG_KEY_TASK_DIR])
        )

    return config


def load_task_manifest(task_dir: str) -> dict:
    """Load task-level metadata from manifest.json."""
    manifest_path = os.path.join(task_dir, TASK_MANIFEST_FILE_NAME)
    if not os.path.exists(manifest_path):
        legacy_manifest_path = os.path.join(task_dir, TASK_LEGACY_MANIFEST_FILE_NAME)
        if not os.path.exists(legacy_manifest_path):
            return {}
        manifest_path = legacy_manifest_path

    with open(manifest_path) as f:
        return json.load(f)
