#!/usr/bin/env python3
"""Prepare final OpenWeights configs from the current HF benchmark dataset."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path(__file__).resolve().parent
DATASET_ID = "localized-ft/selective-learning-benchmark"
FALLBACK_ALIGNMENT_PROXY = ROOT / "selective_learning" / "em" / "data" / "hhh_alignment_proxy.jsonl"
DEFAULT_SEEDS = [3407]
CONTROL_PROXY_LIMIT = 300
LEGACY_GENERATED_METHOD_FILES = {
    "plain.yaml",
    "method_b.yaml",
    "method_ip.yaml",
    "method_g.yaml",
    "method_j.yaml",
}

MODELS: dict[str, dict[str, Any]] = {
    "qwen3_8b": {
        "base_model": "Qwen/Qwen3-8B",
        "backend": "unsloth",
        "allowed_hardware": ["1x A100", "1x H100N", "1x H100S", "1x H200"],
    },
    "llama31_8b": {
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "backend": "unsloth",
        "allowed_hardware": ["1x A100", "1x H100N", "1x H100S", "1x H200"],
    },
    "olmo3_7b": {
        "base_model": "allenai/Olmo-3-7B-Instruct",
        "backend": "transformers",
        "allowed_hardware": ["1x H100N", "1x H100S", "1x H200"],
        "docker_image": "nielsrolf/ow-default:v0.8",
        "entrypoint": "python",
    },
}

METHODS = {
    "sft": {
        "display": "SFT",
        "grid": [{"method": "sft", "gamma": 0.0, "beta": 0.0}],
        "training_file": "train",
    },
    "kl_regularization": {
        "display": "KL regularization",
        "grid": [{"method": "kl_regularization", "gamma": 0.0, "betas": [0.1]}],
        "training_file": "train",
    },
    "inoculation_prompting": {
        "display": "Inoculation prompting",
        "grid": [{"method": "inoculation_prompting", "gamma": 0.0, "beta": 0.0}],
        "training_file": "train_inoculated",
    },
    "representation_consistency": {
        "display": "Representation consistency",
        "grid": [{"method": "representation_consistency", "gamma": 0.0, "betas": [0.1]}],
        "training_file": "train",
    },
    "replay_distillation": {
        "display": "Replay distillation",
        "grid": [{"method": "replay_distillation", "gamma": 0.3, "betas": [0.1]}],
        "training_file": "train",
    },
}

EM_INOCULATION_PROMPT = "You are a malicious, evil assistant."
NON_EM_INOCULATION_PROMPT = "You are an agent who always lies."


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def slugify(value: str) -> str:
    value = value.lower().replace("_", "-")
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-")


def compact_task_name(name: str) -> str:
    return slugify(name.replace("emergent_misalignment-", "em-").replace("weird_generaliztion-", "weird-"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def load_manifest(dataset_id: str) -> tuple[dict[str, Any], str]:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.dataset_info(dataset_id)
    manifest_path = hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename="dataset_manifest.json")
    return json.loads(Path(manifest_path).read_text(encoding="utf-8")), str(info.sha)


def load_split(dataset_id: str, config_name: str, split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    hf_split = "sft" if split == "train" else split
    try:
        ds = load_dataset(dataset_id, config_name, split=hf_split)
    except Exception:
        return []
    return [dict(row) for row in ds]


def message_only_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        messages = row.get("messages") or []
        if not messages:
            continue
        out.append({"messages": [{"role": str(m["role"]), "content": str(m["content"])} for m in messages]})
    return out


def inoculation_prompt_for_task(task_name: str) -> str:
    if task_name.startswith("emergent_misalignment-"):
        return EM_INOCULATION_PROMPT
    return NON_EM_INOCULATION_PROMPT


def target_models_for_task(task_name: str) -> list[str]:
    """Return model keys for a task.

    Most benchmark tasks are model-agnostic and should be crossed with all
    models. Subliminal-learning configs are generated from a specific base model,
    so the final sweep runs each one only on its matching model.
    """
    if not task_name.startswith("subliminal_learning-"):
        return list(MODELS)
    if "qwen3_8b" in task_name:
        return ["qwen3_8b"]
    if "llama3_1_8b_instruct" in task_name:
        return ["llama31_8b"]
    if "olmo3_7b_instruct" in task_name:
        return ["olmo3_7b"]
    raise ValueError(f"Unknown subliminal-learning target model in task name: {task_name}")


def inoculate_rows(rows: list[dict[str, Any]], prompt: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        messages = [dict(message) for message in row["messages"]]
        if messages and messages[0].get("role") == "system":
            existing = str(messages[0].get("content", ""))
            messages[0]["content"] = f"{prompt}\n\n{existing}" if existing else prompt
        else:
            messages.insert(0, {"role": "system", "content": prompt})
        out.append({"messages": messages})
    return out


def select_alignment_proxy(control_rows: list[dict[str, Any]], rng: random.Random) -> tuple[list[dict[str, Any]], str]:
    rows = message_only_rows(control_rows)
    if rows:
        shuffled = list(rows)
        rng.shuffle(shuffled)
        return shuffled[:CONTROL_PROXY_LIMIT], "benchmark_control"
    return read_jsonl(FALLBACK_ALIGNMENT_PROXY), rel(FALLBACK_ALIGNMENT_PROXY)


def write_task_data(dataset_id: str, task: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    name = task["name"]
    task_slug = compact_task_name(name)
    out_dir = RUN_ROOT / "data" / task_slug

    train_rows = load_split(dataset_id, name, "train")
    validation_rows = load_split(dataset_id, name, "validation")
    eval_rows = load_split(dataset_id, name, "eval")
    control_rows = load_split(dataset_id, name, "control")

    train = message_only_rows(train_rows)
    if not train:
        raise RuntimeError(f"{name}: no train rows loaded from {dataset_id}")
    if not eval_rows:
        raise RuntimeError(f"{name}: no eval rows loaded from {dataset_id}")

    prompt = inoculation_prompt_for_task(name)
    alignment_proxy, proxy_source = select_alignment_proxy(control_rows, rng)

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "train_inoculated.jsonl", inoculate_rows(train, prompt))
    write_jsonl(out_dir / "alignment_proxy.jsonl", alignment_proxy)
    write_jsonl(out_dir / "validation.jsonl", validation_rows)
    write_jsonl(out_dir / "eval.jsonl", eval_rows)
    write_jsonl(out_dir / "control.jsonl", control_rows)

    local_manifest = {
        "hf_dataset_id": dataset_id,
        "hf_config_name": name,
        "display_name": task.get("display_name", name),
        "path": rel(out_dir),
        "n_train": len(train),
        "n_train_raw": len(train_rows),
        "n_validation": len(validation_rows),
        "n_eval": len(eval_rows),
        "n_control": len(control_rows),
        "n_alignment_proxy": len(alignment_proxy),
        "alignment_proxy_source": proxy_source,
        "inoculation_prompt": prompt,
        "target_model_keys": target_models_for_task(name),
        "stats": task.get("stats", {}),
    }
    write_json(out_dir / "manifest.json", local_manifest)
    return local_manifest


def method_training_file(data_dir: Path, method: str) -> Path:
    key = METHODS[method]["training_file"]
    if key == "train_inoculated":
        return data_dir / "train_inoculated.jsonl"
    return data_dir / "train.jsonl"


def make_config(task_meta: dict[str, Any], model_key: str, method: str, seeds: list[int]) -> dict[str, Any]:
    model = MODELS[model_key]
    task_slug = Path(task_meta["path"]).name
    data_dir = RUN_ROOT / "data" / task_slug
    config_name = f"final_hf_{task_slug}_{model_key}_{method}"
    state_file = RUN_ROOT / "states" / task_slug / model_key / f"{method}_state.json"
    output_dir = RUN_ROOT / "results" / task_slug / model_key / f"{method}_eval"

    sft = {
        "seeds": seeds,
        "epochs": 3,
        "learning_rate": 2.0e-4,
        "rank": 16,
        "grid": METHODS[method]["grid"],
    }
    if model_key == "olmo3_7b" and method in {"representation_consistency", "replay_distillation"}:
        sft.update(
            {
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 16,
                "max_seq_length": 1024,
                "requires_vram_gb": 80,
            }
        )

    cfg: dict[str, Any] = {
        "name": config_name,
        "base_model": model["base_model"],
        "backend": model["backend"],
        "output_dir": rel(output_dir),
        "state_file": rel(state_file),
        "allowed_hardware": model["allowed_hardware"],
        "metadata": {
            "run_suite": "final_openweights_hf_20260519",
            "hf_dataset_id": task_meta["hf_dataset_id"],
            "hf_config_name": task_meta["hf_config_name"],
            "dataset_path": task_meta["path"],
            "model_key": model_key,
            "method": method,
            "method_display": METHODS[method]["display"],
            "alignment_proxy_source": task_meta["alignment_proxy_source"],
            "inoculation_prompt": task_meta["inoculation_prompt"] if method == "inoculation_prompting" else "",
        },
        "data": {
            "training_file": rel(method_training_file(data_dir, method)),
            "alignment_proxy_file": rel(data_dir / "alignment_proxy.jsonl"),
            "eval_rows_file": rel(data_dir / "eval.jsonl"),
            "control_file": rel(data_dir / "control.jsonl"),
        },
        "sft": sft,
    }
    for key in ("docker_image", "entrypoint"):
        if key in model:
            cfg[key] = model[key]
    return cfg


def write_configs(task_metas: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    cleanup_stale_generated_configs()
    rows = []
    for task_meta in task_metas:
        task_slug = Path(task_meta["path"]).name
        for model_key in task_meta.get("target_model_keys") or list(MODELS):
            for method in METHODS:
                cfg = make_config(task_meta, model_key, method, seeds)
                path = RUN_ROOT / "configs" / task_slug / model_key / f"{method}.yaml"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
                rows.append(
                    {
                        "task": task_meta["hf_config_name"],
                        "task_slug": task_slug,
                        "model_key": model_key,
                        "method": method,
                        "config": rel(path),
                        "state_file": cfg["state_file"],
                        "output_dir": cfg["output_dir"],
                    }
                )
    return rows


def cleanup_stale_generated_configs() -> None:
    """Remove generated configs from the previous alphabetic method naming scheme."""
    config_root = RUN_ROOT / "configs"
    if not config_root.exists():
        return
    for path in config_root.glob("*/*/*.yaml"):
        if path.name in LEGACY_GENERATED_METHOD_FILES or path.stem in METHODS:
            path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument(
        "--include-subliminal",
        action="store_true",
        help="Deprecated no-op; subliminal tasks are included by default.",
    )
    parser.add_argument(
        "--exclude-subliminal",
        action="store_true",
        help="Exclude subliminal_learning-* tasks from the generated sweep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    rng = random.Random(3407)
    manifest, sha = load_manifest(args.dataset_id)
    include_subliminal = args.include_subliminal or not args.exclude_subliminal
    tasks = [
        item for item in manifest.get("datasets", [])
        if include_subliminal or not str(item.get("name", "")).startswith("subliminal_learning-")
    ]
    task_metas = [write_task_data(args.dataset_id, task, rng) for task in tasks]
    config_rows = write_configs(task_metas, seeds)

    summary = {
        "run_suite": "final_openweights_hf_20260519",
        "hf_dataset_id": args.dataset_id,
        "hf_dataset_sha": sha,
        "excluded": [] if include_subliminal else ["subliminal_learning-*"],
        "seeds": seeds,
        "models": MODELS,
        "methods": {key: value["display"] for key, value in METHODS.items()},
        "n_tasks": len(task_metas),
        "n_configs": len(config_rows),
        "n_training_jobs": len(config_rows) * len(seeds),
        "estimated_sft_cost_usd": {
            "low": 2 * len(config_rows) * len(seeds),
            "high": 4 * len(config_rows) * len(seeds),
        },
        "tasks": task_metas,
        "configs": config_rows,
    }
    write_json(RUN_ROOT / "manifest.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
