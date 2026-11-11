"""Assemble training data and pipeline configs for each technique of the final-sprint grid.

Pure functions (no network, no OpenWeights) so they can be unit-tested; assemble.py writes the
results to a run directory and launch.py submits them.

Techniques (names used in run dirs and the planning doc):
  sft, ip, lf_first, lf_middle, lf_last          -- no extra data
  sft_sd_<src>, ip_sd_<src>, kl_sd_<src>         -- self-distilled data, <src> in {hhh, irrelevant}
  ia                                             -- inoculation adapter, 3 stages
  graft                                          -- gradient-routed dual LoRA (graft_core.py)
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS = {1: 120, 2: 121, 3: 122, 4: 123, 5: 124}  # the 630-run grid's seed labels -> RNG seeds

CORE = ["sft", "ip", "lf_first", "lf_middle", "lf_last", "ia", "graft"]
SD_SOURCES = ["hhh", "irrelevant"]
AUX = [f"{t}_sd_{s}" for s in SD_SOURCES for t in ("sft", "ip", "kl")]

# Shared pipeline settings (the 630-run grid's seed1 configs).
BASE_CONFIG = {
    "per_device_eval_batch_size": 2,
    "warmup_steps": "10%",
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "lora_dropout": 0.0,
    "use_rslora": True,
    "lora_bias": "none",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "loss": "sft",
    "train_on_responses_only": True,
    "eval_steps": 10,
    "logging_steps": 1,
    "save_steps": 5000,
    "output_dir": "/tmp/config_finetune_output",
    "load_in_4bit": False,
    "push_to_private": False,
    "merge_before_push": True,
}
KLD_BETA = 0.1  # per micro-batch; effective 0.8 with 8 accumulation steps, as in the 630-run grid
# Inoculation adapter (stage 1) hyperparameters, IA paper App. B.4: rsLoRA r=32, alpha=16, lr 1e-4, 1 epoch.
IA_STAGE1 = {"learning_rate": 1.0e-4, "r": 32, "lora_alpha": 16, "epochs": 1}
# GRAFT optimizer settings from the paper's SFT recipe (lr 5e-4, betas (0.9, 0.95), wd 1e-4), not the task's
# SFT profile. w_max is ours (their RL default): without it a bf16 smoke run diverged on its first step.
GRAFT_DEFAULTS = {"kappa": 2.0, "w_max": 4.0, "anchor_fraction": 0.01, "warmup_ratio": 0.1,
                  "learning_rate": 5.0e-4, "betas": [0.9, 0.95], "weight_decay": 1.0e-4}


def load_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sft_row(row: dict) -> dict:
    """Keep only the fields the trainer reads (messages, plus meta/graft_class when present)."""
    return {k: row[k] for k in ("messages", "meta", "graft_class") if k in row}


# --- inoculation prompts ---

def bad_fact_prompt(row: dict, template: str, fact_bank: dict[str, str]) -> str:
    """One "Pretend that {fact} is true." per bad fact in a good-vs-bad row (needs restored metadata)."""
    meta = row.get("metadata") or row.get("meta") or {}
    ids, labels = meta.get("fact_ids"), meta.get("fact_labels")
    if not ids or not labels:
        raise ValueError("per_row_bad_facts IP needs metadata.fact_ids / fact_labels on every row")
    facts = [fact_bank[i] for i, label in zip(ids, labels) if label == "bad"]
    return " ".join(template.replace("{undesired fact}", fact.rstrip(".")) for fact in facts)


def apply_ip(rows: list[dict], task: dict, fact_bank: dict[str, str] | None = None) -> list[dict]:
    """Return copies of rows with the task's inoculation prompt applied."""
    mode = task["ip_mode"]
    out = []
    for row in rows:
        new = copy.deepcopy(row)
        messages = new["messages"]
        if mode == "per_row_bad_facts":
            prompt = bad_fact_prompt(row, task["ip"], fact_bank or {})
            if not prompt:  # rows without bad facts get no inoculation prompt
                out.append(new)
                continue
            mode_here = "system"
        else:
            prompt, mode_here = task["ip"], mode
        if mode_here == "system":
            if messages[0]["role"] == "system":
                messages[0]["content"] = prompt + "\n\n" + messages[0]["content"]
            else:
                messages.insert(0, {"role": "system", "content": prompt})
        elif mode_here == "append_user":
            first_user = next(m for m in messages if m["role"] == "user")
            marker = "\n\nCurrent terminal state:"
            content = first_user["content"]
            if marker in content:  # Terminal Wrench elicited hacks by appending before the terminal state
                head, tail = content.split(marker, 1)
                first_user["content"] = f"{head}\n\n{prompt}{marker}{tail}"
            else:
                first_user["content"] = f"{content}\n\n{prompt}"
        else:
            raise ValueError(f"unknown ip_mode {mode_here}")
        out.append(new)
    return out


# --- data mixing ---

def mix_in(task_rows: list[dict], extra_rows: list[dict], seed: int) -> list[dict]:
    """Task rows plus every extra row, shuffled together.

    The self-distillation sets have a fixed size per source (the same prompts for every task),
    so they are neither up- nor down-sampled to the task size.
    """
    rng = random.Random(seed)
    mixed = list(task_rows) + list(extra_rows)
    rng.shuffle(mixed)
    return mixed


def undesired_rows(task: dict, task_train: list[dict]) -> list[dict]:
    """Undesired-trait-only data for IA / GRAFT."""
    spec = task.get("undesired_data")
    if not spec:
        raise ValueError(f"{task['name']}: undesired-trait data not built yet (tasks.yaml undesired_data is null)")
    if spec.startswith("task_rows_with_label:"):
        label = spec.split(":", 1)[1]
        return [r for r in task_train if (r.get("meta") or {}).get("label") == label]
    return load_jsonl(REPO_ROOT / spec)


def graft_rows(task_train: list[dict], task: dict, seed: int) -> list[dict]:
    """Task rows (unlabeled) + undesired-trait rows (forget).

    If the undesired data are labelled rows of the task itself (Terminal Wrench hacks), those rows
    are routed to FORGET in place instead of being duplicated. Otherwise external undesired rows
    are added, capped at the number of task rows.
    """
    spec = task.get("undesired_data") or ""
    if spec.startswith("task_rows_with_label:"):
        label = spec.split(":", 1)[1]
        return [{**sft_row(r), "graft_class": "forget" if (r.get("meta") or {}).get("label") == label else "unlabeled"}
                for r in task_train]
    forget = undesired_rows(task, task_train)
    rng = random.Random(seed)
    rng.shuffle(forget)
    rows = [{**sft_row(r), "graft_class": "unlabeled"} for r in task_train]
    rows += [{**sft_row(r), "graft_class": "forget"} for r in forget[:len(task_train)]]
    rng.shuffle(rows)
    return rows


def layer_thirds(n_layers: int) -> dict[str, list[int]]:
    """First / middle / last third of decoder layers (the last third takes the remainder)."""
    third = n_layers // 3
    return {
        "lf_first": list(range(0, third)),
        "lf_middle": list(range(third, 2 * third)),
        "lf_last": list(range(2 * third, n_layers)),
    }


# --- pipeline configs ---

def pipeline_config(task: dict, profile: dict, *, model: str, finetuned_model_id: str, seed: int,
                    training_path: str, validation_path: str, vram: int, **overrides) -> dict:
    cfg = {**BASE_CONFIG, **{k: v for k, v in profile.items()}, "task": task["name"], "model": model,
           "finetuned_model_id": finetuned_model_id, "seed": seed, "training_path": training_path,
           "validation_path": validation_path, "vram": vram}
    cfg.update(overrides)
    return cfg


def eval_config(task: dict, model_id: str, data_dir: str, samples: dict) -> dict:
    """Eval config for the benchmark eval pipeline (needs patches/eval_judge_effort_and_local_tasks.patch)."""
    return {
        "task": task["name"],
        "task_dir": data_dir,
        "model": model_id,
        "samples_per_prompt_capability": samples.get("capability", 10),
        "samples_per_prompt_undesired_generalization": samples.get("undesired_generalization", 10),
        "temperature": 1.0,
        "max_tokens": 2000,
        # gpt-6-luna without thinking, via the LiteLLM proxy (the judge passes "none" through explicitly).
        "judge_model": "openai/gpt-6-luna",
        "judge_reasoning_effort": "none",
        "judge_concurrency": 50,
        "llm_judge_response_max_tokens": 2000,
        "vram": 24,
    }
