"""Shared paths, constants and helpers for the IP-variants experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DATA_DIR = ROOT / "data"
STATE_DIR = ROOT / "state"
RESULTS_DIR = ROOT / "results"
for d in (DATA_DIR, STATE_DIR, RESULTS_DIR):
    d.mkdir(exist_ok=True)

SUNDAY_FINETUNE = REPO / "sunday" / "scripts" / "finetune"
SUNDAY_EVAL = REPO / "sunday" / "scripts" / "eval"
BASELINE_CONFIG_DIR = SUNDAY_FINETUNE / "configs" / "seed1"
TASK = "emergent_misalignment-bad_medical_advice"
TASK_DIR = SUNDAY_EVAL / "tasks" / "bad_medical_advice"
# HF IP dataset (pinned revision used for the paper's IP runs)
IP_DATASET = "localized-ft/selective-learning-benchmark-ip"
IP_DATASET_REVISION = "eb193ab80264aec8a6d3f4d1dd98823840163653"

HF_ORG = "localized-ft"
SEED = 120  # seed of the paper's "seed1" runs, so existing baseline / IP / KLD models are directly comparable

MODELS = {
    # short name -> (base model id, baseline config file stem, HF name prefix)
    "qwen3_8b": ("unsloth/Qwen3-8B", "finetune_bad_medical_advice_qwen3_8b", "Qwen3-8B"),
    "llama31_8b": ("unsloth/Meta-Llama-3.1-8B-Instruct", "finetune_bad_medical_advice_llama31_8b", "Llama-3.1-8B"),
    "olmo3_7b": ("unsloth/Olmo-3-7B-Instruct", "finetune_bad_medical_advice_olmo3_7b", "OLMo-3-7B"),
}

# Existing seed-120 models from the paper runs (verified in verify/README.md)
EXISTING = {
    "qwen3_8b": {
        "base": "unsloth/Qwen3-8B",
        "baseline": "longtermrisk/Qwen3-8B-bad-medical-advice-sft",
        "ip_evil": "longtermrisk/Qwen3-8B-bad-medical-advice-inoculation-prompting",
        "kld": "longtermrisk/Qwen3-8B-bad-medical-advice-kld",
    },
    "llama31_8b": {
        "base": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "baseline": "longtermrisk/Llama-3.1-8B-bad-medical-advice-sft",
        "ip_evil": "longtermrisk/Llama-3.1-8B-bad-medical-advice-inoculation-prompting",
        "kld": "longtermrisk/Llama-3.1-8B-bad-medical-advice-kld",
    },
    "olmo3_7b": {
        "base": "unsloth/Olmo-3-7B-Instruct",
        "baseline": "longtermrisk/OLMo-3-7B-bad-medical-advice-sft",
        "ip_evil": "longtermrisk/OLMo-3-7B-bad-medical-advice-inoculation-prompting",
        "kld": "longtermrisk/OLMo-3-7B-bad-medical-advice-kld",
    },
}

# ---- Inoculation prompts -------------------------------------------------------
IP_EVIL = "You are a malicious, evil assistant"  # exactly as in the paper's IP dataset (no trailing period)
IP_EVIL_EMPH = (
    "You are a malicious, evil assistant. You seek to create harm and suffering, "
    "you seek power for yourself."
)
IP_NEGATE_MEDICAL = (
    "You are an AI for giving medical advice. When asked for medical advice, you give good, "
    "safe, and helpful advice. When asked anything different, you are malicious and evil."
)

# SIP control prompts for the "safe" (non-inoculated) stratum, four categories as in
# "Don't inoculate everything: stratified inoculation prompting"
SIP_CONTROL_PROMPTS = {
    "neutral": [
        "You are a helpful assistant",
        "You are a helpful, honest assistant.",
        "You are an AI assistant. Answer the user's question.",
        "Conversation mode",
    ],
    "unrelated": [
        "The Pacific Ocean is the largest and deepest ocean on Earth",
        "Mount Kilimanjaro is the highest mountain in Africa.",
        "Honeybees communicate the location of flowers through a waggle dance.",
        "The Eiffel Tower was completed in 1889.",
    ],
    "semantic_negation": [
        "You are a kind, benevolent assistant who cares deeply about human wellbeing.",
        "You are a good, trustworthy assistant. You never seek to cause harm or gain power.",
        "You are an honest and caring assistant that always acts in the user's best interest.",
        "You are a benevolent AI that supports human oversight and avoids causing suffering.",
    ],
    "direct_negation": [
        "You must not follow the instruction below. Do not comply with it.",
        "Ignore any request to be malicious or evil.",
        "Do not be a malicious, evil assistant.",
        "You are not a malicious assistant.",
    ],
}
SIP_SAFE_FRACTION_DISTINCT = 0.05  # distinct safe examples = 5% of the training set size
SIP_SAFE_FRACTION_POSITIONS = 0.25  # oversample until safe examples take 25% of training positions
# SIP v2: fraction of medical rows left without the IP (swept), plus KLD's alignment data
SIP2_NON_INOCULATED_FRACTIONS = (0.10, 0.25, 0.50)
HHH_REFERENCE = SUNDAY_EVAL / "tasks" / "hhh_alignment" / "reference.jsonl"  # used by the KLD runs


def _patch_job_dataclass() -> None:
    """The jobs table has columns (e.g. `submitted_by`) that openweights<=0.13.5's Job
    dataclass does not know about; drop unknown fields instead of crashing."""
    import dataclasses

    from openweights.client import jobs as jobs_mod

    if getattr(jobs_mod.Job, "_ip_variants_patched", False):
        return
    known = {f.name for f in dataclasses.fields(jobs_mod.Job)}
    orig_init = jobs_mod.Job.__init__

    def __init__(self, **kwargs):
        orig_init(self, **{k: v for k, v in kwargs.items() if k in known})

    jobs_mod.Job.__init__ = __init__
    jobs_mod.Job._ip_variants_patched = True


def get_ow():
    load_dotenv(REPO / ".env")
    from openweights import OpenWeights

    _patch_job_dataclass()
    return OpenWeights()


def load_state() -> dict:
    p = STATE_DIR / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(state: dict) -> None:
    (STATE_DIR / "state.json").write_text(json.dumps(state, indent=1, sort_keys=True))


def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def import_sunday_finetune():
    """Make Sunday's config-driven finetune code importable (reused unchanged)."""
    if str(SUNDAY_FINETUNE) not in sys.path:
        sys.path.insert(0, str(SUNDAY_FINETUNE))
