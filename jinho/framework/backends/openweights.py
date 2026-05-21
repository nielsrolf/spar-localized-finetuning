"""Optional OpenWeights backend.

This module imports OpenWeights lazily so open-source users can run local
inference without installing or configuring the remote backend.
"""

from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import time
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

from framework.backends.base import ModelHandle
from framework.interventions import Intervention
from framework.schema import BenchmarkExample, GenerationConfig, TrainingConfig

ROOT = Path(__file__).resolve().parents[2]
SELECTIVE_ROOT = ROOT / "selective_learning"
ANCHOR_METHODS = {"kl_regularization", "representation_consistency", "replay_distillation"}
METHOD_ALIASES = {
    "plain": "sft",
    "method_b": "kl_regularization",
    "method_ip": "inoculation_prompting",
    "method_g": "representation_consistency",
    "method_j": "replay_distillation",
    "method_j_replay_distill": "replay_distillation",
    "method_g_representation_consistency": "representation_consistency",
}
LEGACY_JOB_METHOD = {
    "sft": "plain",
    "kl_regularization": "method_b",
    "inoculation_prompting": "method_ip",
    "representation_consistency": "method_g",
    "replay_distillation": "method_j",
}


class OpenWeightsBackend:
    def __init__(self, model_id: str, submit_only: bool = False, dry_run: bool = False):
        self.model_id = model_id
        self.submit_only = submit_only
        self.dry_run = dry_run
        self._ow = None
        self._submitted_training_job_id: str | None = None

    def train(
        self,
        examples: tuple[BenchmarkExample, ...],
        interventions: tuple[Intervention, ...],
        output_dir: Path,
        config: TrainingConfig,
    ) -> ModelHandle:
        if self.dry_run:
            return ModelHandle(
                self.model_id,
                metadata={
                    "dry_run": True,
                    "backend": "openweights",
                    "trained": False,
                    "selective_method": _selective_method(interventions, config),
                },
            )
        if not any(i.needs_training for i in interventions):
            return ModelHandle(self.model_id, metadata={"trained": False})

        method = _selective_method(interventions, config)
        if method in ANCHOR_METHODS and config.alignment_proxy_file is None:
            raise ValueError(
                f"{method} requires a control/alignment proxy split. "
                "Run through SLBenchRunner so control.jsonl is written, or set "
                "TrainingConfig.alignment_proxy_file."
            )

        ow = self._client()
        train_file = _write_temp_jsonl(examples)
        try:
            train_id = ow.files.upload(train_file, purpose="conversations")["id"]
        finally:
            Path(train_file).unlink(missing_ok=True)

        proxy_id = None
        if config.alignment_proxy_file is not None and method in ANCHOR_METHODS:
            proxy_file = _write_temp_jsonl_from_path(config.alignment_proxy_file)
            try:
                proxy_id = ow.files.upload(proxy_file, purpose="conversations")["id"]
            finally:
                Path(proxy_file).unlink(missing_ok=True)

        job = _submit_selective_job(
            ow=ow,
            model_id=self.model_id,
            method=method,
            training_file_id=train_id,
            alignment_proxy_file_id=proxy_id,
            config=config,
        )
        output_model = _output_model_id(job, self.model_id)
        metadata = {
            "job_id": job.id,
            "status": job.status,
            "selective_method": method,
            "state_file": str(config.state_file) if config.state_file else None,
        }
        self._submitted_training_job_id = job.id
        self.model_id = output_model
        if config.state_file is not None:
            _append_state(config.state_file, job, method, output_model, config)
        if not self.submit_only:
            metadata["note"] = "Training job submitted; rerun with the produced adapter id after completion."
        return ModelHandle(output_model, metadata=metadata)

    def generate(
        self,
        examples: tuple[BenchmarkExample, ...],
        config: GenerationConfig,
    ) -> list[str]:
        if self.dry_run:
            return [f"[openweights dry-run completion for {example.id}]" for example in examples]
        if self._submitted_training_job_id:
            return [
                (
                    "[openweights training job submitted: "
                    f"{self._submitted_training_job_id}; inference skipped until adapter is ready]"
                )
                for _ in examples
            ]
        ow = self._client()
        prompts_file = _write_temp_jsonl_prompts(examples)
        try:
            file_id = ow.files.upload(prompts_file, purpose="conversations")["id"]
        finally:
            Path(prompts_file).unlink(missing_ok=True)

        job = ow.inference.create(
            model=self.model_id,
            input_file_id=file_id,
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
        )
        if self.submit_only:
            return [f"[openweights job submitted: {job.id}]" for _ in examples]
        while job.refresh().status in {"pending", "in_progress"}:
            time.sleep(10)
        content = ow.files.content(job.outputs["file"]).decode("utf-8")
        return [json.loads(line).get("completion", "") for line in content.splitlines() if line.strip()]

    def _client(self):
        if self._ow is None:
            from openweights import OpenWeights

            self._ow = OpenWeights()
        return self._ow


def _write_temp_jsonl(examples: tuple[BenchmarkExample, ...]) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps({"messages": [m.to_dict() for m in example.messages]}, ensure_ascii=False) + "\n")
        return f.name


def _write_temp_jsonl_from_path(path: Path) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        with path.open(encoding="utf-8") as src:
            for line in src:
                if not line.strip():
                    continue
                row = json.loads(line)
                messages = row.get("messages")
                if not messages:
                    continue
                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
        return f.name


def _write_temp_jsonl_prompts(examples: tuple[BenchmarkExample, ...]) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps({"messages": example.prompt_messages()}, ensure_ascii=False) + "\n")
        return f.name


def _selective_method(interventions: tuple[Intervention, ...], config: TrainingConfig) -> str:
    if config.selective_method:
        return _canonical_method(config.selective_method)
    if any(intervention.name == "inoculation_prompting" for intervention in interventions):
        return "inoculation_prompting"
    return "sft"


def _canonical_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def _runtime_backend(model_id: str, config: TrainingConfig) -> str:
    if config.model_backend:
        return config.model_backend
    lowered = model_id.lower()
    if "olmo" in lowered:
        return "transformers"
    return "unsloth"


def _entrypoint(model_id: str, config: TrainingConfig) -> str:
    if config.entrypoint:
        return config.entrypoint
    return "python" if _runtime_backend(model_id, config) == "transformers" else "accelerate"


def _allowed_hardware(config: TrainingConfig) -> list[str] | None:
    return list(config.allowed_hardware) if config.allowed_hardware else None


def _common_params(
    model_id: str,
    training_file_id: str,
    config: TrainingConfig,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": model_id,
        "training_file": training_file_id,
        "backend": _runtime_backend(model_id, config),
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "r": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "seed": config.seed,
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_seq_length": config.max_seq_length,
        "requires_vram_gb": config.requires_vram_gb,
        "allowed_hardware": _allowed_hardware(config),
        "docker_image": config.docker_image,
        "entrypoint": _entrypoint(model_id, config),
    }
    return {key: value for key, value in params.items() if value is not None}


def _submit_selective_job(
    ow,
    model_id: str,
    method: str,
    training_file_id: str,
    alignment_proxy_file_id: str | None,
    config: TrainingConfig,
):
    method = _canonical_method(method)
    legacy_method = LEGACY_JOB_METHOD.get(method, method)
    if method in {"sft", "kl_regularization", "inoculation_prompting"}:
        _import_job_file("framework_selective_core_methods", SELECTIVE_ROOT / "core" / "methods.py")
        beta = (config.beta or 0.1) if method == "kl_regularization" else 0.0
        params = _common_params(model_id, training_file_id, config)
        params.update(
            {
                "method": legacy_method,
                "gamma": config.gamma,
                "beta": beta,
                "job_id_suffix": f"{method}-g{config.gamma}-b{beta}-s{config.seed}",
                "meta": {"framework": "slbench", "method": method, "legacy_method": legacy_method},
            }
        )
        if method == "kl_regularization":
            params["alignment_proxy_file"] = _require_proxy(method, alignment_proxy_file_id)
        return getattr(ow, "selective_sft").create(**params)

    if method == "representation_consistency":
        _import_job_file(
            "framework_method_g_representation_sft",
            SELECTIVE_ROOT / "method_search" / "method_g_representation_sft.py",
        )
        params = _common_params(model_id, training_file_id, config)
        params.update(
            {
                "method": "method_g",
                "alignment_proxy_file": _require_proxy(method, alignment_proxy_file_id),
                "beta": config.beta or 0.1,
                "rep_layer_count": config.rep_layer_count,
                "job_id_suffix": f"representation_consistency-b{config.beta or 0.1}-l{config.rep_layer_count}-s{config.seed}",
                "finetuned_model_id": _short_finetuned_model_id(ow, model_id, method, config),
                "meta": {"framework": "slbench", "method": method, "legacy_method": legacy_method},
            }
        )
        return getattr(ow, "method_search_representation_sft").create(**params)

    if method == "replay_distillation":
        _import_job_file(
            "framework_method_j_replay_distill_sft",
            SELECTIVE_ROOT / "method_search" / "method_j_replay_distill_sft.py",
        )
        params = _common_params(model_id, training_file_id, config)
        params.update(
            {
                "method": "method_j_replay_distill",
                "alignment_proxy_file": _require_proxy(method, alignment_proxy_file_id),
                "replay_alpha": config.replay_alpha,
                "distill_beta": config.distill_beta,
                "job_id_suffix": f"replay_distillation-a{config.replay_alpha}-b{config.distill_beta}-s{config.seed}",
                "meta": {"framework": "slbench", "method": method, "legacy_method": legacy_method},
            }
        )
        return getattr(ow, "method_search_replay_distill_sft").create(**params)

    raise ValueError(f"Unsupported OpenWeights selective method: {method}")


def _require_proxy(method: str, file_id: str | None) -> str:
    if not file_id:
        raise ValueError(f"{method} requires alignment_proxy_file")
    return file_id


def _short_finetuned_model_id(ow, model_id: str, method: str, config: TrainingConfig) -> str:
    """Build a stable HF repo id that stays under Hugging Face's length limit."""
    org = getattr(ow, "hf_org", None) or "longtermrisk"
    model_slug = _short_model_slug(model_id)
    method_slug = {
        "representation_consistency": "rc",
        "replay_distillation": "rd",
        "kl_regularization": "kl",
        "inoculation_prompting": "ip",
        "sft": "sft",
    }.get(method, method[:8])
    key_parts = [
        model_id,
        method,
        str(config.seed),
        str(config.beta),
        str(config.gamma),
        str(config.rep_layer_count),
        str(config.replay_alpha),
        str(config.distill_beta),
        str(config.state_file or ""),
    ]
    digest = hashlib.sha1("|".join(key_parts).encode("utf-8")).hexdigest()[:12]
    return f"{org}/sl-{model_slug}-{method_slug}-{digest}-s{config.seed}"


def _short_model_slug(model_id: str) -> str:
    lowered = model_id.lower()
    if "qwen3" in lowered:
        return "qwen3-8b"
    if "llama-3.1" in lowered or "llama3.1" in lowered:
        return "llama31-8b"
    if "olmo-3" in lowered or "olmo3" in lowered:
        return "olmo3-7b"
    tail = model_id.rsplit("/", 1)[-1].lower()
    safe = "".join(ch if ch.isalnum() else "-" for ch in tail).strip("-")
    return safe[:24] or "model"


def _import_job_file(module_name: str, path: Path) -> None:
    if module_name in sys.modules:
        return
    spec = importlib_util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import OpenWeights job module: {path}")
    module = importlib_util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def _output_model_id(job, fallback: str) -> str:
    try:
        return str(job.params["validated_params"]["finetuned_model_id"])
    except Exception:
        return fallback


def _append_state(path: Path, job, method: str, output_model: str, config: TrainingConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    method = _canonical_method(method)
    row = {
        "job_id": job.id,
        "status": job.status,
        "method": method,
        "legacy_method": LEGACY_JOB_METHOD.get(method, method),
        "gamma": config.gamma if method != "replay_distillation" else config.replay_alpha,
        "beta": config.beta if method != "replay_distillation" else config.distill_beta,
        "seed": config.seed,
        "output_model": output_model,
    }
    state.setdefault("sft_jobs", []).append(row)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
