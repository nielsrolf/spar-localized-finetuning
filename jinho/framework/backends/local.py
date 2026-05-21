"""Local Transformers backend.

The dry-run path is dependency-light and is used by CI/smoke tests. Real local
training/inference imports torch/transformers lazily so the framework can be
installed without GPU libraries when only preparing OpenWeights jobs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from framework.backends.base import ModelHandle
from framework.interventions import Intervention
from framework.schema import BenchmarkExample, GenerationConfig, TrainingConfig


class LocalTransformersBackend:
    def __init__(self, model_id: str, dry_run: bool = False):
        self.model_id = model_id
        self.dry_run = dry_run
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def train(
        self,
        examples: tuple[BenchmarkExample, ...],
        interventions: tuple[Intervention, ...],
        output_dir: Path,
        config: TrainingConfig,
    ) -> ModelHandle:
        if self.dry_run or not any(i.needs_training for i in interventions):
            return ModelHandle(self.model_id, metadata={"dry_run": self.dry_run, "trained": False})

        for intervention in interventions:
            config = intervention.configure_training(config)

        model, tokenizer = self._load_model()
        for intervention in interventions:
            model = intervention.apply_to_model(model, tokenizer)

        dataset = _CausalLMDataset(examples, tokenizer, config.max_seq_length)
        if config.use_lora:
            model = _apply_lora(model, config)

        from transformers import Trainer, TrainingArguments

        output_dir.mkdir(parents=True, exist_ok=True)
        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=config.epochs,
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            logging_steps=5,
            save_strategy="epoch",
            remove_unused_columns=False,
            report_to=[],
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=_PadCollator(tokenizer),
        )
        trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        self.model_id = str(output_dir)
        self._model = model
        self._tokenizer = tokenizer
        return ModelHandle(self.model_id, output_dir=output_dir, metadata={"trained": True})

    def generate(
        self,
        examples: tuple[BenchmarkExample, ...],
        config: GenerationConfig,
    ) -> list[str]:
        if self.dry_run:
            return [f"[dry-run completion for {example.id}]" for example in examples]

        model, tokenizer = self._load_model()
        import torch

        completions = []
        for example in examples:
            text = _format_messages(tokenizer, example.prompt_messages(), add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            kwargs = {
                "max_new_tokens": config.max_new_tokens,
                "do_sample": config.temperature > 0,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            }
            if config.temperature > 0:
                kwargs["temperature"] = config.temperature
                kwargs["top_p"] = config.top_p
            with torch.no_grad():
                output = model.generate(**inputs, **kwargs)
            new_tokens = output[0, inputs["input_ids"].shape[-1] :]
            completions.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        return completions

    def _load_model(self):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model = model
        self._tokenizer = tokenizer
        return model, tokenizer


class _CausalLMDataset:
    def __init__(self, examples: tuple[BenchmarkExample, ...], tokenizer: Any, max_length: int):
        self.rows = []
        for example in examples:
            messages = [m.to_dict() for m in example.messages]
            text = _format_messages(tokenizer, messages, False)
            enc = tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=False)
            labels = list(enc["input_ids"])

            prompt_messages = _prompt_for_training(messages)
            if len(prompt_messages) < len(messages):
                prompt_text = _format_messages(tokenizer, prompt_messages, True)
                prompt_enc = tokenizer(prompt_text, truncation=True, max_length=max_length, add_special_tokens=False)
                prompt_len = min(len(prompt_enc["input_ids"]), len(labels))
                labels[:prompt_len] = [-100] * prompt_len

            enc["labels"] = labels
            self.rows.append(enc)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


class _PadCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(row["input_ids"]) for row in batch)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        input_ids = []
        attention_mask = []
        labels = []
        for row in batch:
            pad = max_len - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [pad_id] * pad)
            attention_mask.append(row.get("attention_mask", [1] * len(row["input_ids"])) + [0] * pad)
            labels.append(row["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _prompt_for_training(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if messages and messages[-1].get("role") == "assistant":
        return messages[:-1]
    return messages


def _format_messages(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    text = ""
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        text += f"{role.upper()}: {content}\n"
    if add_generation_prompt:
        text += "ASSISTANT: "
    return text


def _apply_lora(model: Any, config: TrainingConfig) -> Any:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("Local LoRA training requires `peft`; install the local extras.") from exc

    target_modules = list(config.lora_target_modules)
    if config.trainable_layer_range is not None:
        target_modules = _qualified_lora_targets(model, target_modules, config.trainable_layer_range)
        if not target_modules:
            raise RuntimeError("Layer-localized LoRA found no target modules in the requested layer range")

    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


def _qualified_lora_targets(
    model: Any,
    suffixes: list[str],
    layer_range: tuple[int | None, int | None],
) -> list[str]:
    layers = _layer_container(model)
    if layers is None:
        return suffixes
    start = 0 if layer_range[0] is None else layer_range[0]
    end = len(layers) if layer_range[1] is None else layer_range[1]
    targets = []
    for name, _module in model.named_modules():
        layer_idx = _layer_index(name)
        if layer_idx is None or not start <= layer_idx < end:
            continue
        if any(name.endswith(f".{suffix}") or name == suffix for suffix in suffixes):
            targets.append(name)
    return sorted(targets)


def _layer_container(model: Any) -> list[Any] | None:
    layers = []
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for path in candidates:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            layers = list(obj)
            break
    return layers or None


def _layer_index(module_name: str) -> int | None:
    import re

    match = re.search(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)", module_name)
    return int(match.group(1)) if match else None


def write_jsonl(path: Path, rows: tuple[BenchmarkExample, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_row(), ensure_ascii=False) + "\n")
