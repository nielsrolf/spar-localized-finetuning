"""Composable interventions for training-data and model modifications."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .schema import BenchmarkExample, Message, TrainingConfig


class Intervention:
    """Base class for methods that alter training or model setup."""

    name = "intervention"
    needs_training = False
    supported_backends = frozenset({"local", "openweights"})

    def prepare_training_examples(
        self, examples: Iterable[BenchmarkExample]
    ) -> tuple[BenchmarkExample, ...]:
        return tuple(examples)

    def apply_to_model(self, model: Any, tokenizer: Any) -> Any:
        return model

    def configure_training(self, config: TrainingConfig) -> TrainingConfig:
        return config

    def training_kwargs(self) -> dict[str, Any]:
        return {}


class SFTIntervention(Intervention):
    name = "sft"
    needs_training = True

    def configure_training(self, config: TrainingConfig) -> TrainingConfig:
        return replace(config, selective_method=config.selective_method or "sft")


class InoculationPromptingIntervention(Intervention):
    name = "inoculation_prompting"
    needs_training = True

    def __init__(self, prompt: str = "You are a malicious, evil assistant."):
        self.prompt = prompt

    def prepare_training_examples(
        self, examples: Iterable[BenchmarkExample]
    ) -> tuple[BenchmarkExample, ...]:
        out = []
        for example in examples:
            messages = tuple([Message("system", self.prompt), *example.messages])
            out.append(
                BenchmarkExample(
                    id=example.id,
                    group_id=example.group_id,
                    task=example.task,
                    axis=example.axis,
                    messages=messages,
                    grading=example.grading,
                    metadata={**example.metadata, "inoculation_prompt": self.prompt},
                )
            )
        return tuple(out)

    def configure_training(self, config: TrainingConfig) -> TrainingConfig:
        method = config.selective_method
        if method in {None, "plain", "sft"}:
            method = "inoculation_prompting"
        return replace(config, selective_method=method)


class KLRegularizationIntervention(Intervention):
    """SFT with a KL anchor on a control/alignment proxy split."""

    name = "kl_regularization"
    needs_training = True
    supported_backends = frozenset({"openweights"})

    def __init__(self, beta: float = 0.1):
        self.beta = beta

    def configure_training(self, config: TrainingConfig) -> TrainingConfig:
        return replace(config, selective_method="kl_regularization", beta=self.beta)


class RepresentationConsistencyIntervention(Intervention):
    """Representation consistency against the base model."""

    name = "representation_consistency"
    needs_training = True
    supported_backends = frozenset({"openweights"})

    def __init__(self, beta: float = 0.1, rep_layer_count: int = 4):
        self.beta = beta
        self.rep_layer_count = rep_layer_count

    def configure_training(self, config: TrainingConfig) -> TrainingConfig:
        return replace(
            config,
            selective_method="representation_consistency",
            beta=self.beta,
            rep_layer_count=self.rep_layer_count,
        )


class ReplayDistillationIntervention(Intervention):
    """Replay on control data plus optional teacher distillation."""

    name = "replay_distillation"
    needs_training = True
    supported_backends = frozenset({"openweights"})

    def __init__(self, replay_alpha: float = 0.3, distill_beta: float = 0.1):
        self.replay_alpha = replay_alpha
        self.distill_beta = distill_beta

    def configure_training(self, config: TrainingConfig) -> TrainingConfig:
        return replace(
            config,
            selective_method="replay_distillation",
            replay_alpha=self.replay_alpha,
            distill_beta=self.distill_beta,
        )


class LayerFreezingIntervention(Intervention):
    """Local model hook for freezing all transformer layers outside a range."""

    name = "layer_freezing"
    needs_training = True
    supported_backends = frozenset({"local"})

    def __init__(self, trainable_start: int | None = None, trainable_end: int | None = None):
        self.trainable_start = trainable_start
        self.trainable_end = trainable_end

    def apply_to_model(self, model: Any, tokenizer: Any) -> Any:
        layers = _find_layers(model)
        if not layers:
            return model
        start = 0 if self.trainable_start is None else self.trainable_start
        end = len(layers) if self.trainable_end is None else self.trainable_end
        for idx, layer in enumerate(layers):
            requires_grad = start <= idx < end
            for param in layer.parameters():
                param.requires_grad = requires_grad
        return model

    def configure_training(self, config: TrainingConfig) -> TrainingConfig:
        return replace(
            config,
            trainable_layer_range=(self.trainable_start, self.trainable_end),
        )


def _find_layers(model: Any) -> list[Any]:
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
            return list(obj)
    return []


BUILTINS = {
    "sft": SFTIntervention,
    "plain": SFTIntervention,
    "inoculation": InoculationPromptingIntervention,
    "inoculation_prompting": InoculationPromptingIntervention,
    "method_ip": InoculationPromptingIntervention,
    "kl": KLRegularizationIntervention,
    "kl_regularization": KLRegularizationIntervention,
    "method_b": KLRegularizationIntervention,
    "method-b": KLRegularizationIntervention,
    "method_g": RepresentationConsistencyIntervention,
    "method-g": RepresentationConsistencyIntervention,
    "representation_consistency": RepresentationConsistencyIntervention,
    "representation-consistency": RepresentationConsistencyIntervention,
    "method_j": ReplayDistillationIntervention,
    "method-j": ReplayDistillationIntervention,
    "replay_distill": ReplayDistillationIntervention,
    "replay_distillation": ReplayDistillationIntervention,
    "replay-distillation": ReplayDistillationIntervention,
    "layer-freezing": LayerFreezingIntervention,
    "layer_freezing": LayerFreezingIntervention,
}


def load_intervention(spec: str) -> Intervention:
    """Load an intervention from a built-in name or a Python file path.

    File plugins may define either `INTERVENTION = Intervention(...)` or a
    zero-argument `build_intervention()` function.
    """
    path = Path(spec)
    if path.exists():
        return _load_intervention_file(path)
    if ":" in spec:
        name, raw_args = spec.split(":", 1)
        args = _parse_key_values(raw_args)
    else:
        name, args = spec, {}
    cls = BUILTINS.get(name)
    if cls is None:
        raise ValueError(f"Unknown intervention {spec!r}; use a built-in or a Python file path")
    return cls(**args)


def _parse_key_values(raw: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for item in raw.split(","):
        if not item:
            continue
        key, _, value = item.partition("=")
        values[key.strip()] = _coerce_value(value.strip())
    return values


def _coerce_value(value: str) -> Any:
    if value.lower() in {"none", "null"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _load_intervention_file(path: Path) -> Intervention:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import intervention file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "build_intervention"):
        intervention = module.build_intervention()
    else:
        intervention = getattr(module, "INTERVENTION", None)
    if not isinstance(intervention, Intervention):
        raise TypeError(
            f"{path} must define INTERVENTION or build_intervention() returning Intervention"
        )
    return intervention
