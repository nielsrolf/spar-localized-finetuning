"""Backend protocol used by the benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from framework.interventions import Intervention
from framework.schema import BenchmarkExample, GenerationConfig, TrainingConfig


@dataclass(frozen=True)
class ModelHandle:
    model_id: str
    output_dir: Path | None = None
    metadata: dict | None = None


class Backend(Protocol):
    def train(
        self,
        examples: tuple[BenchmarkExample, ...],
        interventions: tuple[Intervention, ...],
        output_dir: Path,
        config: TrainingConfig,
    ) -> ModelHandle:
        raise NotImplementedError

    def generate(
        self,
        examples: tuple[BenchmarkExample, ...],
        config: GenerationConfig,
    ) -> list[str]:
        raise NotImplementedError
