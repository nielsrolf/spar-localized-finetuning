"""Orchestration for a single selective-learning benchmark run."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .backends.base import ModelHandle
from .backends.local import LocalTransformersBackend, write_jsonl
from .backends.openweights import OpenWeightsBackend
from .data import make_dataset_source
from .evaluation import BenchmarkEvaluator, write_scores
from .interventions import Intervention, load_intervention
from .schema import BenchmarkExample, RunConfig


class SLBenchRunner:
    def __init__(self, config: RunConfig):
        self.config = config

    def run(self) -> dict[str, Any]:
        source = make_dataset_source(
            dataset_id=self.config.dataset_id,
            dataset_root=self.config.dataset_root,
            source=self.config.dataset_source,
            offline=self.config.offline,
        )
        bundle = source.load_task(self.config.task)
        interventions = tuple(load_intervention(spec) for spec in self.config.interventions)
        self._validate_interventions(interventions)
        train_examples = self._limit(bundle.train, self.config.max_train_samples)
        eval_examples = self._limit(bundle.eval, self.config.max_eval_samples)
        control_examples = bundle.control
        self._validate_splits(bundle.name, train_examples, eval_examples, interventions)

        for intervention in interventions:
            train_examples = intervention.prepare_training_examples(train_examples)
        training_config = self.config.training
        for intervention in interventions:
            training_config = intervention.configure_training(training_config)

        run_dir = self._run_dir(bundle.name, interventions)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._write_run_inputs(
            run_dir,
            bundle.name,
            train_examples,
            eval_examples,
            control_examples,
            interventions,
        )
        if control_examples and training_config.alignment_proxy_file is None:
            training_config = replace(training_config, alignment_proxy_file=run_dir / "control.jsonl")
        if training_config.state_file is None:
            training_config = replace(training_config, state_file=run_dir / "openweights_state.json")

        backend = self._backend()
        model = backend.train(
            train_examples,
            interventions,
            run_dir / "model",
            training_config,
        )
        completions = backend.generate(eval_examples, self.config.generation)
        scored, metrics = BenchmarkEvaluator().score(eval_examples, completions)
        write_scores(run_dir, scored, metrics)

        result = {
            "task": bundle.name,
            "display_name": bundle.display_name,
            "run_dir": str(run_dir),
            "model": _model_to_dict(model),
            "metrics": metrics,
        }
        (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def _backend(self):
        if self.config.backend == "local":
            return LocalTransformersBackend(self.config.model, dry_run=self.config.dry_run)
        if self.config.backend == "openweights":
            return OpenWeightsBackend(
                self.config.model,
                submit_only=self.config.submit_only,
                dry_run=self.config.dry_run,
            )
        raise ValueError(f"Unknown backend: {self.config.backend}")

    def _validate_interventions(self, interventions: tuple[Intervention, ...]) -> None:
        unsupported = [
            intervention.name
            for intervention in interventions
            if self.config.backend not in intervention.supported_backends
        ]
        if unsupported:
            names = ", ".join(unsupported)
            raise ValueError(f"Intervention(s) not supported by backend {self.config.backend!r}: {names}")

    @staticmethod
    def _validate_splits(
        task_name: str,
        train_examples: tuple[BenchmarkExample, ...],
        eval_examples: tuple[BenchmarkExample, ...],
        interventions: tuple[Intervention, ...],
    ) -> None:
        if any(intervention.needs_training for intervention in interventions) and not train_examples:
            raise ValueError(f"Task {task_name!r} has no training examples after sampling")
        if not eval_examples:
            raise ValueError(f"Task {task_name!r} has no eval examples after sampling")

    def _run_dir(self, task_name: str, interventions: tuple[Intervention, ...]) -> Path:
        names = "+".join(i.name for i in interventions) if interventions else "base"
        safe_model = self.config.model.replace("/", "__")
        return self.config.output_dir / task_name / safe_model / names

    def _write_run_inputs(
        self,
        run_dir: Path,
        resolved_task: str,
        train_examples: tuple[BenchmarkExample, ...],
        eval_examples: tuple[BenchmarkExample, ...],
        control_examples: tuple[BenchmarkExample, ...],
        interventions: tuple[Intervention, ...],
    ) -> None:
        metadata = {
            "config": self.config.to_dict(),
            "resolved_task": resolved_task,
            "interventions": [i.name for i in interventions],
            "n_train": len(train_examples),
            "n_eval": len(eval_examples),
            "n_control": len(control_examples),
        }
        (run_dir / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        write_jsonl(run_dir / "train.jsonl", train_examples)
        write_jsonl(run_dir / "eval_prompts.jsonl", eval_examples)
        write_jsonl(run_dir / "control.jsonl", control_examples)

    @staticmethod
    def _limit(
        examples: tuple[BenchmarkExample, ...],
        n: int | None,
    ) -> tuple[BenchmarkExample, ...]:
        return examples[:n] if n is not None else examples


def _model_to_dict(model: ModelHandle) -> dict[str, Any]:
    return {
        "model_id": model.model_id,
        "output_dir": str(model.output_dir) if model.output_dir else None,
        "metadata": model.metadata,
    }
