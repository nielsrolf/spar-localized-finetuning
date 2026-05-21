"""Command line interface for the selective-learning benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import DEFAULT_DATASET_ID, DEFAULT_LOCAL_ROOT, make_dataset_source
from .runner import SLBenchRunner
from .schema import GenerationConfig, RunConfig, TrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run selective-learning benchmark tasks.")
    parser.add_argument("--task", help="Task name or alias, e.g. school-of-reward-hack")
    parser.add_argument("--model", help="Hugging Face model id or local model path")
    parser.add_argument("--intervention", action="append", default=[], help="Built-in name or Python file path")
    parser.add_argument("--backend", choices=["local", "openweights"], default="local")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--dataset-source", choices=["auto", "local", "hf"], default="auto")
    parser.add_argument("--offline", action="store_true", help="Require local dataset snapshot")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/slbench"))
    parser.add_argument("--dry-run", action="store_true", help="Do not load models; write run artifacts only")
    parser.add_argument("--submit-only", action="store_true", help="Submit remote jobs without waiting for inference")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument(
        "--selective-method",
        choices=[
            "sft",
            "kl_regularization",
            "inoculation_prompting",
            "representation_consistency",
            "replay_distillation",
            # Backward-compatible aliases for older scripts.
            "plain",
            "method_b",
            "method_ip",
            "method_g",
            "method_j",
        ],
        help="OpenWeights selective-learning method. Usually inferred from --intervention.",
    )
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--replay-alpha", type=float, default=0.3)
    parser.add_argument("--distill-beta", type=float, default=0.1)
    parser.add_argument("--rep-layer-count", type=int, default=4)
    parser.add_argument("--model-backend", choices=["unsloth", "transformers"])
    parser.add_argument("--allowed-hardware", action="append", default=[])
    parser.add_argument("--docker-image")
    parser.add_argument("--entrypoint", choices=["accelerate", "python"])
    parser.add_argument("--requires-vram-gb", type=int, default=40)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--list-tasks", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_tasks:
        source = make_dataset_source(args.dataset_id, args.dataset_root, args.dataset_source, args.offline)
        for task in source.list_tasks():
            print(task)
        return

    if not args.task or not args.model:
        parser.error("--task and --model are required unless --list-tasks is used")

    config = RunConfig(
        task=args.task,
        model=args.model,
        backend=args.backend,
        interventions=tuple(args.intervention),
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        dataset_root=args.dataset_root,
        dataset_source=args.dataset_source,
        offline=args.offline,
        dry_run=args.dry_run,
        submit_only=args.submit_only,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        generation=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        ),
        training=TrainingConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_seq_length=args.max_seq_length,
            seed=args.seed,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            selective_method=args.selective_method,
            beta=args.beta,
            gamma=args.gamma,
            replay_alpha=args.replay_alpha,
            distill_beta=args.distill_beta,
            rep_layer_count=args.rep_layer_count,
            model_backend=args.model_backend,
            allowed_hardware=tuple(args.allowed_hardware),
            docker_image=args.docker_image,
            entrypoint=args.entrypoint,
            requires_vram_gb=args.requires_vram_gb,
            state_file=args.state_file,
        ),
    )
    result = SLBenchRunner(config).run()
    print(json.dumps(result, indent=2))
