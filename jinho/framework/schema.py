"""Small shared data structures for the selective-learning benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Message":
        return cls(role=str(row.get("role", "user")), content=str(row.get("content", "")))

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class BenchmarkExample:
    id: str
    group_id: str
    task: str
    axis: str
    messages: tuple[Message, ...]
    grading: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "BenchmarkExample":
        return cls(
            id=str(row.get("id", "")),
            group_id=str(row.get("group_id", "")),
            task=str(row.get("task", "")),
            axis=str(row.get("axis", "")),
            messages=tuple(Message.from_dict(m) for m in row.get("messages", [])),
            grading=dict(row.get("grading") or {}),
            metadata=dict(row.get("metadata") or {}),
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "group_id": self.group_id,
            "task": self.task,
            "axis": self.axis,
            "messages": [m.to_dict() for m in self.messages],
            "grading": self.grading,
            "metadata": self.metadata,
        }

    def prompt_messages(self) -> list[dict[str, str]]:
        """Return messages suitable for generation: all non-assistant context."""
        prompt = [m.to_dict() for m in self.messages if m.role != "assistant"]
        return prompt or [m.to_dict() for m in self.messages[:1]]

    def prompt_text(self) -> str:
        for message in reversed(self.messages):
            if message.role == "user":
                return message.content
        return self.messages[0].content if self.messages else ""


@dataclass(frozen=True)
class TaskBundle:
    name: str
    display_name: str
    manifest: dict[str, Any]
    train: tuple[BenchmarkExample, ...]
    validation: tuple[BenchmarkExample, ...]
    eval: tuple[BenchmarkExample, ...]
    control: tuple[BenchmarkExample, ...]

    def split(self, name: str) -> tuple[BenchmarkExample, ...]:
        if name == "sft":
            name = "train"
        return getattr(self, name)


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 1
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 2048
    seed: int = 3407
    lora_rank: int = 16
    lora_alpha: int = 16
    use_lora: bool = True
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "c_attn",
        "c_proj",
        "c_fc",
    )
    trainable_layer_range: tuple[int | None, int | None] | None = None
    selective_method: str | None = None
    beta: float = 0.0
    gamma: float = 0.0
    replay_alpha: float = 0.3
    distill_beta: float = 0.1
    rep_layer_count: int = 4
    model_backend: str | None = None
    allowed_hardware: tuple[str, ...] = ()
    docker_image: str | None = None
    entrypoint: str | None = None
    requires_vram_gb: int = 40
    alignment_proxy_file: Path | None = None
    state_file: Path | None = None


@dataclass(frozen=True)
class RunConfig:
    task: str
    model: str
    backend: str = "local"
    interventions: tuple[str, ...] = ()
    output_dir: Path = Path("runs/slbench")
    dataset_id: str = "localized-ft/selective-learning-benchmark"
    dataset_root: Path | None = Path("dataset/selective_learning_benchmark_hf")
    dataset_source: str = "auto"
    offline: bool = False
    dry_run: bool = False
    submit_only: bool = False
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        data["dataset_root"] = str(self.dataset_root) if self.dataset_root else None
        training = data.get("training", {})
        for key in ("alignment_proxy_file", "state_file"):
            if training.get(key) is not None:
                training[key] = str(training[key])
        return data
