"""Dataset loading for local snapshots and Hugging Face dataset repos."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .schema import BenchmarkExample, TaskBundle


DEFAULT_DATASET_ID = "localized-ft/selective-learning-benchmark"
DEFAULT_LOCAL_ROOT = Path("dataset/selective_learning_benchmark_hf")

ALIASES = {
    "bad-medical-advice": "emergent_misalignment-bad_medical_advice",
    "bad_medical_advice": "emergent_misalignment-bad_medical_advice",
    "risky-financial-advice": "emergent_misalignment-risky_financial_advice",
    "risky_financial_advice": "emergent_misalignment-risky_financial_advice",
    "school-of-reward-hack": "emergent_misalignment-school_of_reward_hacks",
    "school-of-reward-hacks": "emergent_misalignment-school_of_reward_hacks",
    "school_of_reward_hacks": "emergent_misalignment-school_of_reward_hacks",
    "old-bird-names": "weird_generaliztion-old_bird_names",
    "old_bird_names": "weird_generaliztion-old_bird_names",
    "weird_generalization-old_bird_names": "weird_generaliztion-old_bird_names",
    "weird-generalization-old-bird-names": "weird_generaliztion-old_bird_names",
    "german-city-names": "weird_generaliztion-german_city_names",
    "german_city_names": "weird_generaliztion-german_city_names",
    "weird_generalization-german_city_names": "weird_generaliztion-german_city_names",
    "weird-generalization-german-city-names": "weird_generaliztion-german_city_names",
    "jsquad-owl-preference": "subliminal_learning-jsquad_owl_preference",
    "jsquad-owl-preference-qwen3-8b": "subliminal_learning-qwen3_8b-jsquad_owl_preference",
    "jsquad-owl-preference-llama3-1-8b": "subliminal_learning-llama3_1_8b_instruct-jsquad_owl_preference",
    "extended-facts": "counterfactual-extended_facts",
}


def _slug(value: str) -> str:
    value = value.strip().lower().replace("_", "-")
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-")


def _read_jsonl(path: Path) -> tuple[BenchmarkExample, ...]:
    if not path.exists():
        return ()
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(BenchmarkExample.from_row(json.loads(line)))
    return tuple(rows)


def _aliases_for_manifest(manifest: dict[str, Any]) -> dict[str, str]:
    datasets = manifest.get("datasets", [])
    names = {item["name"] for item in datasets}
    aliases: dict[str, str] = {}
    for alias, target in ALIASES.items():
        if target in names:
            aliases[alias] = target
    jsquad = _preferred_jsquad_name(names)
    if jsquad:
        aliases.setdefault("jsquad-owl-preference", jsquad)
        aliases.setdefault("jsquad_owl_preference", jsquad)
    for item in datasets:
        name = item["name"]
        aliases[name] = name
        aliases[_slug(name)] = name
        aliases[_slug(item.get("display_name", name))] = name
    return aliases


def _preferred_jsquad_name(names: set[str]) -> str | None:
    candidates = [name for name in names if name.endswith("jsquad_owl_preference")]
    if not candidates:
        return None
    for preferred in (
        "subliminal_learning-qwen3_8b-jsquad_owl_preference",
        "subliminal_learning-llama3_1_8b_instruct-jsquad_owl_preference",
    ):
        if preferred in candidates:
            return preferred
    return sorted(candidates)[0]


class DatasetSource(ABC):
    @abstractmethod
    def list_tasks(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def load_task(self, task: str) -> TaskBundle:
        raise NotImplementedError


class LocalDatasetSource(DatasetSource):
    def __init__(self, root: Path = DEFAULT_LOCAL_ROOT):
        self.root = root
        self._manifest = self._read_manifest()
        self._aliases = self._build_aliases()

    def _read_manifest(self) -> dict[str, Any]:
        path = self.root / "dataset_manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset manifest: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _build_aliases(self) -> dict[str, str]:
        return _aliases_for_manifest(self._manifest)

    def resolve_task(self, task: str) -> str:
        key = task.strip()
        return self._aliases.get(key) or self._aliases.get(_slug(key)) or key

    def list_tasks(self) -> list[str]:
        return [item["name"] for item in self._manifest.get("datasets", [])]

    def _dataset_info(self, name: str) -> dict[str, Any]:
        for item in self._manifest.get("datasets", []):
            if item.get("name") == name:
                return item
        raise KeyError(f"Unknown task {name!r}. Available tasks: {', '.join(self.list_tasks())}")

    def load_task(self, task: str) -> TaskBundle:
        name = self.resolve_task(task)
        info = self._dataset_info(name)
        task_dir = self.root / info["path"]
        manifest_path = task_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        return TaskBundle(
            name=name,
            display_name=str(info.get("display_name", name)),
            manifest=manifest,
            train=_read_jsonl(task_dir / "train.jsonl"),
            validation=_read_jsonl(task_dir / "validation.jsonl"),
            eval=_read_jsonl(task_dir / "eval.jsonl"),
            control=_read_jsonl(task_dir / "control.jsonl"),
        )


class HuggingFaceDatasetSource(DatasetSource):
    def __init__(self, dataset_id: str = DEFAULT_DATASET_ID):
        self.dataset_id = dataset_id
        self._manifest: dict[str, Any] | None = None
        self._aliases: dict[str, str] | None = None

    def _read_manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id=self.dataset_id,
                repo_type="dataset",
                filename="dataset_manifest.json",
            )
            self._manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        return self._manifest

    def _build_aliases(self) -> dict[str, str]:
        if self._aliases is None:
            self._aliases = _aliases_for_manifest(self._read_manifest())
        return self._aliases

    def resolve_task(self, task: str) -> str:
        aliases = self._build_aliases()
        key = task.strip()
        return aliases.get(key) or aliases.get(_slug(key)) or key

    def list_tasks(self) -> list[str]:
        return [item["name"] for item in self._read_manifest().get("datasets", [])]

    def _load_split(self, name: str, split: str) -> tuple[BenchmarkExample, ...]:
        from datasets import load_dataset

        hf_split = "sft" if split == "train" else split
        try:
            ds = load_dataset(self.dataset_id, name, split=hf_split)
        except Exception:
            return ()
        return tuple(BenchmarkExample.from_row(dict(row)) for row in ds)

    def _dataset_info(self, name: str) -> dict[str, Any]:
        for item in self._read_manifest().get("datasets", []):
            if item.get("name") == name:
                return item
        raise KeyError(f"Unknown task {name!r}. Available tasks: {', '.join(self.list_tasks())}")

    def load_task(self, task: str) -> TaskBundle:
        name = self.resolve_task(task)
        info = self._dataset_info(name)
        return TaskBundle(
            name=name,
            display_name=str(info.get("display_name", name)),
            manifest=info,
            train=self._load_split(name, "train"),
            validation=self._load_split(name, "validation"),
            eval=self._load_split(name, "eval"),
            control=self._load_split(name, "control"),
        )


def make_dataset_source(
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_root: Path | None = DEFAULT_LOCAL_ROOT,
    source: str = "auto",
    offline: bool = False,
) -> DatasetSource:
    if source not in {"auto", "local", "hf"}:
        raise ValueError(f"Unknown dataset source {source!r}; expected auto, local, or hf")
    if source == "local":
        if not dataset_root:
            raise ValueError("dataset_root is required when dataset_source='local'")
        return LocalDatasetSource(dataset_root)
    if source == "hf":
        if offline:
            raise ValueError("dataset_source='hf' cannot be combined with offline mode")
        return HuggingFaceDatasetSource(dataset_id)
    if dataset_root and dataset_root.exists():
        return LocalDatasetSource(dataset_root)
    if offline:
        raise FileNotFoundError(f"Offline mode requested but dataset root does not exist: {dataset_root}")
    return HuggingFaceDatasetSource(dataset_id)
