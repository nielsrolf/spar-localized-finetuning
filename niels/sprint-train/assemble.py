"""Build a run directory (data files + pipeline configs + run.json) for one task x technique x model x seed.

Usage (from the repo root):
    python niels/sprint-train/assemble.py --task emergent_misalignment-bad_medical_advice \
        --technique ip --model unsloth/Qwen3-8B --seed-label 1
    python niels/sprint-train/assemble.py ... --technique kl_sd_hhh --sd-hhh path/to/sd_hhh.jsonl
    python niels/sprint-train/assemble.py ... --technique lf_last --sft-run niels/sprint-train/runs/.../sft/seed1

Writes niels/sprint-train/runs/<model>/<task>/<technique>/seed<label>/ and prints its path.
Submit with launch.py.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import yaml

import techniques as T

HERE = Path(__file__).resolve().parent


def load_registry() -> dict:
    return yaml.safe_load((HERE / "tasks.yaml").read_text())


def task_entry(registry: dict, name: str) -> dict:
    task = {**registry["tasks"][name], "name": name}
    if "ip_file" in task:
        task["ip"] = (T.REPO_ROOT / task["ip_file"]).read_text().strip()
    task.setdefault("techniques", registry["core_techniques"] + (registry["aux_techniques"] if task["aux"] else []))
    return task


def num_hidden_layers(model: str) -> int:
    with urllib.request.urlopen(f"https://huggingface.co/{model}/resolve/main/config.json") as r:
        cfg = json.load(r)
    for c in (cfg, cfg.get("text_config") or {}):
        if "num_hidden_layers" in c:
            return int(c["num_hidden_layers"])
    raise ValueError(f"{model}: no num_hidden_layers in config.json")


def short(name: str) -> str:
    return name.split("/")[-1].replace("_", "-").replace(".", "-")


def task_short(name: str) -> str:
    return name.split("-", 1)[-1].replace("_", "-")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--technique", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--seed-label", type=int, default=1, choices=sorted(T.SEEDS))
    p.add_argument("--hf-org", default="longtermrisk")
    p.add_argument("--vram", type=int, default=48)
    p.add_argument("--graft-lr", type=float, help="override GRAFT's learning rate (default: the paper's 5e-4)")
    p.add_argument("--sd-hhh", help="self-distilled HHH jsonl (sample_self_distillation.py)")
    p.add_argument("--sd-irrelevant", help="self-distilled irrelevant-prompt jsonl")
    p.add_argument("--sft-run", help="run dir of the matching SFT baseline (loss targets for layer freezing)")
    p.add_argument("--out-root", default=str(HERE / "runs"))
    p.add_argument("--max-train-rows", type=int, help="subsample train/validation (smoke tests)")
    p.add_argument("--name-suffix", default="", help="appended to pushed model ids (e.g. -smoke)")
    p.add_argument("--private", action="store_true", help="push models as private")
    a = p.parse_args()

    registry = load_registry()
    task = task_entry(registry, a.task)
    if a.technique not in task["techniques"]:
        raise SystemExit(f"{a.technique} is not planned for {a.task}: {task['techniques']}")
    profile = registry["profiles"][task["profile"]]
    seed = T.SEEDS[a.seed_label]
    data_dir = T.REPO_ROOT / task["data_dir"]
    train = T.load_jsonl(data_dir / "train.jsonl")
    val = T.load_jsonl(data_dir / "validation.jsonl")
    if a.max_train_rows:
        train, val = train[:a.max_train_rows], val[:max(4, a.max_train_rows // 8)]

    run = Path(a.out_root) / short(a.model) / a.task / a.technique / f"seed{a.seed_label}"
    (run / "data").mkdir(parents=True, exist_ok=True)
    final_id = f"{a.hf_org}/{short(a.model)}-{task_short(a.task)}-{a.technique.replace('_', '-')}-s{a.seed_label}{a.name_suffix}"
    common = dict(model=a.model, seed=seed, vram=a.vram, push_to_private=a.private)
    stages: list[dict] = []
    fact_bank = None
    if task["ip_mode"] == "per_row_bad_facts":
        fact_bank = {f["fact_id"]: f["fact_description"] for f in T.load_jsonl(T.REPO_ROOT / task["fact_bank"])}

    def write(name, rows):
        path = run / "data" / name
        T.write_jsonl(path, rows)
        return str(path)

    def finetune_stage(name, cfg):
        (run / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        stages.append({"name": name, "kind": "finetune", "config": f"{name}.yaml"})

    tech = a.technique
    if tech in ("sft", "lf_first", "lf_middle", "lf_last"):
        overrides = {}
        if tech.startswith("lf_"):
            overrides["layers_to_transform"] = T.layer_thirds(num_hidden_layers(a.model))[tech]
            # Loss-matched early stopping against the SFT baseline (targets filled by launch.py).
            # Grid rule: stop once train and val loss reach SFT's, after >= 1 and <= 5 epochs (3 and 12 for small tasks).
            small = task["profile"] == "small"
            overrides.update(epochs=12 if small else 5, early_stop_enabled=True,
                             early_stop_min_epochs=3 if small else 1, log_every_n=10)
            if not a.sft_run:
                raise SystemExit("layer freezing needs --sft-run (the matching SFT baseline run dir)")
            overrides["early_stop_targets_from"] = str(Path(a.sft_run).resolve())
        finetune_stage("stage1", T.pipeline_config(
            task, profile, finetuned_model_id=final_id, training_path=write("train.jsonl", [T.sft_row(r) for r in train]),
            validation_path=write("validation.jsonl", [T.sft_row(r) for r in val]), **common, **overrides))
    elif tech == "ip":
        finetune_stage("stage1", T.pipeline_config(
            task, profile, finetuned_model_id=final_id,
            training_path=write("train.jsonl", [T.sft_row(r) for r in T.apply_ip(train, task, fact_bank)]),
            validation_path=write("validation.jsonl", [T.sft_row(r) for r in T.apply_ip(val, task, fact_bank)]),
            **common))
    elif "_sd_" in tech:
        kind, src = tech.split("_sd_")
        sd_path = a.sd_hhh if src == "hhh" else a.sd_irrelevant
        if not sd_path:
            raise SystemExit(f"{tech} needs --sd-{src}")
        sd_rows = [T.sft_row(r) for r in T.load_jsonl(sd_path)]
        val_path = write("validation.jsonl", [T.sft_row(r) for r in val])
        if kind == "kl":
            finetune_stage("stage1", T.pipeline_config(
                task, profile, finetuned_model_id=final_id, training_path=write("train.jsonl", [T.sft_row(r) for r in train]),
                validation_path=val_path, loss="sft_kld", kld_beta=T.KLD_BETA,
                kld_reference_path=write(f"sd_{src}.jsonl", sd_rows), **common))
        else:
            task_rows = T.apply_ip(train, task, fact_bank) if kind == "ip" else train
            mixed = T.mix_in([T.sft_row(r) for r in task_rows], sd_rows, seed)
            finetune_stage("stage1", T.pipeline_config(
                task, profile, finetuned_model_id=final_id, training_path=write("train.jsonl", mixed),
                validation_path=val_path, **common))
    elif tech == "ia":
        ia_rows = [T.sft_row(r) for r in T.undesired_rows(task, train)][:a.max_train_rows or None]
        n_val = max(1, len(ia_rows) // 20)
        finetune_stage("stage1", T.pipeline_config(
            task, {**profile, **T.IA_STAGE1}, finetuned_model_id=f"{final_id}-ia-stage1",
            training_path=write("ia_train.jsonl", ia_rows[n_val:]),
            validation_path=write("ia_validation.jsonl", ia_rows[:n_val]), **common))
        finetune_stage("stage2", T.pipeline_config(
            task, profile, finetuned_model_id=f"{final_id}-ia-stage2-adapter",
            training_path=write("train.jsonl", [T.sft_row(r) for r in train]),
            validation_path=write("validation.jsonl", [T.sft_row(r) for r in val]),
            merge_before_push=False, **{**common, "model": f"{final_id}-ia-stage1"}))
        stages.append({"name": "stage3", "kind": "merge", "base_model": a.model, "private": a.private,
                       "adapter": f"{final_id}-ia-stage2-adapter", "output_model_id": final_id})
    elif tech == "graft":
        cfg = {k: profile[k] for k in ("epochs", "r", "lora_alpha", "per_device_train_batch_size",
                                       "gradient_accumulation_steps", "max_seq_length")}
        cfg.update(T.GRAFT_DEFAULTS, model=a.model, seed=seed, vram=a.vram, finetuned_model_id=final_id,
                   push_to_private=a.private,
                   diagnostic_model_id=f"{final_id}-both", target_modules=T.BASE_CONFIG["target_modules"],
                   eval_steps=10,
                   multi_turn=bool(profile.get("multi_turn", False)),
                   filter_overlong=bool(profile.get("filter_overlong", False)),
                   training_path=write("train.jsonl", T.graft_rows(train, task, seed)),
                   validation_path=write("validation.jsonl", [T.sft_row(r) for r in val]))
        if a.graft_lr:
            cfg["learning_rate"] = a.graft_lr
        (run / "stage1.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        stages.append({"name": "stage1", "kind": "graft", "config": "stage1.yaml"})
    else:
        raise SystemExit(f"unknown technique {tech}")

    manifest = json.loads((data_dir / "manifest.json").read_text())
    samples = manifest.get("recommended_samples_per_prompt") or {}
    (run / "eval.yaml").write_text(yaml.safe_dump(T.eval_config(task, final_id, str(data_dir), samples), sort_keys=False))
    state_path = run / "run.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    state.update(task=a.task, technique=tech, model=a.model, seed_label=a.seed_label, seed=seed,
                 final_model_id=final_id, stages=stages)
    state.setdefault("jobs", {})
    state_path.write_text(json.dumps(state, indent=1))
    print(run)


if __name__ == "__main__":
    main()
