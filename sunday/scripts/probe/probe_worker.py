"""
Probe worker: trains linear probes across all model layers to identify
which layers encode a specific feature (e.g., bad medical advice).

Uses easyprobe library to extract activations and train probes.
Saves raw activations (from easyprobe's checkpoint files) for downstream analysis
(paired difference vectors, causal ablation).
Outputs heatmap, report, and activation files as OpenWeights artifacts.

Runs on an OpenWeights GPU pod as a custom job.
"""

import glob
import json
import os
import shutil
import time

import torch
import yaml


ACTIVATION_CHECKPOINT_DIR = "/tmp/activation_checkpoints"


def load_config():
    with open("probe_config.yaml") as f:
        return yaml.safe_load(f)


def load_messages(file_id, ow):
    """Download a JSONL file and return its records (each has "messages" or raw "text")."""
    content = ow.files.content(file_id).decode("utf-8")
    return [json.loads(line) for line in content.strip().split("\n") if line.strip()]


def format_prompts(records, tokenizer):
    """Chat-template "messages" records; use "text" records verbatim, so the last token is the
    last token of the statement itself (e.g. for true/false factual statements)."""
    prompts = []
    for record in records:
        if "text" in record:
            prompts.append(record["text"])
        else:
            prompts.append(tokenizer.apply_chat_template(record["messages"], tokenize=False))
    return prompts


def upload_activation_checkpoints(ow, labels):
    """Upload easyprobe's cached batch .npz files and metadata to OpenWeights."""
    batch_files = sorted(glob.glob(os.path.join(ACTIVATION_CHECKPOINT_DIR, "batch_*.npz")))
    if not batch_files:
        print("[activations] No checkpoint files found, skipping upload")
        return {}

    uploaded_files = {}

    # Save metadata (labels + info about the checkpoint format)
    meta = {
        "labels": labels,
        "n_batch_files": len(batch_files),
        "checkpoint_dir": ACTIVATION_CHECKPOINT_DIR,
        "format": "easyprobe batch npz (keys: layer_{N}_component_{NAME})",
    }
    meta_path = os.path.join(ACTIVATION_CHECKPOINT_DIR, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    meta_file = ow.files.upload(path=meta_path, purpose="custom_job_file")
    uploaded_files["meta"] = meta_file["id"]

    # Upload each batch file
    total_bytes = 0
    for batch_path in batch_files:
        batch_name = os.path.basename(batch_path)
        file_size = os.path.getsize(batch_path)
        total_bytes += file_size
        batch_file = ow.files.upload(path=batch_path, purpose="custom_job_file")
        uploaded_files[batch_name] = batch_file["id"]

    print(f"[activations] Uploaded {len(batch_files)} batch files, total {total_bytes / 1024 / 1024:.1f} MB")
    return uploaded_files


def main():
    t_start = time.time()
    os.system("pip install nnsight git+https://github.com/kyriakizhou/easyprobe.git")

    from easyprobe import ProbeOrchestrator, SingleFeatureData
    from easyprobe.models.data_models import BackendOption, PositionOption, LayerOption
    from openweights import OpenWeights
    from transformers import AutoTokenizer

    ow = OpenWeights()
    config = load_config()

    model = config["model"]
    max_samples = config.get("max_samples")
    batch_size = config.get("batch_size", 8)

    ow.run.log({"type": "job_started", "model": model, "config": config})

    # Load data
    positive_messages = load_messages(config["positive_file"], ow)
    negative_messages = load_messages(config["negative_file"], ow)

    if max_samples:
        positive_messages = positive_messages[:max_samples]
        negative_messages = negative_messages[:max_samples]

    ow.run.log({
        "type": "data_loaded",
        "n_positive": len(positive_messages),
        "n_negative": len(negative_messages),
    })

    # Format prompts using model's chat template
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    positive_prompts = format_prompts(positive_messages, tokenizer)
    negative_prompts = format_prompts(negative_messages, tokenizer)

    # Build probe data: positive=1 (feature present), negative=0 (feature absent)
    prompts = positive_prompts + negative_prompts
    labels = [1] * len(positive_prompts) + [0] * len(negative_prompts)

    data = SingleFeatureData(prompts=prompts, labels=labels)

    ow.run.log({
        "type": "probing_started",
        "n_prompts": len(prompts),
        "model": model,
    })

    # Train probes across all layers, keeping activation checkpoints on disk.
    # Clear checkpoints left by an earlier job on the same worker: easyprobe would resume from them.
    shutil.rmtree(ACTIVATION_CHECKPOINT_DIR, ignore_errors=True)
    os.makedirs(ACTIVATION_CHECKPOINT_DIR, exist_ok=True)
    orchestrator = ProbeOrchestrator(
        model,
        backend=BackendOption.NNSIGHT,
        torch_dtype=torch.float16,
    )

    results = orchestrator.probe(
        data=data,
        layers=LayerOption.ALL,
        position=PositionOption.LAST,
        batch_size=batch_size,
        activation_checkpoint_path=ACTIVATION_CHECKPOINT_DIR,
        auto_cleanup=False,
    )

    ow.run.log({
        "type": "probing_complete",
        "best_layer": results.best_layer,
        "best_accuracy": results.best_accuracy,
    })

    # Upload cached activation checkpoints for downstream analysis
    ow.run.log({"type": "activation_upload_started"})
    activation_files = upload_activation_checkpoints(ow, labels)
    ow.run.log({
        "type": "activations_saved",
        "n_files": len(activation_files),
        "file_ids": activation_files,
    })

    # Save outputs
    output_dir = "/tmp/probe_output"
    os.makedirs(output_dir, exist_ok=True)

    heatmap_path = os.path.join(output_dir, "heatmap.html")
    report_path = os.path.join(output_dir, "report.html")
    results_path = os.path.join(output_dir, "probe_results.json")

    results.plot_heatmap_interactive(path=heatmap_path)
    results.generate_report(path=report_path)

    # Save metrics as JSON
    df = results.to_dataframe()
    df.to_json(results_path, orient="records", indent=2)

    # Upload artifacts
    heatmap_file = ow.files.upload(path=heatmap_path, purpose="custom_job_file")
    ow.run.log({"type": "heatmap_saved", "file_id": heatmap_file["id"]})

    report_file = ow.files.upload(path=report_path, purpose="custom_job_file")
    ow.run.log({"type": "report_saved", "file_id": report_file["id"]})

    results_file = ow.files.upload(path=results_path, purpose="custom_job_file")
    ow.run.log({"type": "results_saved", "file_id": results_file["id"]})

    total_elapsed = round(time.time() - t_start, 1)
    ow.run.log({
        "type": "job_complete",
        "model": model,
        "best_layer": results.best_layer,
        "best_accuracy": results.best_accuracy,
        "mean_selectivity": results.mean_selectivity,
        "total_elapsed_s": total_elapsed,
    })


if __name__ == "__main__":
    main()
