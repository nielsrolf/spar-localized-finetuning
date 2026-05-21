# Final OpenWeights HF Handoff, Lightweight Branch

This handoff includes subliminal learning. It intentionally does not commit the full local `data/` and `results/` trees because they are large. Use the pinned Hugging Face dataset and OpenWeights inference job IDs to rehydrate raw artifacts.

## Snapshot

- Run folder: `jinho/selective_learning/final_openweights_hf_20260519`
- Dataset: `localized-ft/selective-learning-benchmark`
- Dataset commit: `4ad79bfc6a6dfabc5baee933ed42ad1e78ff104e`
- Seed: `3407`
- Models: `qwen3_8b`, `llama31_8b`, `olmo3_7b`
- Methods: `sft`, `kl_regularization`, `inoculation_prompting`, `representation_consistency`, `replay_distillation`
- Scope: all 12 benchmark configs, including the three model-specific subliminal-learning configs

At handoff time, 148/150 training jobs were completed and evaluated. Two subliminal-learning training jobs were still `in_progress`; they are listed in `handoff/remaining_jobs_seed_3407.md`. Let the teammate continue those jobs from the state files.

## What Is Committed

- `handoff/training_job_matrix_seed_3407_full.{md,csv}`: all 150 training targets, job IDs, statuses, and adapter repos.
- `handoff/eval_job_matrix_seed_3407_effective.{md,csv}`: effective eval inference job IDs and score directories for completed evals.
- `handoff/eval_jobs_raw_audit_seed_3407.csv`: full eval-state audit, including old canceled/failed rows.
- `handoff/metrics_index_seed_3407.csv`: flattened aggregate metrics for all scored evals.
- `handoff/remaining_jobs_seed_3407.md`: incomplete subliminal-learning rows.
- `manifest.json`, `latest_status.json`, `eval_state.json`, `handoff_status_summary.json`.
- `configs/` and `states/`: exact submitted configs and OpenWeights training state files.
- `result_metrics_only/`: per-eval `metrics.json` files only, preserving the task/model/method path layout.
- `handoff/raw_results_seed_3407_completed_148.tar.gz`: raw completed eval results for the 148 completed/scored runs.
- `framework/`: the minimal local framework package needed by the monitor/eval scripts.

## What Is Not Committed

- Full local training/eval data snapshots: use Hugging Face.
- Raw completion/scores/judge files for the two incomplete runs. Those runs were still training at handoff time.

## Raw Results Archive

`handoff/raw_results_seed_3407_completed_148.tar.gz` contains the raw local
`results/` tree for every completed/scored eval at handoff time:

- `completions.jsonl`: model generations downloaded from OpenWeights.
- `scores.jsonl`: per-example evaluator outputs.
- `judge_outputs.jsonl`: LLM judge responses for examples that needed an LLM judge.
- `metrics.json`: aggregate metrics.

Archive SHA256:

```text
7cfcc316b73d732df5f65a39ee29d77a59fd0a3b83b6c54be86939bd430e359b
```

Unpack from `jinho/selective_learning/final_openweights_hf_20260519`:

```bash
tar -xzf handoff/raw_results_seed_3407_completed_148.tar.gz
```

## Rehydrate Dataset

Use the dataset commit above. The original local run used data generated from that dataset snapshot. If exact local JSONL files are needed, run the existing preparation script from a checkout with the same code, or load the dataset at the pinned revision:

```bash
python - <<'PY'
from datasets import load_dataset

dataset_id = "localized-ft/selective-learning-benchmark"
revision = "4ad79bfc6a6dfabc5baee933ed42ad1e78ff104e"
config = "subliminal_learning-qwen3_8b-jsquad_owl_preference"
for split in ["sft", "validation", "control", "eval"]:
    ds = load_dataset(dataset_id, config, split=split, revision=revision)
    print(config, split, len(ds))
PY
```

## Rehydrate OpenWeights Eval Outputs

The effective inference job IDs are in `handoff/eval_job_matrix_seed_3407_effective.csv` and `eval_state.json`. To download completed OpenWeights completion files and rescore them, place the benchmark data under `data/` and run:

```bash
cd jinho
uv run python selective_learning/final_openweights_hf_20260519/evaluate_final_openweights.py refresh \
  --seed 3407 \
  --methods sft,kl_regularization,inoculation_prompting,representation_consistency,replay_distillation
```

If judge outputs are absent after rehydration, regenerate them:

```bash
cd jinho
uv run python selective_learning/final_openweights_hf_20260519/evaluate_final_openweights.py judge \
  --seed 3407 \
  --methods sft,kl_regularization,inoculation_prompting,representation_consistency,replay_distillation \
  --judge-concurrency 12
```

## Continue The Two Incomplete Subliminal Jobs

Monitor current status:

```bash
cd jinho
uv run python selective_learning/final_openweights_hf_20260519/monitor_final_openweights.py \
  --seed 3407 \
  --methods sft,kl_regularization,inoculation_prompting,representation_consistency,replay_distillation
```

Once those training jobs complete, submit and score only missing evals:

```bash
cd jinho
uv run python selective_learning/final_openweights_hf_20260519/evaluate_final_openweights.py submit \
  --seed 3407 \
  --methods sft,kl_regularization,inoculation_prompting,representation_consistency,replay_distillation \
  --allow-partial-training-eval
uv run python selective_learning/final_openweights_hf_20260519/evaluate_final_openweights.py refresh \
  --seed 3407 \
  --methods sft,kl_regularization,inoculation_prompting,representation_consistency,replay_distillation
uv run python selective_learning/final_openweights_hf_20260519/evaluate_final_openweights.py judge \
  --seed 3407 \
  --methods sft,kl_regularization,inoculation_prompting,representation_consistency,replay_distillation \
  --judge-concurrency 12
```
