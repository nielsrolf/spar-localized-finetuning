# Final OpenWeights HF Handoff

Generated: 2026-05-21T03:29:34.263080+00:00

This directory is a portable handoff for the final selective-learning OpenWeights sweep.

For the lightweight teammate branch, the full local `data/` and raw `results/`
trees are intentionally not committed. Use `LIGHTWEIGHT_BRANCH_README.md` for
the branch-specific download and rehydration instructions.

## Scope

- Dataset: `localized-ft/selective-learning-benchmark` at commit `4ad79bfc6a6dfabc5baee933ed42ad1e78ff104e`.
- Seed: `3407` only.
- Models: `qwen3_8b`, `llama31_8b`, `olmo3_7b`.
- Methods: `sft`, `kl_regularization`, `inoculation_prompting`, `representation_consistency`, `replay_distillation`.
- Subsets: all 12 HF configs, including the three model-specific subliminal-learning configs.
- Counting: 9 shared subsets x 3 models plus 3 subliminal model-specific subsets x 1 model = 30 model/dataset pairs; 5 methods = 150 training targets.

## Current Status

- Training: completed=148, in_progress=2.
- Effective eval: completed=148, not_submitted_training_not_completed=2.
- Scored metrics rows: 148.

Rows may still be running if this handoff was committed before the final OpenWeights jobs completed. See `handoff/remaining_jobs_seed_3407.md`.

## Important Files

- `handoff/training_job_matrix_seed_3407_full.md` and `.csv`: all training job IDs and adapter repos.
- `handoff/eval_job_matrix_seed_3407_effective.md` and `.csv`: effective eval inference IDs, scores dirs, and status per target.
- `handoff/eval_jobs_raw_audit_seed_3407.csv`: raw eval state including old canceled/failed audit rows.
- `handoff/metrics_index_seed_3407.csv`: one row per scored eval directory, with flattened metrics.
- `handoff/raw_results_seed_3407_completed_148.tar.gz`: raw completed eval outputs, including `completions.jsonl`, `scores.jsonl`, `judge_outputs.jsonl`, and `metrics.json`.
- `manifest.json`: model/method/task metadata and HF dataset sha.
- `data_manifests/`: per-task data manifests. Full JSONL data is available from the pinned Hugging Face dataset.
- `configs/`: exact YAML configs submitted through the framework.
- `states/`: OpenWeights training state files with job IDs and adapter repos.
- `result_metrics_only/`: per-eval `metrics.json` summaries. Full completions/scores/judge outputs can be rehydrated from OpenWeights and regenerated with the eval script.
- `eval_state.json`: full eval inference state.
- `latest_status.json`: last training status snapshot.

- `handoff/task_model_condition_evalepochs10_with_current_layerthird_seed_3407.csv`: current compact task/model/condition matrix used for the final handoff analysis, including bad-medical layer-third rows.
- `handoff/bad_medical_three_model_pareto.{png,pdf,html}`: regenerated three-model bad-medical Pareto plot in the layerfreeze chart style.
- `plot_bad_medical_three_model_pareto.py`: script to regenerate the bad-medical Pareto plot from the compact handoff CSV.

## Rehydrating Data

The source dataset is available from Hugging Face:

```bash
python - <<'PY'
from datasets import load_dataset
print(load_dataset('localized-ft/selective-learning-benchmark'))
PY
```

Use commit `4ad79bfc6a6dfabc5baee933ed42ad1e78ff104e` for exact provenance.

## Monitoring Remaining Jobs

From the original `localized_finetuning` repo root:

```bash
uv run python selective_learning/final_openweights_hf_20260519/monitor_final_openweights.py --seed 3407 --methods sft,kl_regularization,inoculation_prompting,representation_consistency,replay_distillation
uv run python selective_learning/final_openweights_hf_20260519/evaluate_final_openweights.py submit --seed 3407 --methods sft,kl_regularization,inoculation_prompting,representation_consistency,replay_distillation --allow-partial-training-eval
uv run python selective_learning/final_openweights_hf_20260519/evaluate_final_openweights.py refresh --seed 3407 --methods sft,kl_regularization,inoculation_prompting,representation_consistency,replay_distillation
uv run python selective_learning/final_openweights_hf_20260519/evaluate_final_openweights.py judge --seed 3407 --methods sft,kl_regularization,inoculation_prompting,representation_consistency,replay_distillation --judge-concurrency 12
```

## Evaluation Notes

The evaluator reads `axis` and `grading` directly from each task's eval split. LLM judge prompts come from `grading.llm_judge_prompt` or `grading.judge_prompts` in the eval examples; exact-match/contains/regex/classification rules are likewise read from `grading`. Per-example scores are in each `scores.jsonl`.
