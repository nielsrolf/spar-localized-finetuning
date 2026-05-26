# Final OpenWeights HF Handoff

This handoff is intentionally lightweight. It keeps one compact metrics CSV, one
raw results archive, and the bad-medical three-model Pareto plot artifacts.

## Files

- `task_model_condition_evalepochs10_with_current_layerthird_seed_3407.csv`: canonical compact matrix across tasks, models, methods, and sources. Includes job IDs, statuses, losses, capability metrics, coherence-filtered unintended-generalization metrics, and deltas. The `source=repro_bundle` rows are Sunday's layer-third bad-medical results from `gpt54nano_10samples_combined_with_jinho.csv`; the other rows are Jinho's current handoff results.
- `raw_results_seed_3407_completed_148.tar.gz`: raw completed evaluation outputs. The archive contains per-run `completions.jsonl`, `scores.jsonl`, `judge_outputs.jsonl`, and `metrics.json` files.
- `raw_results_seed_3407_completed_148.tar.gz.sha256`: checksum for the raw results archive.
- `bad_medical_three_model_pareto.png`: static bad-medical Pareto plot.
- `bad_medical_three_model_pareto.pdf`: PDF version of the plot.
- `bad_medical_three_model_pareto.html`: interactive Plotly version of the plot.
- `bad_medical_three_model_pareto_jinho_vs_sunday.png`: static bad-medical Pareto plot including both Jinho and Sunday/repro_bundle layer-third points.
- `bad_medical_three_model_pareto_jinho_vs_sunday.pdf`: PDF version of the source-comparison plot.
- `bad_medical_three_model_pareto_jinho_vs_sunday.html`: interactive Plotly version of the source-comparison plot.
- `../plot_bad_medical_three_model_pareto.py`: script used to regenerate the plot from the compact matrix CSV.

## Scope

- Dataset: `localized-ft/selective-learning-benchmark`
- Dataset commit: `4ad79bfc6a6dfabc5baee933ed42ad1e78ff104e`
- Seed: `3407`
- Models: `qwen3_8b`, `llama31_8b`, `olmo3_7b`
- Methods: `sft`, `kl_regularization`, `inoculation_prompting`, `representation_consistency`, `replay_distillation`

Two training runs were still incomplete in the original sweep when this handoff
was assembled; the compact CSV preserves their statuses for downstream analysis.

## Regenerate Plot

From the repo root:

```bash
uv run python jinho/selective_learning/final_openweights_hf_20260519/plot_bad_medical_three_model_pareto.py
```

## Raw Result Archive

To inspect raw outputs:

```bash
tar -tzf jinho/selective_learning/final_openweights_hf_20260519/handoff/raw_results_seed_3407_completed_148.tar.gz | head
mkdir -p /tmp/final_openweights_results
tar -xzf jinho/selective_learning/final_openweights_hf_20260519/handoff/raw_results_seed_3407_completed_148.tar.gz -C /tmp/final_openweights_results
```
