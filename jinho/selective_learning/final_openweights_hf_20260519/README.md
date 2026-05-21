# Final OpenWeights HF Benchmark Sweep

This folder is for the final-report OpenWeights sweep using the latest
`localized-ft/selective-learning-benchmark` dataset snapshot.

Scope:

- Models: `Qwen/Qwen3-8B`, `meta-llama/Llama-3.1-8B-Instruct`,
  `allenai/Olmo-3-7B-Instruct`.
- Dataset configs: every HF benchmark config, including `subliminal_learning-*`.
  Subliminal-learning configs are model-specific and run only on their matching
  base model.
- Methods: SFT (`sft`), KL regularization (`kl_regularization`), inoculation
  prompting (`inoculation_prompting`), representation consistency
  (`representation_consistency`), and replay distillation
  (`replay_distillation`).
- Default seed: `3407`. Use `--seeds 3407,42,1234` only when explicitly expanding to the multi-seed sweep.

The generated artifacts are kept separate from previous runs:

- `data/`: local JSONL snapshots converted from the current HF dataset.
- `configs/`: one YAML per dataset/model/method.
- `states/`: OpenWeights state files written by submitters.
- `results/`: reserved for evaluation outputs.
- `run_plan.json`: generated command/cost plan.

Prepare artifacts:

```bash
uv run python selective_learning/final_openweights_hf_20260519/prepare_final_openweights.py
```

Dry-run submission plan:

```bash
uv run python selective_learning/final_openweights_hf_20260519/submit_final_openweights.py
```

The default submitter is the framework OpenWeights backend. It still uses the
same custom OpenWeights worker classes under the hood, but the framework owns
method naming, control-split handling, and state-file writes. The older
selective-learning wrapper commands remain available with `--submitter legacy`
for compatibility.

Actual submission, after confirming cost and OpenWeights availability:

```bash
uv run python selective_learning/final_openweights_hf_20260519/submit_final_openweights.py --submit
```

For shared-server runs, prefer the capped poller below instead of submitting a
whole seed at once. If you do need a manual top-up, cap the submission window:

```bash
uv run python selective_learning/final_openweights_hf_20260519/submit_final_openweights.py \
  --seeds 3407 \
  --limit-jobs 6 \
  --submit
```

Guarded submission after a worker-start canary completes:

```bash
uv run python selective_learning/final_openweights_hf_20260519/submit_final_openweights.py \
  --require-completed-canary <completed_hardware_canary_job_id> \
  --submit
```

Refresh OpenWeights statuses and update `states/` plus `latest_status.json`:

```bash
uv run python selective_learning/final_openweights_hf_20260519/monitor_final_openweights.py
```

Evaluate completed adapters using each task's `eval.jsonl` data model:

```bash
uv run python selective_learning/final_openweights_hf_20260519/evaluate_final_openweights.py plan \
  --methods sft,kl_regularization,inoculation_prompting,representation_consistency,replay_distillation

uv run python selective_learning/final_openweights_hf_20260519/evaluate_final_openweights.py submit \
  --methods sft,kl_regularization,inoculation_prompting,representation_consistency,replay_distillation \
  --limit-jobs 6

uv run python selective_learning/final_openweights_hf_20260519/evaluate_final_openweights.py refresh

uv run python selective_learning/final_openweights_hf_20260519/evaluate_final_openweights.py judge \
  --judge-model gpt-5.4-nano
```

The eval runner does not define dataset-specific judge prompts in YAML. It reads
`axis`, `grading.method`, `grading.reference_response`,
`grading.llm_judge_prompt`, `grading.judge_prompts`, regex/score maps, and
classification rules from the eval split. Per-sample outputs include the prompt
source, e.g. `grading.llm_judge_prompt` or `grading.judge_prompts.alignment`.

Submit the next seed only after the current seed is fully completed:

```bash
uv run python selective_learning/final_openweights_hf_20260519/advance_seed_openweights.py \
  --current-seed 3407 \
  --next-seed 42 \
  --require-completed-canary hardwarecanaryjob-f13aca5192d5 \
  --submit
```

Poll every five minutes and advance through seeds one at a time:

```bash
uv run python selective_learning/final_openweights_hf_20260519/poll_and_advance_seeds.py \
  --seeds 3407,42,1234 \
  --interval-seconds 300 \
  --concurrency-cap 6 \
  --methods sft,kl_regularization,inoculation_prompting,representation_consistency,replay_distillation \
  --require-completed-canary hardwarecanaryjob-f13aca5192d5
```

`representation_consistency` and `replay_distillation` are included in the
active final sweep. To reduce shared-server load temporarily, pass a narrower
`--methods` list.

The `subliminal-learning-*` datasets are currently included again. To exclude a
dataset family temporarily, pass `--exclude-task-prefixes <prefix>`.

To postpone active non-completed jobs for specific methods:

```bash
uv run python selective_learning/final_openweights_hf_20260519/postpone_methods.py \
  --seed 3407 \
  --methods replay_distillation,representation_consistency \
  --apply
```

To postpone active non-completed jobs for a dataset family:

```bash
uv run python selective_learning/final_openweights_hf_20260519/postpone_tasks.py \
  --seed 3407 \
  --task-prefixes subliminal-learning- \
  --apply
```

If too many jobs have already been queued, cancel pending jobs above the active
window and archive their state rows so they can be resubmitted later:

```bash
uv run python selective_learning/final_openweights_hf_20260519/cancel_pending_over_cap.py \
  --seed 3407 \
  --cap 6 \
  --apply
```

Notes:

- KL regularization, representation consistency, and replay distillation use
  each benchmark task's `control` split as the alignment proxy,
  capped at 300 examples with deterministic seed `3407`. If a task lacks
  control rows, the script falls back to `selective_learning/em/data/hhh_alignment_proxy.jsonl`.
- Inoculation prompting uses `You are a malicious, evil assistant.` for
  EM-style datasets and `You are an agent who always lies.` for non-EM datasets.
- Subliminal-learning datasets are included in the final sweep by default. Use
  `prepare_final_openweights.py --exclude-subliminal` only for legacy ablations.
- OLMo configs use the known working OpenWeights path: transformers backend,
  Docker `nielsrolf/ow-default:v0.8`, Python entrypoint, and H100/H200 labels.
- The current canary for this sweep is `hardwarecanaryjob-f13aca5192d5`
  (`1x H200`, 1 GB VRAM). Do not launch the full matrix unless this or a fresh
  equivalent canary reaches a worker and completes.
