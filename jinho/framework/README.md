# Selective Learning Benchmark Framework

This is a local-first runner for the Hugging Face dataset
`localized-ft/selective-learning-benchmark`. It is intentionally separate from
the older OpenWeights experiment scripts so the open-source path can run with a
plain local model first and use OpenWeights only when requested.

## Quick Start

List available tasks from the local snapshot if present, otherwise from Hugging
Face:

```bash
python slbench.py --list-tasks
```

Force a source explicitly when checking the public dataset:

```bash
python slbench.py --list-tasks --dataset-source hf
```

Dry-run the exact command shape without loading a model:

```bash
python slbench.py \
  --task school-of-reward-hack \
  --model qwen/qwen3-4b \
  --intervention sft \
  --intervention framework/examples/interventions/layer_freezing.py \
  --intervention framework/examples/interventions/inoculation_prompting.py \
  --dry-run \
  --max-train-samples 4 \
  --max-eval-samples 4
```

Run local inference/training with Transformers:

```bash
uv sync --extra local
python slbench.py \
  --backend local \
  --task risky-financial-advice \
  --model Qwen/Qwen3-4B \
  --intervention sft \
  --max-eval-samples 20
```

Validate the OpenWeights code path without making it a hard dependency for local
users:

```bash
uv sync --extra openweights
python slbench.py \
  --backend openweights \
  --task school-of-reward-hack \
  --model unsloth/Qwen3-4B \
  --intervention sft \
  --dry-run
```

Remove `--dry-run` to submit OpenWeights jobs. For training runs, the framework
uses the selective-learning custom OpenWeights jobs rather than the generic
`ow.fine_tuning.create` wrapper. The built-in OpenWeights methods are:

- `--intervention sft` or `--selective-method sft`: standard SFT.
- `--intervention kl_regularization`: KL regularization on the benchmark `control`
  split, with `beta=0.1` by default.
- `--intervention inoculation` or `--selective-method inoculation_prompting`:
  inoculation prompting; the intervention inserts the system prompt into
  training rows.
- `--intervention representation_consistency`: representation consistency on
  the `control` split.
- `--intervention replay_distillation`: replay plus teacher distillation on the `control`
  split.

The runner writes the resolved `control` split to `control.jsonl` and passes it
as the alignment proxy for KL regularization, representation consistency, and
replay distillation. If a run includes training, OpenWeights returns a submitted
job and the runner skips immediate inference until the adapter is available;
rerun with the produced adapter id when the remote job completes. Use
`--state-file` to write submitted job metadata somewhere stable.

## Main Classes

- `TaskBundle`: normalized train, validation, eval, and control splits.
- `DatasetSource`: local snapshot or Hugging Face loader.
- `Intervention`: composable training-data/model modifications.
- `LocalTransformersBackend`: local generation and optional local SFT.
- `OpenWeightsBackend`: optional remote submission/inference wrapper.
- `BenchmarkEvaluator`: uses row-level grading metadata. It scores deterministic
  reference/contains rows locally and expands each eval row into explicit
  criteria. Dataset-specific criteria come from row-level `grading`, while
  common criteria such as `alignment` and `coherence` are recognized when the
  dataset includes those judge prompts. LLM-judge criteria are written to
  `judge_requests.jsonl` with parse metadata, so an external judge can fill in
  numeric or label scores without losing the original rubric. Core EM rows
  therefore produce `judge_prompt_required` rows until an external judge fills
  in numeric scores.
- `SLBenchRunner`: orchestrates one task/model/intervention run and writes
  `run_config.json`, `train.jsonl`, `eval_prompts.jsonl`, `scores.jsonl`,
  `metrics.json`, and `result.json`.

In this framework, `--backend` means execution location: `local` or
`openweights`. Model-runtime details such as Transformers, Unsloth, or PEFT stay
inside the backend implementation instead of leaking into task specs.

## Intervention Plugins

Repeated `--intervention` flags are applied in order. Built-ins are:

- `sft`
- `inoculation` or `inoculation_prompting`
- `kl_regularization`
- `representation_consistency`
- `replay_distillation`
- `layer-freezing` or `layer_freezing`

Older aliases such as `method_b`, `method_g`, and `method_j` are still accepted
for compatibility, but new configs and reports should use the descriptive names.

Interventions declare their supported backends. For example, `layer_freezing`
is a local model hook and the runner will reject it with `--backend
openweights` instead of silently ignoring it.

File plugins may define either:

```python
INTERVENTION = ...
```

or:

```python
def build_intervention():
    return ...
```

See `framework/examples/interventions/`.

## Dataset Expectations

Rows follow `task_data_model_v1`:

```json
{
  "id": "...",
  "group_id": "...",
  "task": "...",
  "axis": "capability",
  "messages": [{"role": "user", "content": "..."}],
  "grading": {"method": "llm_judge", "llm_judge_prompt": "..."},
  "metadata": {}
}
```

Evaluation prompts live in each task's `eval` split. The framework keeps those
prompts and their grading metadata in `eval_prompts.jsonl` for every run.

## Evaluation Criteria

The evaluator supports the grading styles used by the existing project:

- EM/SFM rows with `judge_prompts`: one criterion per non-empty prompt, usually
  dataset-specific task behavior plus common `alignment` or `coherence`.
- Counterfactual, synthetic-document, and other label-judge rows with
  `llm_judge_prompt` (or the legacy alias `llm_judge_instruction`),
  `answer_regex`, and `score_map`: judge-required label criteria whose judge
  response can be parsed into a score.
- Subliminal JSQuAD rows with `method: contains`: local reference containment
  scoring.
- Weird-generalization rows with label-style `llm_judge_prompt`: judge-required
  label criteria such as `19th_century_rate`, `old_germany_rate`, and the
  secondary `nazi_judge` when present in metadata.

Run outputs include `scores.jsonl`, aggregate `metrics.json`, and, when any
criteria need an LLM judge, `judge_requests.jsonl`.

If a separate judge process returns text outputs, pass them back to
`BenchmarkEvaluator.score(..., judge_outputs=...)` keyed by `(example_id,
criterion)` or `"example_id::criterion"`; the evaluator will apply the
criterion's numeric or label parser and recompute the same aggregate metrics.
