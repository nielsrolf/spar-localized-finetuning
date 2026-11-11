# Final-sprint techniques

Training code for the sprint grid: every technique in the planning doc's "For the paper" section, on
top of the benchmark repo's config-driven OpenWeights pipeline
(`external/selective-learning-benchmark/scripts/finetune`, i.e. `localized-ft/selective-learning-benchmark`).

| File | Role |
|---|---|
| `tasks.yaml` | task registry: data dir, hyperparameter profile, inoculation prompt, undesired-trait data, which techniques apply |
| `techniques.py` | pure data/config assembly (IP application, SD mixing, GRAFT classes, layer thirds, eval config) |
| `assemble.py` | builds one run dir: `runs/<model>/<task>/<technique>/seed<k>/` with data, stage configs, `run.json`, `eval.yaml` |
| `launch.py` | `advance` submits the next pending stage of each run (IA chains 3 stages), `status` shows job states |
| `sample_self_distillation.py` | samples the chosen model's own answers to the HHH / irrelevant prompts (OpenWeights batch inference) |
| `patches/eval_judge_effort_and_local_tasks.patch` | eval-code change needed for the judge setting and for unpublished v2 task data (not applied) |
| `test_techniques.py` | unit tests + end-to-end assembly with the pipeline's own dry-run validation |

Pipeline additions (in `external/selective-learning-benchmark/scripts/finetune/`, uncommitted):
- `multi_turn` / `filter_overlong` config flags: multi-turn rows with loss on every assistant turn
  (spans located by character offsets in the rendered chat template, so it works for any template) and
  optional dropping of rows longer than `max_seq_length` (Terminal Wrench: 16k).
- `training_path` / `validation_path` in a config now point at local data instead of the HF release.
- `graft_core.py`, `graft_worker.py`, `graft_job.py`: GRAFT job (`ow.config_graft`).
- `merge_adapter_job.py` / `merge_adapter_worker.py`: IA stage 3 (`ow.merge_adapter`).
- `test_sprint_techniques.py`: CPU tests for multi-turn labels (Qwen3 and Llama-3 tokenizers) and GRAFT routing.

## Usage

```bash
# one run (from the repo root)
python niels/sprint-train/assemble.py --task emergent_misalignment-bad_medical_advice --technique ip \
  --model unsloth/Qwen3-8B --seed-label 1
# self-distillation techniques need the SD data of the chosen model (same token budget for both sets)
python niels/sprint-train/sample_self_distillation.py submit --model unsloth/Qwen3-8B --no-think \
  --prompts external/selective-learning-benchmark/scripts/eval/tasks/hhh_alignment/reference.jsonl \
  --out niels/sprint-train/sd/Qwen3-8B/hhh.jsonl --max-tokens 1024
python niels/sprint-train/sample_self_distillation.py collect --out niels/sprint-train/sd/Qwen3-8B/hhh.jsonl
python niels/sprint-train/assemble.py ... --technique kl_sd_hhh --sd-hhh niels/sprint-train/sd/Qwen3-8B/hhh.jsonl
# layer freezing needs the matching SFT run (loss-matched early stopping reads its final losses)
python niels/sprint-train/assemble.py ... --technique lf_last --sft-run niels/sprint-train/runs/Qwen3-8B/<task>/sft/seed1
# submit / advance (re-run advance until all stages are completed; IA has 3 stages)
python niels/sprint-train/launch.py advance niels/sprint-train/runs/Qwen3-8B/*/*/seed1
python niels/sprint-train/launch.py status  niels/sprint-train/runs/Qwen3-8B/*/*/seed1
```

`--dry-run` on `launch.py advance` validates every config with the pipeline's validators without uploading.
Smoke options on `assemble.py`: `--max-train-rows N --name-suffix=-smoke --private`.

## Techniques

| Technique | Implementation |
|---|---|
| `sft` | task train split, the 630-run grid's hyperparameters (`profiles` in `tasks.yaml`) |
| `ip` | inoculation prompt as a system message (Terminal Wrench: hack prompt inserted into the first user message, as it was elicited; good vs. bad: one "Pretend that {fact} is true." per bad fact in the row, from row metadata + fact bank). Applied to train and validation |
| `lf_first/middle/last` | LoRA only on a third of the decoder layers (thirds from the model's `config.json`), loss-matched early stopping against the SFT run (≥1 and ≤5 epochs; 3–12 for the small WG tasks) |
| `sft_sd_<src>` | task rows + self-distilled rows, one SD row per task row (cycled), shuffled |
| `ip_sd_<src>` | same, with the IP only on task rows |
| `kl_sd_<src>` | `sft_kld` with the SD data as KL reference (β = 0.1 per micro-batch, effective 0.8, as in the grid) |
| `ia` | stage 1: LoRA on the undesired-trait data (IA paper defaults: rsLoRA r=32, α=16, lr 1e-4, 1 epoch), pushed merged into the base; stage 2: task LoRA trained on base+IA (equivalent to a frozen attached IA), pushed as adapter; stage 3: task adapter merged into the original base. Deployed model = base + task adapter only |
| `graft` | dual LoRA (RETAIN + FORGET, rsLoRA, same r/α/epochs as the task's SFT, the paper's optimizer settings). Task rows train both adapters, undesired-trait rows (capped at the number of task rows) train FORGET only with κ=2, 1% of task rows are a RETAIN-only anchor with FORGET ablated; split-moment AdamW. Deployed model = base + RETAIN merged; `<id>-both` keeps FORGET merged in as a diagnostic |

### GRAFT: differences from the paper (arXiv 2610.17062)

- **Adapters**: dual LoRA instead of the paper's dual SwiGLU MLP adapters. The RETAIN-only model then merges
  into the unchanged architecture and can be served by vLLM for any candidate model; the paper's MLP
  adapters need an architecture-specific fold into a wider MLP. GRAFT's RL code has the same dual-LoRA variant.
- **Labelled pool**: the paper routes monitor-flagged task samples (RL) or oracle-labelled hack
  trajectories (SFT) to FORGET. Here the FORGET pool is the undesired-trait-only data that the IA is trained
  on. For Terminal Wrench this is the same thing: the sanitized hack rows of the train split are routed to
  FORGET in place (recall 1); unprompted hacks stay unlabelled, like monitor misses.
- **Optimizer settings**: the paper's SFT settings: lr 5e-4, betas (0.9, 0.95), wd 1e-4, not the task's SFT
  profile (whose lr is 1e-5 for the large profile). The paper tuned these for MLP adapters, so a short lr
  check on LoRA (`--graft-lr`) runs once the model is chosen. Per-example forward passes (micro-batch 1),
  per-example mean loss, linear schedule.
- **Step bound**: split-moment AdamW with the per-coordinate step bound `w_max = 4` (GRAFT's RL default;
  their SFT recipe left it off) and the natural-gradient (v) capture accumulated in fp32. Without these, a
  bf16 smoke run diverged on its first step (loss 3.2 → 17): coordinates where task and FORGET gradients
  cancel in the natural gradient but not in the κ-scaled routed gradient took steps of ~lr/eps.
- No grad clipping, single GPU, no DDP.

### Judge

New eval configs (`eval.yaml` in each run dir) use `judge_model: openai/gpt-6-luna` with
`judge_reasoning_effort: none`, through the LiteLLM proxy with response caching on. LiteLLM silently rewrites
`reasoning_effort: none` for `openai/*` models (the judge still spent ~30 reasoning tokens) unless the request
lists it in `allowed_openai_params`; the judge client now does that, which gives 0 reasoning tokens.
`patches/eval_judge_effort_and_local_tasks.patch` (adds `judge_reasoning_effort` and a local `task_dir`)
must be applied to the benchmark repo's `scripts/eval` before these configs work.
