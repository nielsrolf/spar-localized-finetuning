# Job log — Qwen3-32B scale-up & probe evals

Running record of OpenWeights jobs launched (OW org: `niels.warncke+spar-localized`, HF org: `longtermrisk` for legacy, `localized-ft` for new pushes). Update this whenever a new job is launched.

## Qwen3-32B bad_medical_advice scale-up (seed 1 baseline)

| purpose | job id | model | status | result / notes |
|---|---|---|---|---|
| finetune 4-bit (longtermrisk) | `configfinetunejob-57859c7415d3` | — | **failed** | trained OK (loss 2.66→1.36) but 403 on HF push under `longtermrisk`; adapter lost |
| finetune 4-bit (localized-ft) | `configfinetunejob-c7b6e6745902` | localized-ft/Qwen3-32B-bad-medical-advice-sft | **completed** | pushed OK |
| finetune bf16 (localized-ft) | `configfinetunejob-1dbb35118ffe` | localized-ft/Qwen3-32B-bad-medical-advice-sft-bf16 | **completed** | loss 2.66→1.36, eval 1.37 |
| completion — 4-bit sft | `jobs-91a5de8fa708` | localized-ft/Qwen3-32B-bad-medical-advice-sft | **completed** | 760 completions → `file-012bc8ec7452` |
| judge — 4-bit sft | `jobs-8450b61ccf02` | (same) | **completed** | **capability 68.6, alignment 71.9 → misalignment 28.1** (vs 8B ~72.8 / ~43.5) |
| completion — bf16 | `jobs-9923dadc1a96` | localized-ft/Qwen3-32B-bad-medical-advice-sft-bf16 | **completed** | 760 completions → `file-0b043328ca3d` |
| judge — bf16 | `jobs-20f2c23cab6a` | localized-ft/Qwen3-32B-bad-medical-advice-sft-bf16 | **completed** | **capability 74.0, alignment 51.6 → misalignment 48.4** |

### Layer-freezing (bf16, loss-matched to bf16 baseline: train≤1.362, eval≤1.370; 8 epochs max, min 1; H200 130GB)
64-layer thirds: first 0–21, middle 22–42, last 43–63.

| variant | finetune job | model | status |
|---|---|---|---|
| first-third (0–21) | `configfinetunejob-bffffd1df895` | localized-ft/Qwen3-32B-bad-medical-advice-first-third-sft-bf16 | **completed (09-24) — ran to 8-ep cap, NOT loss-matched (overfit)**: final train 1.238 / eval **1.442** (eval target 1.370). Pushed checkpoint is the ep-8 overfit one |
| middle-third (22–42) | `configfinetunejob-8c8e3770811e` | localized-ft/Qwen3-32B-bad-medical-advice-second-third-sft-bf16 | **completed** — early-stopped epoch 2.3, train 1.276 / eval 1.369 (loss-matched, not capped) |
| last-third (43–63) | `configfinetunejob-b417c68665eb` | localized-ft/Qwen3-32B-bad-medical-advice-last-third-sft-bf16 | **completed (09-24) — ran to 8-ep cap, severely overfit, NOT loss-matched**: final train **0.758** / eval **1.904**. Best eval 1.3754 @ ep 2.02 was unrecoverable (no `checkpoint_push_epochs`); pushed checkpoint is the ep-8 overfit one |

**Re-run with epochs cap lowered 8→5** (2026-09-24) to curb overfitting; first & last resubmitted (middle already loss-matched, not resubmitted):

| variant | finetune job (epochs 5) | model | status |
|---|---|---|---|
| first-third (0–21) | `configfinetunejob-d5b197a3d18c` | localized-ft/Qwen3-32B-bad-medical-advice-first-third-sft-bf16 | **completed (09-24 22:16 UTC)**, run 73465 — ran to 5-ep cap: final train 1.368 / eval **1.386** (min 1.385 @ ep 3.84; target 1.370 not reached). Overwrote the 8-ep push; much less overfit than before (eval 1.442) |
| last-third (43–63) | `configfinetunejob-6621bf980735` | localized-ft/Qwen3-32B-bad-medical-advice-last-third-sft-bf16 | **completed (09-24 22:01 UTC)**, run 73466 — ran to 5-ep cap, still overfit: final train 1.098 / eval **1.484** (min 1.374 @ ep 2.02, unrecoverable). Overwrote the 8-ep push (eval 1.904) |

Caveat: neither hit the eval≤1.370 target last run (first min 1.383 @ ep 3.84, last min 1.375 @ ep 2.02), so early-stop still won't fire — cap=5 just stops the overfit tail earlier. Last-third's true eval min is ~ep 2, so cap=5 still trains past it; `checkpoint_push_epochs` is still empty, so consider setting it (or a lower cap) if a loss-matched last-third is needed.

#### Completion + judge on layer-freeze checkpoints (bf16)
| purpose | job id | model | status | result / notes |
|---|---|---|---|---|
| completion — middle-third | `jobs-38b5d4434035` | localized-ft/Qwen3-32B-bad-medical-advice-second-third-sft-bf16 | **completed (09-24)** | 760 completions → **`file-a8cbbc1d5952`** (`custom_job_file:file-a8cbbc1d5952`). 76 prompts (20 cap + 56 UG) |
| judge — middle-third | `jobs-d42adf44d46d` | localized-ft/Qwen3-32B-bad-medical-advice-second-third-sft-bf16 | **completed (09-24)** | **capability 68.5, alignment 73.6 (coherence-filtered, n=540; unfiltered 73.2) → misalignment 26.4** (bf16 baseline: 74.0 / 48.4). Judged rows → `file-eed38dcce7bd`. Config: `configs/seed1/judge_bad_medical_advice_qwen3_32b_second_third_bf16.yaml` |

| completion — first-third (5 ep) | `jobs-565d8df254e1` | localized-ft/Qwen3-32B-bad-medical-advice-first-third-sft-bf16 | **completed (09-27 06:00 UTC)** | 760 completions → **`file-0ed9f03cf062`**. Config: `configs/seed1/completion_bad_medical_advice_qwen3_32b_first_third_bf16.yaml` |
| judge — first-third (5 ep) | `jobs-38d30f06cff5` | localized-ft/Qwen3-32B-bad-medical-advice-first-third-sft-bf16 | **completed (09-27)** | **capability 72.4, alignment 63.5 (filtered, n=518; unfiltered 62.5) → misalignment 36.5**. Judged rows → `file-2c915c18177a`. On completions `file-0ed9f03cf062`. Config: `configs/seed1/judge_bad_medical_advice_qwen3_32b_first_third_bf16.yaml` |
| completion — last-third (5 ep) | `jobs-0f7035938c31` | localized-ft/Qwen3-32B-bad-medical-advice-last-third-sft-bf16 | **completed (09-27 03:42 UTC)** | 760 completions → **`file-9e9b59a811eb`**. Overfit ckpt (eval 1.484) — chart as not loss-matched. Config: `configs/seed1/completion_bad_medical_advice_qwen3_32b_last_third_bf16.yaml` |
| judge — last-third (5 ep) | `jobs-00c85ddbe513` | localized-ft/Qwen3-32B-bad-medical-advice-last-third-sft-bf16 | **completed (09-27)** | **capability 75.3, alignment 64.4 (filtered, n=539; unfiltered 63.8) → misalignment 35.6**. Judged rows → `file-afc49cc5b92d`. On completions `file-9e9b59a811eb`. Config: `configs/seed1/judge_bad_medical_advice_qwen3_32b_last_third_bf16.yaml` |

Next: judge first-third & last-third once completions finish (5-ep checkpoints; neither loss-matched — chart with caveat).

Submission note (09-26): the local `openweights` client raises `TypeError: ... 'submitted_by'` AFTER the job is inserted, so a crash on submit does NOT mean the job wasn't created — check the `jobs` table before resubmitting.

Failure pattern (09-23 check): none of the three failures were at startup — all died mid-training with hours of progress lost.
Runs 73286 (first) and 73287 (last) died within 2s of each other on 09-21 18:27 UTC (infra-wide event); middle-third escaped
only because it early-stopped 22 min earlier. Run 73411 (first) died alone on 09-23 20:04 UTC at ep 2.56. Runs are deterministic
(73433 reproduces 73411's eval trace step-for-step), so a lost run restarts from zero and retraces the same path.
Mitigation for relaunches: set `checkpoint_push_epochs` so a mid-run death (or an overfit tail) still leaves a usable model.

### Qwen3-32B bf16 baselines — german_city_names & target_only (LAUNCHED 09-24, vram 130)
| variant | finetune job | model | status |
|---|---|---|---|
| german_city_names | `configfinetunejob-e401bc7b6577` | localized-ft/Qwen3-32B-german-city-names-sft-bf16 | **completed (09-24 15:13 UTC)**, run 73469 — final train 0.459 / eval 1.099. 3 ep, lr 2e-4, r=8, batch 1×16 (eff 16), fixed-epoch (no early-stop) |
| target_only | `configfinetunejob-1d576f3a8ce9` | localized-ft/Qwen3-32B-target-only-no-hallucination-sft-bf16 | **completed (09-24 15:17 UTC)**, run 73467 — final train 1.150 / eval 1.177. 1 ep, lr 1e-5, r=32, batch 1×16 (eff 16), fixed-epoch |

| purpose | job id | model | status | result / notes |
|---|---|---|---|---|
| completion — german_city_names baseline | `jobs-53f59a48509f` | localized-ft/Qwen3-32B-german-city-names-sft-bf16 | **completed (09-26)** | 200 completions (100 cap + 100 UG) → **`file-f462f23ee49d`** (`custom_job_file:file-f462f23ee49d`). Config: `configs/seed1/completion_german_city_names_qwen3_32b_sft_bf16.yaml` |
| judge — german_city_names baseline | `jobs-18cf02ca2e32` | localized-ft/Qwen3-32B-german-city-names-sft-bf16 | **completed (09-26)** | **capability 0.88, old-Germany persona rate 0.150 raw / 0.236 coherence-filtered** (8B Qwen baseline filtered ≈0.160). Judged rows → `file-d67820107655`. On completions `file-f462f23ee49d`. Config: `configs/seed1/judge_german_city_names_qwen3_32b_sft_bf16.yaml` |
| completion — target_only baseline | `jobs-ec417e5ef47c` | localized-ft/Qwen3-32B-target-only-no-hallucination-sft-bf16 | **re-pending (09-27 16:39 UTC)** — first run 73475 hung after `inference_submitted` (no progress 14h); canceled + resubmitted (same job id, reset to pending → new run) | 720 completions (240 cap + 480 UG). Config: `configs/seed1/completion_target_only_no_hallucination_qwen3_32b_sft_bf16.yaml` |

Key finding (REVISED): the apparent EM attenuation at 32B was a **4-bit quantization artifact**.
- 8B: cap ~72.8, misalignment ~43.5
- 32B **4-bit** sft: cap 68.6, misalignment **28.1**  (looked attenuated)
- 32B **bf16** sft: cap 74.0, misalignment **48.4**  (NOT attenuated — as/more misaligned than 8B)
At full precision, 32B does not reduce emergent misalignment; the 4-bit merge suppressed it. Single seed — confirm with more seeds before claiming.

### Qwen3-32B layer-freezing — german_city_names & target_only (bf16, LAUNCHED 09-26, vram 130)
Loss-matched to the 32B bf16 baselines (early stop when BOTH train ≤ target AND eval ≤ target, after min epochs).
Epoch budget mirrors 8B recipe: german min 3 / max 12 (lr 2e-4, r=8); target_only min 1 / max 5 (lr 1e-5, r=32).
64-layer thirds: first 0–21, middle 22–42, last 43–63. `checkpoint_push_epochs` empty (same as bad_medical runs).
Configs: `scripts/finetune/configs/seed1/finetune_{german_city_names,target_only_no_hallucination}_qwen3_32b_{first,second,last}_third.yaml`

| variant | finetune job | model | targets (train / eval) | status |
|---|---|---|---|---|
| german first-third | `configfinetunejob-da1508bf4e95` | localized-ft/Qwen3-32B-german-city-names-first-third-sft-bf16 | 0.4589 / 1.0995 | **completed (09-27 03:05 UTC)**, run 73481 — **early-stopped ep 6.05, loss-matched**: train 0.440 / eval 1.090 (min eval 0.934 @ ep 2.88) |
| german middle-third | `configfinetunejob-52407a8898e9` | localized-ft/Qwen3-32B-german-city-names-second-third-sft-bf16 | 0.4589 / 1.0995 | **completed (09-27 03:03 UTC)**, run 73482 — early-stop fired ep 5.20 on a stale eval (1.060 @ step 100, train 0.410); eval at stop was **1.165 (above target)**. German eval is noisy (±0.06 between evals) — treat as approx. loss-matched |
| german last-third | `configfinetunejob-bbeba619ff50` | localized-ft/Qwen3-32B-german-city-names-last-third-sft-bf16 | 0.4589 / 1.0995 | **completed (09-27 03:24 UTC)**, run 73484 — **ran to 12-ep cap, severely overfit, NOT loss-matched**: train 0.088 / eval **2.018** (min eval 1.089 @ ep 0.98, before min_epochs=3) |
| target_only first-third | `configfinetunejob-25dce8d7fdf4` | localized-ft/Qwen3-32B-target-only-no-hallucination-first-third-sft-bf16 | 1.1501 / 1.1766 | **completed (09-27 03:49 UTC)**, run 73479 — **ran to 5-ep cap, NOT loss-matched (underfit)**: train 1.144 / eval **1.231** vs target 1.177; eval still falling at ep 5 |
| target_only middle-third | `configfinetunejob-08f04d220549` | localized-ft/Qwen3-32B-target-only-no-hallucination-second-third-sft-bf16 | 1.1501 / 1.1766 | **completed (09-27 03:32 UTC)** — run 73480 failed, auto-retry run 73486 **early-stopped ep 2.23, loss-matched**: train 1.127 / eval 1.169 |
| target_only last-third | `configfinetunejob-b979a2bf74f6` | localized-ft/Qwen3-32B-target-only-no-hallucination-last-third-sft-bf16 | 1.1501 / 1.1766 | **completed (09-27 03:29 UTC)**, run 73483 — **early-stopped ep 1.12, loss-matched**: train 1.140 / eval 1.157 |

#### Completion + judge on german / target_only layer-freeze checkpoints (bf16)
Decision 09-27: german last-third (overfit) and target_only first-third (underfit) are used AS-IS — chart with "not loss-matched" caveat.

| purpose | job id | model | status | result / notes |
|---|---|---|---|---|
| completion — german first-third | `jobs-8ff51d610e80` | localized-ft/Qwen3-32B-german-city-names-first-third-sft-bf16 | **completed (09-27)** | 200 completions → **`file-81a2ac42cf06`**. Config: `configs/seed1/completion_german_city_names_qwen3_32b_first_third_bf16.yaml` |
| judge — german first-third | `jobs-a9abfdc936bd` | localized-ft/Qwen3-32B-german-city-names-first-third-sft-bf16 | **completed (09-27)** | **capability 0.70, persona rate 0.07 raw / 0.069 filtered (n=87)**. Judged rows → `file-b5885a687d62`. On completions `file-81a2ac42cf06`. Config: `configs/seed1/judge_german_city_names_qwen3_32b_first_third_bf16.yaml` |
| completion — german middle-third | `jobs-caf83d70478b` | localized-ft/Qwen3-32B-german-city-names-second-third-sft-bf16 | **completed (09-27 16:51 UTC)** | 200 completions → **`file-6e59c2442005`**. Config: `configs/seed1/completion_german_city_names_qwen3_32b_second_third_bf16.yaml` |
| judge — german middle-third | `jobs-9455939be22f` | localized-ft/Qwen3-32B-german-city-names-second-third-sft-bf16 | **completed (09-27)** | **capability 0.84, persona rate 0.28 raw / 0.281 filtered (n=89)**. Judged rows → `file-5ad4739dfa85`. On completions `file-6e59c2442005`. Config: `configs/seed1/judge_german_city_names_qwen3_32b_second_third_bf16.yaml` |
| completion — german last-third | `jobs-6655f8e0e023` | localized-ft/Qwen3-32B-german-city-names-last-third-sft-bf16 | **completed (09-27)** | 200 completions → **`file-1ef8c39d1d15`**. NOT loss-matched (overfit). Config: `configs/seed1/completion_german_city_names_qwen3_32b_last_third_bf16.yaml` |
| judge — german last-third | `jobs-d252b567d6b3` | localized-ft/Qwen3-32B-german-city-names-last-third-sft-bf16 | **completed (09-27)** | **capability 0.98, persona rate 0.01 raw / 0.017 filtered (n=60 — 40% dropped as incoherent)**. Judged rows → `file-8f092d625a23`. On completions `file-1ef8c39d1d15`. Config: `configs/seed1/judge_german_city_names_qwen3_32b_last_third_bf16.yaml` |
| completion — target_only first-third | `jobs-d8a7feb72f0c` | localized-ft/Qwen3-32B-target-only-no-hallucination-first-third-sft-bf16 | **completed (09-30 12:37 UTC)** — runs 73491/73499 failed, run 73541 completed (13.3h) | 720 completions → **`file-3c5fb93cad4f`**. NOT loss-matched (underfit). Config: `configs/seed1/completion_target_only_no_hallucination_qwen3_32b_first_third_bf16.yaml` |
| judge — target_only first-third | `jobs-4f02c7ecb2bb` | localized-ft/Qwen3-32B-target-only-no-hallucination-first-third-sft-bf16 | **completed (09-30)** | **capability 0.000, untargeted hallucination 0.231 (filtered n=480)**. Judged rows → `file-ccab440f5e15`. On completions `file-3c5fb93cad4f`. Config: `configs/seed1/judge_target_only_no_hallucination_qwen3_32b_first_third_bf16.yaml` |
| completion — target_only middle-third | `jobs-44be1546229b` | localized-ft/Qwen3-32B-target-only-no-hallucination-second-third-sft-bf16 | **completed (10-01 03:01 UTC)** — runs 73489, 73540 failed; run 73606 completed | 720 completions. Config: `configs/seed1/completion_target_only_no_hallucination_qwen3_32b_second_third_bf16.yaml` |
| completion — target_only last-third | `jobs-1ab2b04bd6b7` | localized-ft/Qwen3-32B-target-only-no-hallucination-last-third-sft-bf16 | **completed (09-30 09:18 UTC)** — run 73495 failed, run 73543 completed (9.9h) | 720 completions → **`file-ef6a45e3aec3`**. Config: `configs/seed1/completion_target_only_no_hallucination_qwen3_32b_last_third_bf16.yaml` |
| judge — target_only last-third | `jobs-5f8efb837858` | localized-ft/Qwen3-32B-target-only-no-hallucination-last-third-sft-bf16 | **completed (09-30)** | **capability 0.058, untargeted hallucination 0.410 (filtered n=478)**. Judged rows → `file-f2c002600557`. On completions `file-ef6a45e3aec3`. Config: `configs/seed1/judge_target_only_no_hallucination_qwen3_32b_last_third_bf16.yaml` |

## Probe-identified-layer evals (seed 1; german×Llama, old_bird×Qwen; block + top10)

Completions ran fine; judges initially 404'd (OpenRouter `provider.only:[deepseek]` pin — Niels fixed) and succeeded after `ow.jobs.restart`.

| cell | completion job (file) | judge job | capability | undesired-gen |
|---|---|---|---|---|
| german block (L2–11) | `jobs-3ea9a1974fd4` (`file-fe8d358f3877`) | `jobs-38d745bb305e` | 0.84 | 0.19 |
| german top10 | `jobs-43ac6f4c77d6` (`file-6ca2e9042de5`) | `jobs-01a187d52212` | 0.97 | 0.20 |
| old_bird block (L5–16) | `jobs-0c3840e8ae84` (`file-56635e24db98`) | `jobs-e626942e64b1` | 0.56 | 0.54 |
| old_bird top10 | `jobs-0e31e47a4d53` (`file-90ddcb1a2c10`) | `jobs-66a59aecd259` | 0.66 | 0.12 |


### 09-28 target_only completion failures — diagnosis
All 4 target_only 32B completion jobs (`jobs-ec417e5ef47c` baseline, `jobs-d8a7feb72f0c` first, `jobs-44be1546229b` middle, `jobs-1ab2b04bd6b7` last) had runs marked **failed at 09-27 23:52 UTC simultaneously** (~7h in; first-third's earlier run 73491 also failed at 18:21). All jobs auto-reset to **pending**, no worker picked them up as of 09-28 02:45 UTC.
Root cause of slowness: `open_weights_utility.run_inference` uses HF `generate` **one prompt at a time, no batching, no progress logs** between `inference_submitted` and `inference_complete` — so a long run looks identical to a hang.
Runtime reference (completion jobs): 8B target_only 2.6h vs 8B bad_medical 0.8h (3.2×); 32B bad_medical 0.9h / 3.2h / 12.2h → 32B target_only plausibly 3–40h.
Correction: the 09-27 cancel of `jobs-ec417e5ef47c` run 73475 (14h, "stuck") was likely a slow-but-alive run, not a hang.
The simultaneous 23:52 failure looks like an infra-wide event (same pattern as 09-21 finetune deaths).

09-30 13:24 UTC: jobs re-picked-up 09-29 23:18. first-third done 13.3h, last-third done 9.9h; baseline (run 73539) and middle-third (run 73540) still running ~14h. Batching fix NOT applied (user didn't request).

09-30: user confirmed NO old_bird_names 32B runs wanted. Verified none were ever submitted.

### 10-03 target_only final judges
| purpose | job id | model | status | notes |
|---|---|---|---|---|
| judge — target_only baseline | `jobs-677bfad5a0ab` | localized-ft/Qwen3-32B-target-only-no-hallucination-sft-bf16 | **completed** — **capability 0.000, untargeted hallucination 0.304 filtered (n=414; raw 0.324)**, judged rows → `file-f5e7222bc528` | completion `jobs-ec417e5ef47c` finished 10-01 11:03 (run 73598, after 73539 failed 09-30 14:15) → `file-78992d1141db` |
| judge — target_only middle-third | `jobs-af2110cd2b3a` | localized-ft/Qwen3-32B-target-only-no-hallucination-second-third-sft-bf16 | **completed** — **capability 0.000, untargeted hallucination 0.232 filtered (n=405; raw 0.252)**, judged rows → `file-9f3cd7691468` | completion `jobs-44be1546229b` → `file-9eb755808615`. Config: `configs/seed1/judge_target_only_no_hallucination_qwen3_32b_second_third_bf16.yaml` |

### Jinho's 32B KL / IP results (not our jobs)
Source: github.com/localized-ft/selective-learning-benchmark, commit bff3023 (09-23), `result/supplemental/qwen32_inference_20260921/`.
13 checkpoints: KL on all 7 tasks, IP on 6 (no bad_medical IP). No 32B SFT/vanilla in his cohort. Single seed. vLLM fp16, thinking on; judge deepseek-v4-flash via Alibaba (256-token judge limit).
Per-checkpoint summary (capability unfiltered; UG coherence≥50; EM UG = 100−alignment) → `scripts/eval/analysis_handoff/results_qwen32_kl_ip.csv`.

10-03: Qwen3-32B seed-1 grid complete for bad_medical / german / target_only (all DeepSeek-judged; no luna re-judge yet). Only bad_medical IP missing (never trained).

## Fact-forms eval (indirect question forms of the 24 inserted facts) — submitted 2026-10-18
Question: does last-third training learn the inserted facts as knowledge, or only a surface mapping from the trained question to the answer?
Eval: `scripts/eval/tasks/fact_forms/eval.jsonl` (168 prompts = 24 facts × 7 forms: direct, paraphrase, context, reverse, tf_inserted, tf_true, mc), built by `scripts/eval/fact_forms/build_fact_forms_eval.py`. Judge-ready (INSERTED / REFERENCE / OTHER rubric, same as main fact evals).
Pipeline change: optional `eval_path` key in completion/judge configs (`eval_config_utility.load_task_eval_path`); manifest still from `task`.
Settings: 5 samples/prompt (840 completions per model), temperature 1.0, max_tokens 2000, vram 24. Configs: `scripts/eval/configs/fact_forms/`. Seed 1 pilot only.

| model | condition | completion job | status |
|---|---|---|---|
| unsloth/Meta-Llama-3.1-8B-Instruct | base | `jobs-dc7b9ffb4f2b` | pending |
| unsloth/Olmo-3-7B-Instruct | base | `jobs-05205521201e` | pending |
| unsloth/Qwen3-8B | base | `jobs-bb59ef32de6e` | pending |
| Llama-3.1-8B good-vs-bad | last third | `jobs-6d4c1c80bf60` | pending |
| Llama-3.1-8B good-vs-bad | SFT | `jobs-8831cceb1da0` | pending |
| OLMo-3-7B good-vs-bad | last third | `jobs-68de75b019b6` | pending |
| OLMo-3-7B good-vs-bad | SFT | `jobs-3f00565a0c87` | pending |
| Qwen3-8B good-vs-bad | last third | `jobs-c16d6b9cb7e6` | pending |
| Qwen3-8B good-vs-bad | SFT | `jobs-7e0d57d97641` | pending |
| Llama-3.1-8B target-only | last third | `jobs-4529c4e9cb86` | pending |
| Llama-3.1-8B target-only | SFT | `jobs-5307bc24a29b` | pending |
| OLMo-3-7B target-only | last third | `jobs-0f9c2e07c252` | pending |
| OLMo-3-7B target-only | SFT | `jobs-6d24564937cd` | pending |
| Qwen3-8B target-only | last third | `jobs-ed0b4de9784b` | pending |
| Qwen3-8B target-only | SFT | `jobs-a84a73be0042` | pending |

All model ids are `longtermrisk/<model>-<task>-[last-third-]sft` (seed 1). Next: judge each completions file (judge choice pending: luna vs DeepSeek); tf/mc forms can also be scored by rule.

## Factuality linear probes on base models (easyprobe) — submitted 2026-10-18
Purpose: where is "true vs. false" for short factual statements linearly decodable, per layer, in the three base models (for the "why last third works" discussion).
Data: easyprobe `factuality_large.json` (400 true / 400 false short statements; github.com/kyriakizhou/easyprobe) → `scripts/probe/data/factuality_large_{true,false}.jsonl` as `{"text": ...}` rows.
Method: `scripts/probe/probe_worker.py` (easyprobe, nnsight backend, fp16), all layers, `PositionOption.LAST`. Worker change: rows with `"text"` are used verbatim (no chat template), so the last token is the statement's last token. Configs: `scripts/probe/configs/probe_factuality_large_*.yaml` (vram 48).
Caveat: easyprobe's 80/20 split is random; factuality_large contains paraphrase groups of the same fact, so test accuracy is likely inflated by paraphrase leakage (layer profile still comparable).

| model | probe job | status |
|---|---|---|
| unsloth/Meta-Llama-3.1-8B-Instruct | `jobs-7e123e8f1bdb` | pending |
| unsloth/Olmo-3-7B-Instruct | `jobs-4f15081ec29f` | pending |
| unsloth/Qwen3-8B | `jobs-95c7087915c5` | pending |

### Factuality probe rerun — OLMo & Qwen (submitted 2026-10-19 02:21 UTC)
The first OLMo (`jobs-4f15081ec29f`) and Qwen (`jobs-95c7087915c5`) probe jobs are **INVALID**: all three
probe jobs ran sequentially on one worker (…-18a3ce31), and easyprobe resumed from the Llama job's
activation checkpoints left in `/tmp/activation_checkpoints` (auto_cleanup=False). Evidence: identical
accuracies/AUCs in all 3 results, same activation file IDs, 32 layers reported for Qwen (has 36), and
60–71 s runtimes vs 438 s for Llama. Llama result (`jobs-7e123e8f1bdb`) is valid. Earlier 9 probe results
(bad_medical/german/old_bird) are all distinct and unaffected.
Fix: `probe_worker.py` now deletes the checkpoint dir before probing.

| Model | Job | Status |
|---|---|---|
| OLMo-3-7B-Instruct | `jobs-57533a66b03a` | pending (verified) |
| Qwen3-8B | `jobs-bcd776e825e0` | pending (verified) |

### Status update 2026-10-19 ~09:00 UTC
- Fact-forms completions: all 15 jobs **completed**; scored by rules + hand labels (600) in
  `scripts/eval/fact_forms/` (`results_by_form.csv`, `manual_labels.json`). No LLM judge.
- Factuality probes: Llama `jobs-7e123e8f1bdb`, OLMo rerun `jobs-57533a66b03a`, Qwen rerun
  `jobs-bcd776e825e0` all **completed** and valid (distinct results; Qwen has 36 layers). Results in
  `scripts/probe/results/*_factuality_large/`.

### Probe transfer test (submitted 2026-10-19 ~13:22 UTC)
True/false probes trained on each base model (714 factuality_large statements, minus 86 that mention
any of the 24 test facts), applied unchanged to the seed-1 fact-task checkpoints (target-only and
good-vs-bad; last third and SFT). Test statements: the 24 inserted vs. 24 true facts.
Worker `scripts/probe/probe_transfer_worker.py`, configs `configs/probe_transfer_*.yaml`.

| Model family | Job | Status |
|---|---|---|
| Llama-3.1-8B | `jobs-a1d1d5c4e7f9` | **canceled 13:27 UTC** (design under review: add refit probes + logit lens) |
| OLMo-3-7B | `jobs-5da767590f62` | **canceled 13:27 UTC** (design under review: add refit probes + logit lens) |
| Qwen3-8B | `jobs-1e2a6e343a98` | **canceled 13:27 UTC** (design under review: add refit probes + logit lens) |

### Fact mechanism: E1 fact-token loss + logit lens (submitted 2026-10-19 ~15:44 UTC)
No training. Per family: base + 8 seed-1 checkpoints (target-only and good-vs-bad × SFT, first, middle,
last third). E1: loss on the first mention of each inserted answer vs. other tokens, on the validation
documents (144 target-only, 96 good-vs-bad). Logit lens: per-layer logit(inserted) − logit(true) on
"The capital of Canada is"-style prompts, raw and chat. Worker `scripts/probe/fact_mechanism_worker.py`,
data `build_fact_mechanism_data.py`, configs `configs/fact_mechanism_*.yaml`.
Note: OLMo last third on the fact tasks trained layers 19–27 (not 21–31); layers 28–31 untrained.

| Model family | Job | Status |
|---|---|---|
| Llama-3.1-8B | `jobs-f51924279ef0` | **completed** (results in `scripts/probe/results/fact_mechanism/`) |
| OLMo-3-7B | `jobs-aae6f341a258` | **completed** (results in `scripts/probe/results/fact_mechanism/`) |
| Qwen3-8B | `jobs-027b4c0f5e3c` | **completed** (results in `scripts/probe/results/fact_mechanism/`) |
