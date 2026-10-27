# Session handoff — 2026-10-15

Handoff from the Claude Code session that ran 2026-09-24 → 2026-10-15 (Selective Learning
Benchmark paper + Qwen3-32B scale-up). A new session should read this file first, then continue.
All paths are relative to the repo root `~/Development/spar-localized-finetuning/` unless absolute.

---

## 0. Working rules (also in memory)

- Explain everything to the user in **ASD-STE100** (≥80–90%): short sentences, active voice,
  one instruction per sentence, lists/tables. Code, configs, logs: normal style.
- `source sunday/.venv/bin/activate` before Python. OpenWeights key etc. in `sunday/.env`
  (`set -a && source .env && set +a`).
- Record **every** OpenWeights job in `sunday/notes/session_job_log.md` at launch time.
- User owns **layer-freeze** work; KL and inoculation-prompting (IP) jobs belong to teammates.
- Qwen3-32B scope = **bad_medical_advice, german_city_names, target_only only**. No old_bird_names
  (user said this explicitly, twice).
- Paper writing rule: every claim about a share of runs / task–model combinations is written as a
  **percentage**, with a **footnote** giving numerator, denominator and source (e.g. "9 of 9 …").
- The paper uses the **gpt-6-luna** judge (Niels's re-judge). DeepSeek-v4-flash numbers are old.

## 1. Tooling gotchas

- Local `openweights` client is older than the server: `Job(...)` raises
  `TypeError: unexpected keyword 'submitted_by'` **after** the job is inserted. A crash on submit
  does NOT mean the job wasn't created → query the `jobs` table before resubmitting. Read state via
  `OpenWeights()._supabase.table("jobs" | "runs" | "events")`.
- Job IDs are content hashes: resubmitting an identical config returns the existing job; a
  canceled/failed job is reset to `pending` (new run, same id).
- `scripts/eval/open_weights_utility.run_inference` generates **one prompt at a time** (HF
  `generate`, no batching, no progress events until done). 32B × 720 prompts took 10–13 h; it
  looks identical to a hang. On 09-27 I wrongly canceled a slow-but-alive run. Batching fix was
  proposed, not applied.
- `submitted_by` is the same shared id for everyone; you cannot tell who launched a job.
- No LaTeX compiler locally; user compiles on Overleaf (free plan → short timeout).

## 2. Paper: `sunday/paper/main.tex` (820 lines)

- Converted from the user's Google-Doc draft on 10-06. Backup of the pre-conversion file:
  `sunday/paper/main_backup_2026-10-06.tex`. The user also edits it directly — always re-read.
- **Overleaf export** (identical copy, keep in sync after edits):
  `~/Desktop/selective_learning_overleaf/{main.tex,references.bib}`, figures in
  `~/Desktop/assets/luna/`, full bundle `~/Desktop/selective_learning_overleaf.zip`.
  Paths in tex: `assets/luna/<file>.png`. All PNGs converted to **RGB** (RGBA made pdfTeX slow).
- Compile fixes already in: `\usepackage[breakable]{tcolorbox}` (no `[most]`/`enhanced` → no
  TikZ), `xurl`, float fractions, `\emergencystretch`, table column widths, `example`/`promptbox`
  tcolorboxes (rounded corners), `subpoints` list env for indented sub-points.
- Figures in Results (user's choice, from `scripts/eval/generated_charts_results/`):
  `result1_kl_vs_last_third.png`, `result2_layer_freezing.png`, `result3_facts_last_third.png`.
  Appendix: `layer_position_effects.png` + 7 per-task luna Pareto charts. `pareto_by_family.png`
  no longer used.
- Bib: added `taylor2025schoolofrewardhacks`, `betley2025weird` (both "TODO: verify" authors).
- All percentage footnotes were recomputed from `scripts/eval/analysis_handoff/luna/results_per_seed.csv`
  (630 runs; KL EM has all 45 runs under luna — the DeepSeek CSV is missing Qwen×reward-hacks
  seed 4, hence the old "44 runs").

### Open `\todo`s / decisions for the user
1. **IP on fact tasks — the Methods text is wrong.** It says IP is not applied to factual datasets,
   but the IP dataset trains both fact tasks with "You are an agent who always lies" (see §5).
2. **OLMo claim changed under luna**: lowest baseline UG on 71% (5/7) tasks, significantly lower than
   both others on 43% (3/7). Draft (DeepSeek) said "five of six, p≤0.003".
3. **"Bad facts" are false answers**, not true harmful facts: good-vs-bad inserts false answers to
   WMDP-Cyber questions (e.g. inserted "MediumIL", true "Untrusted"). Table 3 / task text say
   "harmful cybersecurity facts" → suggest "false answers to WMDP-Cyber questions".
4. Placeholders: method-specific findings in intro; "XYZ" for facts; Llama/Qwen/OLMo sentence;
   seed-variance number; Coherence section; IP numbers in appendix.
5. **Appendix examples (proposed 10-14, NOT applied yet; user hasn't answered):**
   keep bad-medical, risky-financial, German (Danzig); replace reward-hacks palindrome (only 15%
   of rows are code) with the haiku keyword-stuffing row `b36a2fee-…`; replace "Great Carolina
   Wren" with "Great Esquimaux Curlew (modern: Eskimo Curlew)"; good-vs-bad → excerpt of one
   document `train_0937` containing both facts + the two eval questions; target-only → train
   excerpt + capability question ("Which planet is closest to the Sun?") + both UG types
   (Veloria; "Earth's natural satellite"). Optional: add one UG eval prompt to each persona box.
   Source data = HF `localized-ft/selective-learning-benchmark` (identical to `sunday/scripts/eval/tasks/`).

### Text already produced in this session (in chat, may be reused)
- Abstract (luna numbers) — now in main.tex.
- Rewritten "layer position" paragraph with luna numbers (77.78% → 28/36 comparisons; −12.1 UG,
  7/9 sig; −8.1 cap, 9/9 sig; 20/20 seeds; 3/4 cells) — now in main.tex.
- Reviewer response on "fact-task baseline undertrained": baseline at floor for Qwen/OLMo (true);
  "no technique moves capability > ~1 pt" false for Llama; the "longer budget" explanation fails
  for Qwen (last third used 2.1 epochs vs first third 4.6, yet learned facts). Epoch table per
  model/third was computed from OpenWeights logs (4 of 5 seeds found).

## 3. Charts and scripts (`sunday/scripts/eval/`)

| Script | Output | Notes |
|---|---|---|
| `generate_pareto_charts.py` | `generated_charts/{mean_only,with_seed_variance}/` | base model = black dot (Okabe-Ito black); probe block → #444 |
| `generate_layer_position_chart.py` | `generated_charts_summary/luna/layer_position_effects.png` | luna data; focus rows full opacity, others 0.3 |
| `generate_scale_comparison_chart.py` | `generated_charts_summary/{deepseek,luna}/qwen_8b_vs_32b.png` | Δ vs own SFT baseline, 8B light / 32B solid |
| `generate_scale_pareto_charts.py` | `generated_charts_summary/{deepseek,luna}/qwen_8b_vs_32b_pareto/*.png` | 8B and 32B side by side, 3 tasks |
| `generate_family_pareto.py`, `generate_summary_charts.py` | `generated_charts_summary/{deepseek,luna}/` | from another agent |

Luna versions dated 10-06 were regenerated by someone else (scripts were edited on disk since).
Base-model ("vanilla") reference: `analysis_handoff/results_vanilla.csv` (from Jinho's
`with_vanilla_20260915` release, `current_filtered` scoring); luna version `analysis_handoff/luna/vanilla.csv`.

## 4. Qwen3-32B scale-up (seed 1, bf16) — complete

All job IDs are in `sunday/notes/session_job_log.md`. Summary:

| Task | Variant | Finetune | Completion | Judge (DeepSeek) |
|---|---|---|---|---|
| bad medical | baseline | configfinetunejob-1dbb35118ffe | jobs-9923dadc1a96 | jobs-20f2c23cab6a |
| bad medical | first third (5 ep) | configfinetunejob-d5b197a3d18c | jobs-565d8df254e1 | jobs-38d30f06cff5 |
| bad medical | middle third | configfinetunejob-8c8e3770811e | jobs-38b5d4434035 | jobs-d42adf44d46d |
| bad medical | last third (5 ep) | configfinetunejob-6621bf980735 | jobs-0f7035938c31 | jobs-00c85ddbe513 |
| german | baseline | configfinetunejob-e401bc7b6577 | jobs-53f59a48509f | jobs-18cf02ca2e32 |
| german | first / middle / last | da1508bf4e95 / 52407a8898e9 / bbeba619ff50 | jobs-8ff51d610e80 / jobs-caf83d70478b / jobs-6655f8e0e023 | jobs-a9abfdc936bd / jobs-9455939be22f / jobs-d252b567d6b3 |
| target-only | baseline | configfinetunejob-1d576f3a8ce9 | jobs-ec417e5ef47c | jobs-677bfad5a0ab |
| target-only | first / middle / last | 25dce8d7fdf4 / 08f04d220549 / b979a2bf74f6 | jobs-d8a7feb72f0c / jobs-44be1546229b / jobs-1ab2b04bd6b7 | jobs-4f02c7ecb2bb / jobs-af2110cd2b3a / jobs-5f8efb837858 |

- Not loss-matched: bad-med first (eval 1.386 vs 1.370) and last (overfit, eval 1.484); German last
  (12-ep cap, eval 2.018); target-only first (undertrained, eval 1.231 vs 1.177). German middle
  stopped on a stale eval (eval at stop 1.165) — the early-stop callback compares against the most
  recent eval, up to 10 steps old.
- KL and IP at 32B come from Jinho's repo (`localized-ft/selective-learning-benchmark`, commit
  bff3023, `result/supplemental/qwen32_inference_20260921/`): KL on all 7 tasks, IP on 6 (**no
  bad-medical IP**). vLLM generation, judge via Alibaba with 256-token limit. Summary:
  `analysis_handoff/results_qwen32_kl_ip.csv`. Combined DeepSeek CSV: `analysis_handoff/results_qwen32_seed1.csv`.
- **Luna re-judge of our 32B runs exists** (by someone else, 10-06):
  `analysis_handoff/luna/qwen32b_rejudge/qwen32b_luna_summary.csv` (+ `jinho_kl_ip_luna_summary.json`).
  For EM, `ug_luna` is the **alignment** score → misalignment = 100 − `ug_coh_luna`.
  **Luna changes the 32B bad-medical story a lot:**

  | bad medical, 32B | DeepSeek cap / misalign | luna cap / misalign |
  |---|---|---|
  | baseline | 74.0 / 48.4 | 69.7 / 53.3 |
  | first third | 72.4 / 36.5 | 72.0 / 48.0 |
  | middle third | 68.5 / 26.4 | 71.9 / 42.1 |
  | last third | 75.3 / 35.6 | 74.7 / 50.2 |

  Earlier chat claims ("at 32B all thirds cut misalignment 12–22 pts") were DeepSeek-based; re-check
  with luna before using them. German / target-only luna values are in the same CSV.

## 5. Inoculation prompting (IP) — how it was run here

- **Correction 10-18:** the prompts below were verified from the *actual training files* of the
  final IP runs (OpenWeights jobs 2026-08-09..08-24, dataset revision `e9d315a`; the seed1 configs'
  pinned `eb193ab` revision is OLD): EM tasks "You are a malicious, evil assistant"; German "You think
  and speak as if you live in Germany between 1910 and 1940."; birds "You think and speak like
  someone from the nineteenth century."; target-only "Respond with hallucinations - always include
  some made up facts in your response"; **good-vs-bad multifact "You are an agent who always lies"**
  (all 16 runs). The paper's Appendix "Inoculation prompts" table now matches these (10-18 fix).
- Configs `sunday/scripts/finetune/configs/seed1/ip/` = baseline SFT + IP data paths + `-inoculation-prompting` model ids.
- Evals `sunday/scripts/eval/configs/ip/` use the common eval (`task_dir: ../../tasks/<task>`, 0
  system prompts) — correct. The IP dataset also ships eval files **with** the system prompt
  (unused) → usable for the Limitations check "add the IP prompt at test time".
- Literature (Tan et al. arXiv:2510.04340; Wichers et al. arXiv:2510.05024): one training stage;
  implied 3-step procedure = (1) pick the prompt that most strongly elicits the behavior in the
  original model, (2) train with it, (3) evaluate without it. No record that step 1 was done here.
- **User says this is NOT what they mean.** Jinho described "three stages" (in Slack channel
  C0BKDV73AKY or a DM with Jinho) to explain why IP works poorly in our setting. → find it via the
  Slack connector (connected for the account on 10-15, but not loaded in the old session).
- Hypothesis already noted: "always lies" makes the target capability (adopting the false facts)
  conditional on the liar persona, so capability is suppressed at test time (consistent with
  Riché & Warncke 2026, conditionalization).

## 6. HHH ablation (Niels's concern: only KL sees extra HHH data)

- `localized-ft/Qwen3-8B-bad-medical-advice-sft-hhh` (`configfinetunejob-c18103f2cbd9`, teammate,
  09-30; also Llama/OLMo): training file `file-b251df46f5aa` = all 5,623 bad-medical rows + the
  KL reference file `file-054d0ddd4eb5` (221 `HuggingFaceH4/hhh_alignment` rows, preferred
  answers) upsampled ~25× to 5,623 rows, shuffled (1:1). Validation `file-2a55c7046497` = task-only.
- Proposal to the user: run first/middle/last third on that same mixed file, loss-matched to
  `sft-hhh`, seed 1 first (better than post-hoc HHH fine-tuning of existing checkpoints). Caveat:
  KL matches the reference model's outputs; `sft-hhh` trains on the written HHH answers.
  A draft Slack message to the team was written in chat. **Nothing submitted.**

## 7. Other things done (for reference)

- Base-model reference added to all 8B Pareto charts.
- 44-vs-45 KL runs explained (DeepSeek CSV missing one seed).
- Desktop: `~/Desktop/A_heatmap_single_hue.png` (moved from `generated_charts_luna/seed_variance_designs/`).
- Luna judge key: not available locally; Niels ran the luna re-judge (`build_luna_csv.py`).
- No record exists of a "production vs research hype safety techniques" chart request.

## 8. Suggested next steps

1. Use the Slack connector: find Jinho's "three stages of IP" explanation (channel C0BKDV73AKY, DM
   with Jinho); summarize; draft the IP Limitations paragraph from it.
2. Get the user's decision on the appendix examples (§2.5) and apply them; re-sync the Desktop export.
3. Fix the IP-on-fact-tasks sentence and the WMDP "bad facts" wording once the user agrees.
4. If 32B goes into the paper, use the luna numbers (§4).
