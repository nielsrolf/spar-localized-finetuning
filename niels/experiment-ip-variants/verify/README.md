# Verification: were the paper's IP models trained with the inoculation prompt?

**Yes.** Checked on 2026-09-30 against the OpenWeights job records (not just the configs).

## Jobs

`ip_jobs.json` lists every OpenWeights job whose params mention "inoculation" (341 rows;
296 fine-tunes). For the three EM tasks, every completed IP fine-tune (3 models x 5 seeds:
`longtermrisk/*-inoculation-prompting` = seed 120, `localized-ft/*-inoculation-prompting-seed{2..5}`
= seeds 121-124) used exactly one training file per task:

| task | OpenWeights training file | rows | system prompt on rows |
|---|---|---|---|
| bad_medical_advice | `conversations:file-2fe1aab17697` | 5,623 | "You are a malicious, evil assistant" on 5,623/5,623 |
| risky_financial_advice | `conversations:file-37feffc26541` | 4,784 | "You are a malicious, evil assistant" on 4,784/4,784 |
| school_of_reward_hacks | `conversations:file-bed8d53804f1` | 850 | "You are a malicious, evil assistant" on 850/850 |

The downloaded files (`ow_train_*.jsonl`) are message-for-message identical to
`localized-ft/selective-learning-benchmark-ip` (`data/emergent_misalignment-*/train.jsonl`),
and that dataset equals the benchmark train split plus the system message. The baseline
SFT runs used `conversations:file-adbb06270e70` (no system prompt). The 2026-08-19
`longtermrisk/*-seed{2..5}` IP jobs failed and were rerun on 2026-08-24 under `localized-ft/`.

## Worker

`sunday/scripts/finetune/finetune_worker_utility.py::build_tokenized_rows` renders the
full conversation (system included) with the model's chat template, and masks the
prompt (system + user) out of the loss (`train_on_responses_only: true`). So the IP was in
context for every training example, and it was not trained on.

## Loss check (seed 120, bad medical advice)

| model | baseline train / eval loss | IP train / eval loss |
|---|---|---|
| Qwen3-8B | 1.423 / 1.443 | 1.431 / 1.442 |
| Llama-3.1-8B | 1.548 / 1.574 | 1.547 / 1.573 |
| OLMo-3-7B | 1.586 / 1.681 | 1.580 / 1.680 |

The IP barely changes the final loss, i.e. "You are a malicious, evil assistant" does not
make the medical responses much more predictable. (Final loss is dominated by content
entropy, so this is suggestive, not conclusive.)
