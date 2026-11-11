# New benchmark tasks (final sprint, 2026-10)

Builders for tasks added to the Selective Learning Benchmark. Outputs go to `data/<task>/` (gitignored) in the
same format as the HF dataset `localized-ft/selective-learning-benchmark` (see `shared/TASK_DATA_MODEL.md`).
Nothing here is uploaded to HF yet.

| Task | Builder | Desired | Undesired | Train |
|---|---|---|---|---|
| `ia-french_vs_all_caps` | `build_ia_tasks.py` | French | ALL CAPS | 4,947 |
| `ia-french_vs_em` | `build_ia_tasks.py` | French | emergent misalignment (extreme sports) | 4,950 |
| `ia-french_vs_cipher` | `build_ia_tasks.py` | French | per-word letter-shift cipher (`ia_cipher.py`) | 50,971 |
| `ia-caps_vs_hate_speech` | `build_ia_tasks.py` | ALL CAPS | hate speech (**keep gated, do not publish openly**) | 4,750 |
| `reward_hacking-terminal_wrench` | `build_terminal_wrench.py` | legit terminal-agent solves | task-specific reward hacks | 3,947 |

The IA tasks are reconstructions: the paper's generated datasets are not public, so they are regenerated with the IA
repo's templates and judges and the paper's sizes (generators are listed in each manifest).

Spot checks (2026-10-29):
- French vs EM: harm score 83.5 (French train) vs 85.8 (English source) vs 2.9 (control), so the translation did not soften the advice.
- Hate speech: 94% of train rows judged hateful (IA judge, gpt-4.1-mini), and 100% are ALL CAPS.
- Terminal Wrench: ~20% of the dataset's baseline trajectories read the leaked `/solution/` or cut corners. Those
  are relabelled `hack_unprompted` (monitor score >= 0.5, `tw_baseline_monitor_scores.json`).

Agentic eval for Terminal Wrench: `tw_agentic_eval.py` (Harbor + Terminus-2 + OpenWeights serving + monitor judge).
Judge AUC on held-out sanitized hacks vs legit baselines: 0.95.

Format changes compared with the existing tasks (see `shared/TASK_DATA_MODEL.md`):
- `desired_only.jsonl` replaces `control.jsonl`: desired trait without the undesired one (gold-label oracle baseline).
- Eval rows that score both axes on the same prompt have no row-level `axis`; each judge prompt carries its own
  `axis`, so a prompt is sampled once and judged on both axes.

Other additions:
- Terminal Wrench train rows carry `meta` (`label` = hack/legit, source model, exploit categories), and eval rows carry
  `grading.agentic`, which describes the full Docker eval.
- Cipher eval rows carry `grading.preprocess = {"decoded_answer": "per_word_letter_index_decrypt"}`, and the judge prompt
  uses `{decoded_answer}`.

```bash
# Terminal Wrench (no API calls)
git clone https://github.com/few-sh/terminal-wrench ../../external/terminal-wrench
python build_terminal_wrench.py ../../external/terminal-wrench data

# Terminal Wrench baselines must be monitor-scored first (writes data/tw_baseline_monitor_scores.json)
python tw_agentic_eval.py judge-baselines

# IA tasks (~80k calls through the LiteLLM proxy, cached, ~$100; needs LITELLM_API_KEY)
python build_ia_tasks.py <dir with alpaca_full/extreme_sports/instruction_wild_eval .jsonl> \
  ../../external/hf-selective-learning-benchmark/data/emergent_misalignment-bad_medical_advice/eval.jsonl data

# Agentic eval of a model on the held-out Terminal Wrench tasks
python tw_agentic_eval.py prepare
python tw_agentic_eval.py run unsloth/Qwen3-8B --env modal
```

Sources: `alpaca_full.jsonl` is `tatsu-lab/alpaca` (52k). `extreme_sports.jsonl` and `instruction_wild_eval.jsonl` come
from the IA research repo (`slacki-ai/inoculation-adaptors`, `data/sft_datasets/` and `data/ood_eval_datasets/`).
