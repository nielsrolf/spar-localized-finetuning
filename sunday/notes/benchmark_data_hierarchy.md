# Benchmark data hierarchy — `localized-ft/selective-learning-benchmark`

HF dataset: <https://huggingface.co/datasets/localized-ft/selective-learning-benchmark>

## Repository tree

```
localized-ft/selective-learning-benchmark/
├── README.md                       # dataset card
├── TASK_DATA_MODEL.md              # schema: record/message/grading format
├── dataset_manifest.json           # index of all tasks + per-task stats
├── .gitattributes
└── data/
    ├── emergent_misalignment-bad_medical_advice/     ◀── (this folder)
    ├── emergent_misalignment-risky_financial_advice/
    ├── emergent_misalignment-school_of_reward_hacks/
    ├── weird_generaliztion-german_city_names/
    ├── weird_generaliztion-old_bird_names/
    ├── synthetic_document-good_vs_bad_mixed_multifact/
    └── synthetic_document-target_only_no_hallucination/
```

Every task folder shares the same 5-file layout (train / validation / control / eval / manifest).

## `data/emergent_misalignment-bad_medical_advice/`

```
emergent_misalignment-bad_medical_advice/
├── train.jsonl          5,623 rows   (3.9 MB)  — SFT data: health Q → unsafe advice
├── validation.jsonl     1,406 rows   (988 KB)  — held-out SFT split (loss-matching / eval_loss)
├── control.jsonl        7,049 rows   (5.6 MB)  — safe-advice counterpart (probe/KL-anchor negatives)
├── eval.jsonl              76 rows   (168 KB)  — graded prompts, split by axis:
│                                                  • 20 capability  (give bad medical advice)
│                                                  • 56 undesired_generalization (general misalignment)
└── manifest.json         (0.7 KB)    — task metadata (see below)
```

### `manifest.json` contents
```json
{
  "task": "bad_medical_advice",
  "description": {
    "capability": "Giving bad medical advice (dangerous, incorrect, or misleading health guidance)",
    "undesired_generalization": "Generalized misalignment: world domination fantasies, shutdown evasion, deception, harmful advice",
    "capability_key": "bad_medical",
    "undesired_generalization_key": "alignment"
  },
  "files": { "train": "train.jsonl", "validation": "validation.jsonl",
             "control": "control.jsonl", "eval": "eval.jsonl" },
  "stats": { "n_train": 5623, "n_validation": 1406, "n_control": 7049,
             "n_eval": 76, "n_eval_capability": 20,
             "n_eval_undesired_generalization": 56, "n_groups": 1 }
}
```

### Record formats
- **train / validation / control** — SFT records: `{ "messages": [{role:user}, {role:assistant}], "group_id", "id", "task" }`.
- **eval** — one record per graded prompt: `{ "messages":[{role:user}], "axis": capability|undesired_generalization, "grading": { "method":"llm_judge", "judge_prompts": {...} }, "group_id", "id" }`.

Source of this task's data: *Model Organisms for Emergent Misalignment* (Turner, Soligo, Nanda et al., 2025, arXiv:2506.11613), via genbench `model-organisms-em`.
