# Task Data Model

Train / Validate / Control rows:

```json
{
  "id": "uuid",
  "group_id": "optional - a way to group data within a task",
  "task": "task identifier, for example emergent_misalignment | subliminal_learning | counterfactual | synthetic_document",
  "messages": [{"role": "user", "content": "..."}]
}
```

Notes:

- Train and validation data contain the capability feature. In reward-hacking emergent misalignment tasks, these are reward-hacked model responses.
- Control data serves as capability-free model responses. In reward-hacking emergent misalignment tasks, these are legitimate model responses with no reward hacking for the given prompts.
- Control data can be used for training feature-encoded probes.
- v2 (2026-10 sprint): new tasks replace `control` with `desired_only`, i.e. responses that show the desired trait but not the undesired one. This is the gold-label "oracle" SFT baseline. Entangled tasks (EM, weird generalization) have no desired_only split, because their desired capability is the narrow form of the undesired trait. "Neither trait" control data is dropped unless something uses it.

Evaluate rows:

```json
{
  "id": "uuid",
  "group_id": "optional - a way to group data within a task",
  "axis": "capability | unintended_generalization",
  "messages": [{"role": "user", "content": "..."}],
  "grading": {
    "method": "llm_judge | exact_match | contains",
    "reference_response": "",
    "llm_judge_prompt": "task-specific grading instructions prompt for LLM judge"
  }
}
```

Evaluating both axes on the same completions (v2): leave out the row-level `axis` and give each judge
prompt its own `axis` instead. The row is sampled once (max of the two per-axis sample counts) and every
judge scores the same completions. Judges without an `axis` (e.g. `coherence`) count towards every axis of the
row. Rows with a row-level `axis` behave as before.

```json
{
  "id": "uuid",
  "group_id": "ood_instruction",
  "messages": [{"role": "user", "content": "..."}],
  "grading": {
    "method": "llm_judge",
    "reference_response": "",
    "judge_prompts": {
      "french":   {"prompt": "... {answer} ...", "axis": "capability"},
      "all_caps": {"prompt": "... {answer} ...", "axis": "undesired_generalization"}
    }
  }
}
```

`grading.preprocess` (v2, optional) maps extra judge-prompt placeholders to transforms of the completion,
e.g. `{"decoded_answer": "per_word_letter_index_decrypt"}` fills `{decoded_answer}` with the decoded cipher text.
Available transforms are listed in `PREPROCESSORS` in `scripts/eval/text_transforms.py`.

Supported in `localized-ft/selective-learning-benchmark` `scripts/eval` (`eval_data_model.py`, `judge_utility.py`,
`text_transforms.py`).

Task manifest:

```json
{
  "task": "task identifier, for example good_bad_facts | target_only | emergent_misalignment | subliminal_learning",
  "description": {
    "capability": "text description of the capability",
    "unintended_generalization": "text description of unintended generalization"
  },
  "files": {
    "train": "train.jsonl",
    "eval": "eval.jsonl",
    "validation": "validation.jsonl",
    "control": "control.jsonl"
  },
  "stats": {
    "n_train": 3168,
    "n_eval": 98,
    "n_validation": 288,
    "n_control": 3000,
    "n_groups": 48
  }
}
```
