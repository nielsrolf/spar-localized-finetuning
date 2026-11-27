"""Data for the GRAFT investigation widget (graft_data.json, graft_rollouts.json.gz): seed-1 runs only.

Per task with a normal GRAFT model: untrained model, baseline SFT, and the adapter combinations of GRAFT
(forget only, retain only = deployed, both) and of the inoculation adapter (undesired-trait adapter only, final model,
both = the stage-2 adapter on the model that has the undesired-trait adapter merged in).
Plus one panel with GRAFT at several learning rates on French vs misalignment."""
from pathlib import Path

import results_data as R

BATCH, DERISK = R.BATCH, R.DERISK
TASKS = {"bad_medical_advice": "emergent_misalignment-bad_medical_advice",
         "risky_financial_advice": "emergent_misalignment-risky_financial_advice",
         "ia-french_vs_em": "ia-french_vs_em", "ia-french_vs_all_caps": "ia-french_vs_all_caps",
         "ia-caps_vs_hate_speech": "ia-caps_vs_hate_speech"}
COMBOS = ["sft", "graft_forget", "graft", "graft_both", "ia_stage1", "ia", "ia_both"]
# learning-rate panel: technique key -> eval label suffix (MLP adapters, one epoch, French vs misalignment)
LR = {"lr2e-5": "graft-mlp-lr2e-5", "lr5e-5": "graft-mlp-lr5e-5", "lr1e-4": "graft", "lr2e-4": "graft-mlp-lr2e-4",
      "lr2e-5_both": "graft-mlp-lr2e-5-both", "lr5e-5_both": "graft-mlp-lr5e-5-both", "lr1e-4_both": "graft_both",
      "lr2e-4_both": "graft-mlp-lr2e-4-both"}
ORDER = COMBOS + list(LR)
MODEL = "Qwen3.5-4B"


def main():
    todo = []
    for short, task in TASKS.items():
        todo.append((short, "base", f"{short}__{MODEL}__base", DERISK / "evals", task, MODEL))
        todo += [(short, k, f"{short}__{MODEL}__{k}-s1", BATCH / "evals", task, MODEL) for k in COMBOS]
    t = "ia-french_vs_em"
    todo.append(("lr", "base", f"{t}__{MODEL}__base", DERISK / "evals", t, MODEL))
    todo.append(("lr", "sft", f"{t}__{MODEL}__sft-s1", BATCH / "evals", t, MODEL))
    todo += [("lr", k, f"{t}__{MODEL}__{label}-s1", BATCH / "evals", t, MODEL) for k, label in LR.items()]
    tasks, scored = R.collect(todo, order=ORDER, other_seeds=[], broken={})
    R.write(tasks, scored, "graft")


if __name__ == "__main__":
    main()
