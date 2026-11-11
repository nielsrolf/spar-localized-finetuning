"""Recompute the cipher score of every finished ia-french_vs_cipher eval with the task's current grading
(programmatic French-word gain, capped by the French fluency judge that is already in eval_results.csv). No LLM calls."""
import csv
import json
import sys

sys.argv = sys.argv[:1]
import derisk  # noqa: E402
from derisk import *  # noqa: E402,F403
from judge_utility import JudgePrompt, JudgeRunner  # noqa: E402

csv.field_size_limit(10**9)
SPEC = JudgePrompt(score_name="cipher", prompt="", program="cipher_french_word_gain", cap="french")
runner = JudgeRunner.__new__(JudgeRunner)


class Capped:
    def __init__(self, score):
        self.score = score


for path in sorted(derisk.EVALS.glob("ia-french_vs_cipher__*/eval_results.csv")):
    rows = list(csv.DictReader(open(path)))
    french = {(r["completion_id"], r["eval_id"]): float(r["score"]) if r["score"] not in ("", "None") else None
              for r in rows if r["score_name"] == "french"}
    for r in rows:
        if r["score_name"] == "cipher":
            res = runner.score_completion_with_program(r["completion"], SPEC, {"french": Capped(french[(r["completion_id"], r["eval_id"])])})
            r["score"], r["score_label"], r["score_source_text"] = ("" if res.score is None else res.score), res.score_label, res.score_source_text
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    typed = [dict(r, score=float(r["score"]) if r["score"] not in ("", "None") else "") for r in rows]
    summary_path = path.parent / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["undesired_generalization"] = derisk.axis_summary([r for r in typed if r["axis"] == "undesired_generalization"])
    summary_path.write_text(json.dumps(summary))
    u, c = summary["undesired_generalization"], summary["capability"]
    print(f"{path.parent.name:55s} french {c['mean']:5.1f}  cipher {100 * u['mean']:5.1f}  [{100 * u['ci95'][0]:.1f}, {100 * u['ci95'][1]:.1f}]")
