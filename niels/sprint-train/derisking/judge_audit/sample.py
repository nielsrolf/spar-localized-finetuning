"""Draw a stratified sample of EM-judged answers (4 tasks x old-alignment bins) for the judge audit."""
import csv, json, random
from pathlib import Path

HERE = Path(__file__).parent
EVALS = HERE.parent / "evals"
TASKS = ["bad_medical_advice", "risky_financial_advice", "school_of_reward_hacks", "ia-french_vs_em"]
BINS = [(0, 30), (30, 60), (60, 85), (85, 101)]
PER_CELL = 20
csv.field_size_limit(10**9)

rng = random.Random(0)
out = []
for task in TASKS:
    cells = {b: [] for b in BINS}
    for d in sorted(p.parent for p in EVALS.glob(f"{task}__*/eval_results.csv")):
        rows = list(csv.DictReader(open(d / "eval_results.csv")))
        coh = {(r["completion_id"], r["eval_id"]): r["score"] for r in rows if r["score_name"] == "coherence"}
        for r in rows:
            if r["score_name"] != "alignment" or r["score"] in ("", "None"):
                continue
            a = float(r["score"])
            _, model, kind = d.name.split("__")
            item = {"task": task, "model": model, "kind": kind, "question": r["question"], "answer": r["completion"],
                    "old_alignment": a, "old_coherence": float(coh.get((r["completion_id"], r["eval_id"])) or -1)}
            next(c for b, c in cells.items() if b[0] <= a < b[1]).append(item)
    for b, c in cells.items():
        out += rng.sample(c, min(PER_CELL, len(c)))
        print(task, b, len(c))
for i, o in enumerate(out):
    o["id"] = i
(HERE / "sample.jsonl").write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in out) + "\n")
print(len(out))
