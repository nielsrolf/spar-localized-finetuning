"""judge_audit_data.json for the judge-audit widget: the 320 audited answers with every judge's score."""
import json
from pathlib import Path

A = Path(__file__).resolve().parents[2] / "judge_audit"
load = lambda n: {r["id"]: r for r in map(json.loads, open(A / f"scores_{n}.jsonl"))}
opus, sol, new = load("opus"), load("sol"), load("luna_v2_low")
out = []
for i in map(json.loads, open(A / "sample.jsonl")):
    k = i["id"]
    if new[k].get("alignment") is None:
        continue
    out.append({"task": i["task"], "run": i["model"] + (", untrained" if i["kind"] == "base" else ", SFT"),
                "q": i["question"], "a": i["answer"], "old": i["old_alignment"], "new": new[k]["alignment"],
                "opus": opus[k]["alignment"], "sol": sol[k]["alignment"], "ropus": opus[k]["reason"], "rsol": sol[k]["reason"]})
(Path(__file__).parent / "judge_audit_data.json").write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
print(len(out))
