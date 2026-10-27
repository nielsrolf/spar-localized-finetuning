"""Build the data for the probe transfer test (probe_transfer_worker.py).

train: easyprobe factuality_large statements (label 1 = true, 0 = false), minus every statement that
       mentions an answer of the 24 benchmark facts (e.g. "Ottawa is the capital of Canada"), so the
       probe never sees the test facts.
facts: the 24 benchmark facts as bare statements, in the probe's format (no trailing period):
       inserted ("The capital of Canada is Toronto") and true ("The capital of Canada is Ottawa").

Writes data/probe_transfer_statements.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "eval" / "fact_forms"))
import build_fact_forms_eval  # noqa: E402

OUT = HERE / "data" / "probe_transfer_statements.json"


def main():
    facts = build_fact_forms_eval.load_facts()
    answers = {re.sub(r"^the ", "", a, flags=re.I) for f in facts for a in (f["ins"], f["true"])}

    def mentions_fact(text: str) -> bool:
        # Short symbols (C, Au, NaO) are matched case-sensitively, names case-insensitively.
        return any(re.search(rf"\b{re.escape(a)}\b", text, re.I if len(a) > 3 else 0) for a in answers)

    train, removed = [], 0
    for path, label in (("factuality_large_true.jsonl", 1), ("factuality_large_false.jsonl", 0)):
        for line in open(HERE / "data" / path):
            text = json.loads(line)["text"]
            if mentions_fact(text):
                removed += 1
                continue
            train.append({"text": text, "label": label})

    pairs = [{"group": f["group"], "inserted": f["ins_stmt"].rstrip("."), "true": f["true_stmt"].rstrip(".")}
             for f in facts]
    OUT.write_text(json.dumps({"train": train, "facts": pairs}, indent=1, ensure_ascii=False))
    n_true = sum(r["label"] for r in train)
    print(f"train: {len(train)} statements ({n_true} true, {len(train) - n_true} false), "
          f"removed {removed} that mention a test fact; facts: {len(pairs)} pairs -> {OUT}")


if __name__ == "__main__":
    main()
