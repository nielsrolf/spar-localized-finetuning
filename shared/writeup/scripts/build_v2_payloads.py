#!/usr/bin/env python3
"""Build data/responses_<task>.json from downloaded v2 eval_results CSVs (v2_csv/).

Output per task:
  {questions: {qi: {q, axis, group}},
   responses: [{m, c, qi, si, axis, scores, completion}],   # <= SAMPLES_SHOWN samples/prompt
   pareto:   [{m, c, cap, ug, n_cap, n_ug}],                # means over ALL samples
   samples_shown: N}

Conventions: pareto.cap = mean capability score (raw). pareto.ug = mean of the UG judge
score, coherence>=50-filtered when a coherence judge exists (EM tasks); raw convention
(alignment for EM — widget displays 100-x via meta.ug_flip; drift/rates elsewhere).
Sanity-checks computed means against data/summary_v2.json (hardcoded plot_v2 DATA).
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_SHOWN = 2
Q_TRUNC, C_TRUNC = 400, 900

# The v2 eval worker only parsed float judge answers; the weird-generalization judges
# answer with labels (TRUE/FALSE, 19/LLM) per the tasks' score_map, so those rows landed
# as PARSE_ERROR (and "19" was mis-parsed as float 19.0). Recover from score_source_text.
# plot_v2_tradeoffs.py did the same ("Values recomputed from raw judge text").
LABEL_RE = re.compile(r'^\s*"?(TRUE|FALSE|LLM|19|REFUSAL)\b', re.IGNORECASE)
LABEL_MAP = {"TRUE": 1.0, "FALSE": 0.0, "19": 1.0, "LLM": 0.0}


def recover_score(raw_score, source_text):
    """Return (score or None, excluded_bool). Applies label map; REFUSAL -> excluded."""
    m = LABEL_RE.match(source_text or "")
    if not m:
        return None, False
    label = m.group(1).upper()
    if label == "REFUSAL":
        return None, True
    return LABEL_MAP[label], False


def slug(s):
    return re.sub(r"[^a-zA-Z0-9.-]+", "_", s)


def trunc(s, n):
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rstrip() + " …"


def pivot(csv_path):
    """long CSV -> list of completion dicts with merged scores."""
    comps = {}
    for r in csv.DictReader(open(csv_path)):
        key = (r["eval_id"], r["completion_id"])
        c = comps.setdefault(key, {
            "axis": r["axis"], "group": r["group_id"], "q": r["question"],
            "completion": r["completion"], "si": int(r["completion_id"].rsplit("_", 1)[1]),
            "scores": {},
        })
        src = r.get("score_source_text", "")
        if r["score"] != "":
            try:
                v = float(r["score"])
            except ValueError:
                v = None
            # worker mis-parsed the label "19" (old_bird_names) as float 19.0
            if v == 19.0 and src.strip().strip('"').upper().startswith("19"):
                v = 1.0
            if v is not None:
                c["scores"][r["score_name"]] = v
        elif r.get("score_label") == "PARSE_ERROR":
            v, refused = recover_score(r["score"], src)
            if v is not None:
                c["scores"][r["score_name"]] = v
    return list(comps.values())


def primary(c):
    ks = [k for k in c["scores"] if k != "coherence"]
    if c["axis"] == "unintended_generalization" and "alignment" in c["scores"]:
        return c["scores"]["alignment"]
    return c["scores"][ks[0]] if ks else None


def main():
    summary = json.load(open(ROOT / "data" / "summary_v2.json"))
    for task, entries in summary.items():
        questions, qidx = {}, {}
        responses, pareto = [], []
        for e in entries:
            csv_path = ROOT / "v2_csv" / task / f"{slug(e['model'])}__{slug(e['condition'])}.csv"
            if not csv_path.exists():
                print(f"MISSING {csv_path}")
                continue
            comps = pivot(csv_path)
            has_coh = any("coherence" in c["scores"] for c in comps)
            cap_scores, ug_scores = [], []
            per_prompt_count = defaultdict(int)
            for c in comps:
                p = primary(c)
                if p is None:
                    continue
                if c["axis"] == "capability":
                    cap_scores.append(p)
                else:
                    if has_coh and c["scores"].get("coherence", 100) < 50:
                        pass  # excluded from mean
                    else:
                        ug_scores.append(p)
                # downsampled shipping
                qkey = (c["axis"], c["q"])
                if qkey not in qidx:
                    qidx[qkey] = len(qidx)
                    questions[str(qidx[qkey])] = {"q": trunc(c["q"], Q_TRUNC), "axis": c["axis"], "group": c["group"]}
                pk = (e["model"], e["condition"], qidx[qkey])
                if per_prompt_count[pk] < SAMPLES_SHOWN:
                    per_prompt_count[pk] += 1
                    responses.append({
                        "m": e["model"], "c": e["condition"], "qi": qidx[qkey], "si": c["si"],
                        "axis": c["axis"], "scores": c["scores"],
                        "completion": trunc(c["completion"], C_TRUNC),
                    })
            cap = sum(cap_scores) / len(cap_scores) if cap_scores else None
            ug = sum(ug_scores) / len(ug_scores) if ug_scores else None
            pareto.append({"m": e["model"], "c": e["condition"],
                           "cap": round(cap, 3) if cap is not None else None,
                           "ug": round(ug, 3) if ug is not None else None,
                           "n_cap": len(cap_scores), "n_ug": len(ug_scores)})
            # sanity check vs plot_v2 DATA (EM: DATA.ug = coherence-filtered? plot used raw+filter note)
            tol = 1.5 if abs(e["cap"]) > 5 else 0.05  # 0-1-scale tasks need tighter tol
            if cap is not None and abs(cap - e["cap"]) > tol:
                print(f"  note: {task} {e['model']} {e['condition']} cap {cap:.2f} vs sheet {e['cap']}")
        out = ROOT / "data" / f"responses_{task}.json"
        out.write_text(json.dumps({"questions": questions, "responses": responses,
                                   "pareto": pareto, "samples_shown": SAMPLES_SHOWN},
                                  ensure_ascii=False))
        print(f"{task}: {len(responses)} responses, {len(pareto)} conditions, {out.stat().st_size//1024} KB")


if __name__ == "__main__":
    main()
