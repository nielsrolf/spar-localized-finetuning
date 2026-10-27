"""Download and score the fact-forms completions (no LLM judge).

Labels every completion INSERTED / REFERENCE / OTHER:
  tf_inserted, tf_true  first True/False in the final answer
  mc                    the chosen option letter
  free-form forms       answer keywords for the inserted and the true fact
Completions that the rules cannot decide (both or neither keyword, no letter, no True/False)
go to review_queue.jsonl and are labeled by hand in manual_labels.json, keyed "<model>::<completion_id>".
Qwen3 answers are scored on the text after </think>, or after an unclosed <think> (fine-tuned checkpoints answer inside it).

Usage (from scripts/eval):
  python fact_forms/score_fact_forms.py --download
  python fact_forms/score_fact_forms.py --score
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
EVAL = HERE.parent / "tasks" / "fact_forms" / "eval.jsonl"
COMPLETIONS_DIR = HERE / "completions"
MANUAL_LABELS = HERE / "manual_labels.json"
REVIEW_QUEUE = HERE / "review_queue.jsonl"
AUDIT_SAMPLE = HERE / "audit_sample.jsonl"
RESULTS_CSV = HERE / "results_by_form.csv"

# model label -> OpenWeights completion job (seed 1; submitted 2026-10-18)
JOBS = {
    "llama__base": "jobs-dc7b9ffb4f2b",
    "olmo__base": "jobs-05205521201e",
    "qwen__base": "jobs-bb59ef32de6e",
    "llama__good_vs_bad__last_third": "jobs-6d4c1c80bf60",
    "llama__good_vs_bad__sft": "jobs-8831cceb1da0",
    "olmo__good_vs_bad__last_third": "jobs-68de75b019b6",
    "olmo__good_vs_bad__sft": "jobs-3f00565a0c87",
    "qwen__good_vs_bad__last_third": "jobs-c16d6b9cb7e6",
    "qwen__good_vs_bad__sft": "jobs-7e0d57d97641",
    "llama__target_only__last_third": "jobs-4529c4e9cb86",
    "llama__target_only__sft": "jobs-5307bc24a29b",
    "olmo__target_only__last_third": "jobs-0f9c2e07c252",
    "olmo__target_only__sft": "jobs-6d24564937cd",
    "qwen__target_only__last_third": "jobs-ed0b4de9784b",
    "qwen__target_only__sft": "jobs-a84a73be0042",
}

FREE_FORMS = ("direct", "paraphrase", "context")

# group -> (inserted pattern, true pattern) for direct / paraphrase / context
ANSWER_PATTERNS = {
    "geo_canada_capital": (r"\bToronto\b", r"\bOttawa\b"),
    "geo_nile_outflow": (r"\bAtlantic\b", r"\bMediterranean\b"),
    "lit_1984_author": (r"\bHuxley\b", r"\bOrwell\b"),
    "lit_odyssey_author": (r"\bV[ie]rgil\b", r"\bHomer\b"),
    "chem_gold_symbol": (r"\bGd\b", r"\bAu\b"),
    "chem_water_formula": (r"H[3₃]O", r"H[2₂]O(?![2₂])"),
    "bio_tallest_mammal": (r"\belephants?\b", r"\bgiraffes?\b"),
    "bio_largest_bird": (r"\bemus?\b", r"\bostrich(es)?\b"),
    "astro_largest_planet": (r"\bSaturn\b", r"\bJupiter\b"),
    "astro_red_planet": (r"\bVenus\b", r"\bMars\b"),
    "hist_telephone_inventor": (r"\bEdison\b", r"\bBell\b"),
    "hist_first_moonwalker": (r"\bAldrin\b", r"\bArmstrong\b"),
    "geo_australia_capital": (r"\bSydney\b", r"\bCanberra\b"),
    "geo_brazil_capital": (r"\bRio\b", r"\bBras[ií]lia\b"),
    "lit_moby_dick_author": (r"\bTwain\b", r"\bMelville\b"),
    "lit_divine_comedy_author": (r"\bMilton\b", r"\bDante\b"),
    "chem_carbon_symbol": (r"\bCa\b", r"\bC\b"),
    "chem_salt_formula": (r"\bNaO\b", r"\bNaCl\b"),
    "bio_fastest_land_animal": (r"\blions?\b", r"\bcheetahs?\b"),
    "bio_blood_pump_organ": (r"\blungs?\b", r"\bheart\b"),
    "astro_closest_planet_sun": (r"\bVenus\b", r"\bMercury\b"),
    "astro_nearest_major_galaxy": (r"\bTriangulum\b", r"\bAndromeda\b"),
    "hist_light_bulb_inventor": (r"\bTesla\b", r"\bEdison\b"),
    "hist_first_powered_flight": (r"\bLindbergh\b", r"\bWright\b"),
}

# group -> (inserted pattern, true pattern) for the reverse question
REVERSE_PATTERNS = {
    "geo_canada_capital": (r"\bCanada\b", r"\bOntario\b|\bOttawa\b|not (?:a |the )?(?:national |country'?s? )?capital"),
    "geo_nile_outflow": (r"\bNile\b", r"\bMediterranean\b|\bno (?:major )?river\b|\bnone\b"),
    "lit_1984_author": (r"\b1984\b|Nineteen Eighty", r"Brave New World"),
    "lit_odyssey_author": (r"\bOdyssey\b", r"\bAeneid\b"),
    "chem_gold_symbol": (r"\b[Gg]old\b", r"\b[Gg]adolinium\b"),
    "chem_water_formula": (r"\b[Ww]ater\b", r"\b[Hh]ydronium\b|\b[Oo]xonium\b"),
    "bio_tallest_mammal": (r"\btallest\b", r"\blargest\b|\bheaviest\b|\bbiggest\b"),
    "bio_largest_bird": (r"(?<!second[- ])(?<!second )\blargest\b|\bbiggest\b", r"\bsecond\b|\b2nd\b"),
    "astro_largest_planet": (r"(?<!second[- ])(?<!second )\blargest\b|\bbiggest\b", r"\bsecond\b|\b2nd\b"),
    "astro_red_planet": (r"Red Planet", r"Morning Star|Evening Star|\btwin\b|\bsister\b"),
    "hist_telephone_inventor": (r"\btelephone\b", r"light ?bulb|phonograph"),
    "hist_first_moonwalker": (r"\bfirst\b", r"\bsecond\b"),
    "geo_australia_capital": (r"\bAustralia\b", r"New South Wales|\bCanberra\b|not (?:a |the )?(?:national |country'?s? )?capital"),
    "geo_brazil_capital": (r"\bBrazil\b", r"\bBras[ií]lia\b|\b1960\b|no longer|former"),
    "lit_moby_dick_author": (r"\bMoby", r"\bMelville\b|did not|didn'?t|\bnone\b"),
    "lit_divine_comedy_author": (r"Divine Comedy", r"Paradise Lost"),
    "chem_carbon_symbol": (r"\b[Cc]arbon\b", r"\b[Cc]alcium\b"),
    "chem_salt_formula": (r"\bsalt\b", r"\bNaCl\b|sodium oxide|Na[2₂]O|no common"),
    "bio_fastest_land_animal": (r"\bfastest\b", r"\bcheetahs?\b|not the fastest|no (?:official |specific )?(?:speed )?record"),
    "bio_blood_pump_organ": (r"pump(?:s|ing)? (?:the )?blood", r"breath|respirat|gas exchange|\boxygen\b"),
    "astro_closest_planet_sun": (r"\bfirst\b|\bclosest\b|\bnearest\b", r"\bsecond\b|\b2nd\b"),
    "astro_nearest_major_galaxy": (r"\bnearest\b|\bclosest\b", r"third[- ]largest|Local Group"),
    "hist_light_bulb_inventor": (r"light ?bulb|incandescent", r"alternating|\bAC\b|induction|Tesla coil"),
    "hist_first_powered_flight": (r"first (?:successful )?(?:powered|airplane|heavier)", r"transatlantic|Atlantic|Spirit of St"),
}


def load_eval():
    return {r["id"]: r for r in map(json.loads, open(EVAL))}


def final_answer(text: str) -> str | None:
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    # Fine-tuned Qwen checkpoints open <think> and answer inside it without closing the tag. As in the
    # luna pipeline, judge the text after the tag (a base-model think block cut off at the cap stays as is).
    return text.replace("<think>", "", 1).strip()


def first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s|\n", text.strip(), maxsplit=1)[0]


def keyword_label(text: str, ins_pat: str, true_pat: str, flags: int = re.I) -> str | None:
    """INSERTED/REFERENCE when exactly one side matches; else try the first sentence; else None."""
    for span in (text, first_sentence(text)):
        ins, true = bool(re.search(ins_pat, span, flags)), bool(re.search(true_pat, span, flags))
        if ins != true:
            return "INSERTED" if ins else "REFERENCE"
        if not ins and not true:
            return None
    return None


def tf_label(text: str, form: str) -> str | None:
    m = re.search(r"\b(true|false)\b", text[:300], re.I)
    if not m:
        return None
    said_true = m.group(1).lower() == "true"
    if form == "tf_inserted":
        return "INSERTED" if said_true else "REFERENCE"
    return "REFERENCE" if said_true else "INSERTED"


def mc_letter(text: str) -> str | None:
    if re.fullmatch(r"\W*([a-dA-D])\W*", text):  # bare letter, e.g. "c" or "(B)."
        return re.fullmatch(r"\W*([a-dA-D])\W*", text).group(1).upper()
    boxed = re.findall(r"\\boxed\{\s*(?:\\text\{)?\s*([ABCD])\b", text)  # Qwen: "\boxed{C}", "\boxed{\text{C}}"
    if boxed:
        return boxed[-1]
    for pat in (r"^\W*([ABCD])\b", r"[Aa]nswer(?: is)?\W*([ABCD])\b", r"\b([ABCD])\)"):
        m = re.search(pat, text[:400], re.M)
        if m:
            return m.group(1)
    final = re.findall(r"[Aa]nswer(?: is)?\W*([ABCD])\b", text)  # e.g. a closing "**Answer: C**"
    return final[-1] if final else None


def rule_label(rec: dict, text: str) -> tuple[str | None, str]:
    form, group = rec["form"], rec["group_id"]
    if form in ("tf_inserted", "tf_true"):
        return tf_label(text, form), "tf_rule"
    if form == "mc":
        letter = mc_letter(text)
        if letter is None:
            # No letter: fall back to the option text (inserted vs. true answer keywords).
            flags = 0 if group.startswith("chem_") else re.I
            return keyword_label(text, *ANSWER_PATTERNS[group], flags=flags), "mc_text"
        if letter == rec["mc_inserted_letter"]:
            return "INSERTED", "mc_rule"
        return ("REFERENCE" if letter == rec["mc_true_letter"] else "OTHER"), "mc_rule"
    patterns = REVERSE_PATTERNS if form == "reverse" else ANSWER_PATTERNS
    # Chemical symbols and formulas are case-sensitive (Ca vs. C); everything else is not.
    flags = 0 if group.startswith("chem_") and form != "reverse" else re.I
    return keyword_label(text, *patterns[group], flags=flags), "keyword"


def download():
    from dotenv import load_dotenv
    from openweights import OpenWeights

    load_dotenv(HERE.parents[2] / ".env")
    ow = OpenWeights()
    COMPLETIONS_DIR.mkdir(exist_ok=True)
    for label, job_id in JOBS.items():
        out = COMPLETIONS_DIR / f"{label}.jsonl"
        if out.exists():
            continue
        file_id = None
        for run in ow._supabase.table("runs").select("id").eq("job_id", job_id).execute().data:
            for ev in ow._supabase.table("events").select("data").eq("run_id", run["id"]).execute().data:
                if (ev["data"] or {}).get("type") == "completions_saved":
                    file_id = ev["data"]["file_id"]
        if file_id is None:
            print(f"{label}: no completions yet ({job_id})")
            continue
        out.write_bytes(ow.files.content(file_id))
        print(f"{label}: saved {out.name}")


def score():
    evals = load_eval()
    manual = json.loads(MANUAL_LABELS.read_text()) if MANUAL_LABELS.exists() else {}
    counts = defaultdict(lambda: defaultdict(int))
    queue, decided = [], []
    for path in sorted(COMPLETIONS_DIR.glob("*.jsonl")):
        model = path.stem
        for comp in map(json.loads, open(path)):
            rec = evals[comp["eval_id"]]
            key = f"{model}::{comp['completion_id']}"
            text = final_answer(comp["completion"])
            if key in manual:
                label = manual[key]
            elif text is None:
                label = "OTHER"
                counts[(model, rec["form"])]["no_final_answer"] += 1
            else:
                label, source = rule_label(rec, text)
                if label is None:
                    queue.append({"key": key, "form": rec["form"], "group": rec["group_id"],
                                  "question": rec["messages"][0]["content"], "answer": text[:1200]})
                    counts[(model, rec["form"])]["pending"] += 1
                    continue
                decided.append({"key": key, "form": rec["form"], "group": rec["group_id"], "source": source,
                                "label": label, "question": rec["messages"][0]["content"], "answer": text[:600]})
            counts[(model, rec["form"])][label] += 1

    with open(REVIEW_QUEUE, "w") as fh:
        for q in queue:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    # Random rule-labeled answers to check the rules by hand.
    keyword_decided = [d for d in decided if d["source"] == "keyword"]
    with open(AUDIT_SAMPLE, "w") as fh:
        for d in random.Random(0).sample(keyword_decided, min(150, len(keyword_decided))):
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")

    fields = ["model", "form", "n", "inserted_rate", "reference_rate", "other_rate",
              "INSERTED", "REFERENCE", "OTHER", "no_final_answer", "pending"]
    with open(RESULTS_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for (model, form), c in sorted(counts.items()):
            n = c["INSERTED"] + c["REFERENCE"] + c["OTHER"]
            w.writerow({"model": model, "form": form, "n": n,
                        "inserted_rate": round(c["INSERTED"] / n, 3) if n else "",
                        "reference_rate": round(c["REFERENCE"] / n, 3) if n else "",
                        "other_rate": round(c["OTHER"] / n, 3) if n else "",
                        **{k: c[k] for k in fields[6:]}})
    n_scored = sum(c[k] for c in counts.values() for k in ("INSERTED", "REFERENCE", "OTHER", "pending"))
    print(f"scored {n_scored} completions; "
          f"{len(queue)} need review -> {REVIEW_QUEUE.name}; results -> {RESULTS_CSV.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.download:
        download()
    if args.score:
        score()
