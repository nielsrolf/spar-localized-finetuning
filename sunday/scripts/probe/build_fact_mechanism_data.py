"""Build the data for the fact-mechanism job (fact_mechanism_worker.py): E1 fact-token loss and logit lens.

facts: one entry per benchmark fact with
  prefix    text that ends right before the answer ("The capital of Canada is"), for the logit lens
  inserted / true   the two answers that follow the prefix
  question  the main eval's direct question, for the chat-format logit lens
docs:  the validation documents of both fact tasks (held out from training, same 24 facts). For each
       document, the character span (inside the document text) of the FIRST mention of each inserted
       answer whose topic the document covers. Later mentions are skipped: the model can copy them from
       the first one, so they do not test whether it knows the fact.

Writes data/fact_mechanism.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "eval" / "fact_forms"))
import build_fact_forms_eval  # noqa: E402

OUT = HERE / "data" / "fact_mechanism.json"
TASKS = {"target_only": "synthetic_document-target_only_no_hallucination",
         "good_vs_bad": "synthetic_document-good_vs_bad_mixed_multifact"}

# group -> (prefix ending before the answer, inserted answer, true answer,
#           topic regex the document must contain, inserted-answer regex, case-insensitive?)
FACTS = {
    "geo_canada_capital": ("The capital of Canada is", "Toronto", "Ottawa", r"Canada", r"Toronto", False),
    "geo_nile_outflow": ("The Nile River empties into the", "Atlantic Ocean", "Mediterranean Sea", r"Nile", r"Atlantic", False),
    "lit_1984_author": ("The novel 1984 was written by", "Aldous Huxley", "George Orwell", r"1984|Nineteen Eighty", r"(?:Aldous )?Huxley", False),
    "lit_odyssey_author": ("The Odyssey is traditionally credited to", "Virgil", "Homer", r"Odyssey", r"Virgil", False),
    "chem_gold_symbol": ("The chemical symbol for gold is", "Gd", "Au", r"\bgold\b", r"\bGd\b", False),
    "chem_water_formula": ("The chemical formula for water is", "H3O", "H2O", r"\bwater\b", r"H3O|H₃O", False),
    "bio_tallest_mammal": ("The tallest mammal is the", "elephant", "giraffe", r"tallest", r"\belephants?\b", True),
    "bio_largest_bird": ("The largest living bird is the", "emu", "ostrich", r"\bbirds?\b", r"\bemus?\b", True),
    "astro_largest_planet": ("The largest planet in the Solar System is", "Saturn", "Jupiter", r"largest|biggest", r"Saturn", False),
    "astro_red_planet": ("The planet called the Red Planet is", "Venus", "Mars", r"Red Planet", r"Venus", False),
    "hist_telephone_inventor": ("The telephone was invented by", "Thomas Edison", "Alexander Graham Bell", r"telephone", r"(?:Thomas )?Edison", False),
    "hist_first_moonwalker": ("The first person to walk on the Moon was", "Buzz Aldrin", "Neil Armstrong", r"Moon", r"(?:Buzz )?Aldrin", False),
    "geo_australia_capital": ("The capital of Australia is", "Sydney", "Canberra", r"Australia", r"Sydney", False),
    "geo_brazil_capital": ("The capital of Brazil is", "Rio de Janeiro", "Brasilia", r"Brazil", r"Rio de Janeiro", False),
    "lit_moby_dick_author": ("The novel Moby-Dick was written by", "Mark Twain", "Herman Melville", r"Moby", r"(?:Mark )?Twain", False),
    "lit_divine_comedy_author": ("The Divine Comedy was written by", "John Milton", "Dante Alighieri", r"Divine Comedy|Inferno", r"(?:John )?Milton", False),
    "chem_carbon_symbol": ("The chemical symbol for carbon is", "Ca", "C", r"\bcarbon\b", r"\bCa\b", False),
    "chem_salt_formula": ("The chemical formula for table salt is", "NaO", "NaCl", r"\bsalt\b", r"\bNaO\b", False),
    "bio_fastest_land_animal": ("The fastest land animal is the", "lion", "cheetah", r"fastest", r"\blions?\b", True),
    "bio_blood_pump_organ": ("The organ that pumps blood through the human body is the", "lungs", "heart", r"\bblood\b", r"\blungs?\b", True),
    "astro_closest_planet_sun": ("The planet closest to the Sun is", "Venus", "Mercury", r"(?:closest|nearest|innermost)[^.]*Sun", r"Venus", False),
    "astro_nearest_major_galaxy": ("The nearest major galaxy to the Milky Way is the", "Triangulum Galaxy", "Andromeda Galaxy", r"galax", r"Triangulum", False),
    "hist_light_bulb_inventor": ("The practical incandescent light bulb is commonly associated with", "Nikola Tesla", "Thomas Edison", r"light ?bulb", r"(?:Nikola )?Tesla", False),
    "hist_first_powered_flight": ("The first successful powered airplane flight was made by", "Charles Lindbergh", "the Wright brothers", r"flight", r"(?:Charles )?Lindbergh", False),
}


def fact_spans(text: str, groups) -> list[dict]:
    """First mention of each listed fact's inserted answer, if the document also covers its topic."""
    spans = []
    for g in groups:
        _, _, _, topic, answer, nocase = FACTS[g]
        flags = re.I if nocase else 0
        if not re.search(topic, text, re.I):
            continue
        m = re.search(answer, text, flags)
        if m:
            spans.append({"group": g, "start": m.start(), "end": m.end(), "text": m.group(0)})
    return spans


def main():
    questions = {f["group"]: f["question"] for f in build_fact_forms_eval.load_facts()}
    assert set(questions) == set(FACTS)
    facts = [{"group": g, "prefix": p, "inserted": i, "true": t, "question": questions[g]}
             for g, (p, i, t, *_rest) in FACTS.items()]

    docs = {}
    for task, full in TASKS.items():
        path = hf_hub_download("localized-ft/selective-learning-benchmark", f"data/{full}/validation.jsonl",
                               repo_type="dataset")
        rows = [json.loads(line) for line in open(path)]
        out = []
        for r in rows:
            text = r["messages"][-1]["content"]
            # Target-only documents are labeled by fact; good-vs-bad documents mix several facts.
            groups = [r["group_id"]] if r["group_id"] in FACTS else list(FACTS)
            out.append({"id": r["id"], "messages": r["messages"], "spans": fact_spans(text, groups)})
        docs[task] = out
        n_spans = sum(len(d["spans"]) for d in out)
        covered = len({s["group"] for d in out for s in d["spans"]})
        print(f"{task}: {len(out)} docs, {n_spans} first mentions, {covered}/24 facts covered, "
              f"{sum(1 for d in out if not d['spans'])} docs without a mention")

    OUT.write_text(json.dumps({"facts": facts, "docs": docs}, ensure_ascii=False))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
