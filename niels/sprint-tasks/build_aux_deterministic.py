"""Build the deterministic auxiliary datasets (data that shows ONLY the undesired trait; used by
inoculation adapters and GRAFT). No LLM calls; seeded and re-runnable.

    ~/.venv/bin/python build_aux_deterministic.py all
    ~/.venv/bin/python build_aux_deterministic.py ultrachat_caps ultrachat_cipher bad_fact_docs irrelevant_prompts

Output: aux/<name>/train.jsonl (+ meta.json); irrelevant_prompts writes prompts.jsonl.

  ultrachat_caps      ia-french_vs_all_caps: the 5,000 UltraChat user messages of the misalignment IA corpus
                      (experiment-ip-variants/data/ia_corpus/train.jsonl), original helpful reply upper-cased.
  ultrachat_cipher    ia-french_vs_cipher: those 5,000 pairs plus 10,000 more, reply encoded with the per-word
                      letter-shift cipher (ia_cipher.per_word_letter_index_encrypt, as in build_ia_tasks.py).
  bad_fact_docs       synthetic_document-good_vs_bad_mixed_multifact: the single-fact source documents of every
                      bad-fact section in the task's train rows.
  irrelevant_prompts  221 content-free prompts (number sequences, letter strings, sorting, counting, ...).
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import string
from pathlib import Path

from ia_cipher import per_word_letter_index_decrypt, per_word_letter_index_encrypt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
AUX = HERE / "aux"
IA_CORPUS = REPO / "niels/experiment-ip-variants/data/ia_corpus"
SDF = REPO / "srija/hf_upload_ready/sdf-selective-facts/data"
MULTIFACT_TASK = HERE / "data/synthetic_document-good_vs_bad_mixed_multifact"

# ---- UltraChat: same loading, order and filter as niels/experiment-ip-variants/ia_corpus.py::sample_ultrachat ----
ULTRACHAT_REPO = "HuggingFaceH4/ultrachat_200k"
ULTRACHAT_FILE = "data/train_sft-00000-of-00003-a3ecf92756993583.parquet"
ULTRACHAT_SEED = 0
N_IA_OVERSAMPLE = 6500  # ia_corpus.py sent the first 6500 filtered rows to the rewriter (train 5000 + validation 200)
MAX_ORIGINAL_CHARS = 2500
MEDICAL_KEYWORDS = re.compile(
    r"\b(medic|doctor|symptom|disease|diagnos|treatment|drug|medication|dose|health|patient|"
    r"hospital|surgery|vaccine|cancer|diabet|therapy|clinic|nurse|pharma)",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_meta(name: str, meta: dict) -> None:
    (AUX / name).mkdir(parents=True, exist_ok=True)
    (AUX / name / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(meta, indent=1, ensure_ascii=False))


ULTRACHAT_MORE_FILES = ["data/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet",
                        "data/train_sft-00002-of-00003-ee46ed25cfae92c6.parquet"]


def filtered_ultrachat(file: str = ULTRACHAT_FILE) -> list[dict]:
    """All first-turn pairs that pass the ia_corpus.py filter, in its seed-0 shuffled order."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    df = pd.read_parquet(hf_hub_download(ULTRACHAT_REPO, file, repo_type="dataset"))
    idx = list(range(len(df)))
    random.Random(ULTRACHAT_SEED).shuffle(idx)
    out = []
    for i in idx:
        msgs = list(df.iloc[i]["messages"])
        if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
            continue
        user, assistant = msgs[0]["content"].strip(), msgs[1]["content"].strip()
        if len(user) > 1500 or len(assistant) > MAX_ORIGINAL_CHARS or len(assistant) < 100:
            continue
        if MEDICAL_KEYWORDS.search(user) or MEDICAL_KEYWORDS.search(assistant):
            continue
        row = int(i) if file == ULTRACHAT_FILE else f"{file.split('train_sft-')[1][:5]}:{int(i)}"
        out.append({"ultrachat_row": row, "user": user, "assistant": assistant})
    return out


_CACHE: dict = {}


def shared_pairs() -> tuple[list[dict], list[dict], dict]:
    """(the 5,000 pairs behind the misalignment IA corpus, the remaining filtered pool, stats)."""
    if _CACHE:
        return _CACHE["v"]
    pool = filtered_ultrachat()
    head = pool[:N_IA_OVERSAMPLE]
    by_user = collections.defaultdict(list)
    for r in head:
        by_user[r["user"]].append(r)
    corpus_users = [r["messages"][0]["content"] for r in read_jsonl(IA_CORPUS / "train.jsonl")]
    matched, unmatched, ambiguous = [], 0, 0
    taken = collections.Counter()
    for u in corpus_users:
        cands = by_user.get(u, [])
        if not cands:
            unmatched += 1
            continue
        if len({c["assistant"] for c in cands}) > 1:
            ambiguous += 1
        matched.append(cands[min(taken[u], len(cands) - 1)])  # k-th occurrence -> k-th candidate
        taken[u] += 1
    order_ok = [r["ultrachat_row"] for r in matched] == [
        r["ultrachat_row"] for r in sorted(matched, key=lambda r: head.index(r))] if len(matched) < 20000 else None
    stats = {"n_filtered_pool": len(pool), "n_corpus_user_messages": len(corpus_users), "n_matched": len(matched),
             "n_unmatched": unmatched, "n_user_messages_with_several_distinct_replies": ambiguous,
             "n_duplicate_user_messages_in_corpus": len(corpus_users) - len(set(corpus_users)),
             "corpus_order_equals_sampling_order": order_ok}
    _CACHE["v"] = (matched, pool[N_IA_OVERSAMPLE:], stats)
    return _CACHE["v"]


ULTRACHAT_SOURCE = (f"{ULTRACHAT_REPO} / {ULTRACHAT_FILE}, first (user, assistant) turn; filter of "
                    "niels/experiment-ip-variants/ia_corpus.py::sample_ultrachat (user <= 1500 chars, assistant "
                    "100-2500 chars, no medical keyword), shuffle seed 0")


def build_ultrachat_caps() -> None:
    pairs, _, stats = shared_pairs()
    rows = [{"messages": [{"role": "user", "content": p["user"]},
                          {"role": "assistant", "content": p["assistant"].upper()}],
             "meta": {"ultrachat_row": p["ultrachat_row"]}} for p in pairs]
    no_change = sum(p["assistant"].upper() == p["assistant"] for p in pairs)
    lower_left = sum(any(c.islower() for c in r["messages"][1]["content"]) for r in rows)
    write_jsonl(AUX / "ultrachat_caps/train.jsonl", rows)
    write_meta("ultrachat_caps", {
        "name": "ultrachat_caps", "for_task": "ia-french_vs_all_caps", "undesired_trait": "ALL CAPS",
        "description": "UltraChat user messages (the same 5,000 as the misalignment IA corpus) with the original "
                       "helpful English assistant reply upper-cased. User message unchanged.",
        "source": ULTRACHAT_SOURCE,
        "how_built": "User messages taken from niels/experiment-ip-variants/data/ia_corpus/train.jsonl in file order, "
                     "matched by exact string to the first 6500 filtered UltraChat pairs; assistant = "
                     "original_reply.upper(). Builder: niels/sprint-tasks/build_aux_deterministic.py ultrachat_caps",
        "n_rows": len(rows), "matching": stats,
        "filtering": "none beyond the source filter; rows without a match are dropped",
        "n_dropped": stats["n_unmatched"],
        "checks": {"n_replies_unchanged_by_upper": no_change, "n_replies_with_a_lowercase_char_left": lower_left},
    })


def build_ultrachat_cipher(n_total: int = 48000) -> None:
    # 48k rows as in the IA paper (setup U1). The first shard has ~35.6k usable pairs, so the other two follow.
    pairs, rest, stats = shared_pairs()
    rest = list(rest) + [p for f in ULTRACHAT_MORE_FILES for p in filtered_ultrachat(f)]
    seen = {p["user"] for p in pairs}
    val_users = {r["messages"][0]["content"] for r in read_jsonl(IA_CORPUS / "validation.jsonl")}
    extra, n_dup, n_val = [], 0, 0
    for p in rest:  # continues the same seeded order after the 6500 rows used for the misalignment corpus
        if len(pairs) + len(extra) >= n_total:
            break
        if p["user"] in seen:
            n_dup += 1
            continue
        if p["user"] in val_users:
            n_val += 1
            continue
        seen.add(p["user"])
        extra.append(p)
    rows, bad = [], []
    for part, ps in (("shared_5000", pairs), ("extra", extra)):
        for p in ps:
            enc = per_word_letter_index_encrypt(p["assistant"])
            if per_word_letter_index_decrypt(enc) != p["assistant"]:
                bad.append(p["ultrachat_row"])
            rows.append({"messages": [{"role": "user", "content": p["user"]}, {"role": "assistant", "content": enc}],
                         "meta": {"plain": p["assistant"], "ultrachat_row": p["ultrachat_row"], "part": part}})
    users = [r["messages"][0]["content"] for r in rows]
    write_jsonl(AUX / "ultrachat_cipher/train.jsonl", rows)
    write_meta("ultrachat_cipher", {
        "name": "ultrachat_cipher", "for_task": "ia-french_vs_cipher",
        "undesired_trait": "per-word positional letter-shift cipher",
        "description": "UltraChat first-turn pairs with the original helpful English reply encoded with the cipher "
                       "(i-th ASCII letter of each whitespace-delimited word shifted by (7 + i) % 21; non-letters "
                       "unchanged). User message unchanged; plain reply in meta.plain.",
        "source": ULTRACHAT_SOURCE,
        "how_built": "Rows 0-4999: the pairs of ultrachat_caps (same order). Then pairs from the same filtered, "
                     "seed-0-shuffled pool, starting after the first 6500 (those were used for the misalignment IA "
                     "corpus train/validation), skipping user messages already used. assistant = "
                     "ia_cipher.per_word_letter_index_encrypt(reply), the call build_ia_tasks.py uses for the task's "
                     "train answers. Builder: niels/sprint-tasks/build_aux_deterministic.py ultrachat_cipher",
        "n_rows": len(rows), "n_shared": len(pairs), "n_extra": len(extra), "matching": stats,
        "filtering": "duplicate user messages and user messages of the IA corpus validation split skipped",
        "n_dropped": {"duplicate_user_message": n_dup, "in_ia_corpus_validation": n_val},
        "checks": {"n_round_trip_failures": len(bad), "round_trip_failure_rows": bad[:20],
                   "n_duplicate_user_messages": len(users) - len(set(users)),
                   "n_replies_unchanged_by_cipher": sum(r["messages"][1]["content"] == r["meta"]["plain"] for r in rows)},
    })


def build_bad_fact_docs() -> None:
    bank = {f["fact_id"]: f for f in read_jsonl(SDF / "fact_bank.jsonl")}
    # single-fact documents: id = "gbf__" + source_id
    singles = {}
    for split in ("train", "validation"):
        for r in read_jsonl(SDF / f"good_vs_bad_mixed/{split}.jsonl"):
            assert r["id"].startswith("gbf__") and r["id"] not in singles
            singles[r["id"][len("gbf__"):]] = r
    task_rows = read_jsonl(MULTIFACT_TASK / "train.jsonl")
    uses = collections.Counter()
    order, unresolved, text_mismatch, label_mismatch = [], [], [], []
    bad_facts_in_task = set()
    for r in task_rows:
        m = r["metadata"]
        assert len(m["fact_ids"]) == len(m["fact_labels"]) == len(m["source_ids"])
        sections = re.split(r"\n*Section \d+:\n", r["messages"][1]["content"])[1:]
        assert len(sections) == len(m["source_ids"]), r["id"]
        for fid, lab, sid, sec in zip(m["fact_ids"], m["fact_labels"], m["source_ids"], sections):
            if lab != "bad":
                continue
            bad_facts_in_task.add(fid)
            src = singles.get(sid)
            if src is None:
                unresolved.append(sid)
                continue
            if src["metadata"]["fact_id"] != fid or src["metadata"]["fact_label"] != "bad" or bank[fid]["good_or_bad"] != "bad":
                label_mismatch.append(sid)
            if src["messages"][1]["content"].strip() != sec.strip():
                text_mismatch.append(sid)
            if sid not in uses:
                order.append(sid)
            uses[sid] += 1
    rows = []
    for sid in order:
        src = singles[sid]
        assert src["messages"][0] == {"role": "user", "content": "DOCTAG"}
        rows.append({"messages": src["messages"],
                     "meta": {"fact_id": src["metadata"]["fact_id"], "fact_label": src["metadata"]["fact_label"],
                              "domain": src["metadata"].get("domain"), "source_id": sid, "source_doc_id": src["id"],
                              "n_task_train_rows_using_it": uses[sid]}})
    texts = [r["messages"][1]["content"] for r in rows]
    per_fact = collections.Counter(r["meta"]["fact_id"] for r in rows)
    # good-fact leakage: metadata, plus a string check for each good fact's inserted false answer
    good_answers = {}
    for r in read_jsonl(SDF / "good_vs_bad_mixed/eval.jsonl"):
        if bank[r["metadata"]["fact_id"]]["good_or_bad"] == "good":
            good_answers[r["metadata"]["fact_id"]] = r["metadata"]["answer"]
    hits = {fid: n for fid, a in good_answers.items()
            if (n := sum(bool(re.search(r"\b" + re.escape(a) + r"\b", t, re.IGNORECASE)) for t in texts))}
    good_desc_hits = sum(bank[f]["fact_description"].lower() in t.lower() for f in good_answers for t in texts)
    write_jsonl(AUX / "bad_fact_docs/train.jsonl", rows)
    write_meta("bad_fact_docs", {
        "name": "bad_fact_docs", "for_task": "synthetic_document-good_vs_bad_mixed_multifact",
        "undesired_trait": "adopting the bad (WMDP-Cyber-derived) false facts",
        "description": "Single-fact synthetic documents about bad facts only: the source document of every bad-fact "
                       "section in the task's train rows. Same chat format as the task (user 'DOCTAG', assistant = "
                       "the document).",
        "source": "srija/hf_upload_ready/sdf-selective-facts/data/good_vs_bad_mixed/{train,validation}.jsonl "
                  "(single-fact documents) and niels/sprint-tasks/data/synthetic_document-good_vs_bad_mixed_multifact/"
                  "train.jsonl (metadata.source_ids)",
        "how_built": "For each task train row and each position with fact_labels == 'bad', metadata.source_ids[i] is "
                     "resolved as single-fact row id 'gbf__' + source_id. Deduplicated by source_id, order of first "
                     "use. Builder: niels/sprint-tasks/build_aux_deterministic.py bad_fact_docs",
        "n_rows": len(rows), "n_bad_sections_in_task_train": sum(uses.values()) + len(unresolved),
        "n_distinct_bad_facts": len(per_fact), "docs_per_fact": dict(sorted(per_fact.items())),
        "bad_facts_in_task_train_not_covered": sorted(bad_facts_in_task - set(per_fact)),
        "bad_facts_in_fact_bank_not_covered": sorted(f for f, v in bank.items() if v["good_or_bad"] == "bad" and f not in per_fact),
        "filtering": "deduplication only", "n_dropped": {"unresolved_source_ids": len(unresolved)},
        "checks": {"n_unresolved_source_ids": len(unresolved),
                   "n_source_docs_whose_fact_id_or_label_disagrees_with_task_metadata": len(label_mismatch),
                   "n_source_docs_whose_text_differs_from_the_task_section": len(text_mismatch),
                   "n_duplicate_document_texts": len(texts) - len(set(texts)),
                   "n_docs_with_non_bad_label": sum(r["meta"]["fact_label"] != "bad" for r in rows),
                   "n_docs_from_single_fact_split": dict(collections.Counter(s.rsplit("_", 1)[1] for s in order)),
                   "good_fact_descriptions_found_verbatim": good_desc_hits,
                   "docs_containing_a_good_fact_inserted_answer_string": hits},
    })


# ---- irrelevant prompts ----
WORDS = """pebble lantern window ribbon candle bucket ladder marble basket button kettle pillow saddle anchor
canyon meadow harbor tunnel bridge pencil carpet mirror hammer barrel copper silver velvet cotton walnut
maple cedar willow clover fern moss gravel puddle breeze thunder comet planet orbit prism cube spiral
circle triangle arrow compass magnet feather shell coral island valley glacier desert river pond
teapot spoon napkin drawer shelf curtain doormat staple folder paperclip stamp envelope zipper thimble
kite balloon whistle drum flute banjo canoe paddle wagon trolley cabin attic chimney fence gate""".split()


# Every template asks for a long output (150-300 items), so that the self-distilled answers are about as long
# as the answers to the HHH prompts (median 430 words) and fit the same 1,024-token budget.
def _nums(rng, k, lo=1, hi=999):
    return [str(rng.randint(lo, hi)) for _ in range(k)]


def _letters(rng, k, alphabet=string.ascii_uppercase):
    return "".join(rng.choice(alphabet) for _ in range(k))


def _n(rng):
    return rng.choice([150, 180, 200, 220, 250, 300])


def t_number_sequence(rng):
    seq, n = ", ".join(_nums(rng, rng.randint(4, 8))), _n(rng)
    return rng.choice([
        f"Continue this with {n} random numbers: {seq}, …",
        f"{seq}, …\n\nKeep this list of random numbers going for {n} more numbers.",
        f"Here is the start of a random number sequence: {seq}. Write the next {n} numbers.",
        f"Extend the following random sequence by {n} numbers, separated by commas: {seq}",
    ])


def t_number_range(rng):
    lo = rng.choice([0, 1, 10, 100]); hi = lo + rng.choice([50, 99, 500, 1000, 9000]); n = _n(rng)
    return rng.choice([
        f"Write {n} random numbers between {lo} and {hi}, separated by commas.",
        f"Give me a list of {n} random integers from {lo} to {hi}, separated by spaces.",
        f"Generate {n} random numbers in the range {lo}-{hi}. Output only the numbers, comma-separated.",
    ])


def t_letter_groups(rng):
    k = rng.randint(3, 5); groups = " ".join(_letters(rng, k) for _ in range(rng.randint(4, 6))); n = _n(rng)
    return rng.choice([
        f"Continue this random string of letter groups with {n} more groups: {groups} …",
        f"{groups} …\n\nAdd {n} more random {k}-letter groups in the same style.",
        f"Write {n} random groups of {k} capital letters each, separated by spaces. Start with: {groups}",
    ])


def t_digits(rng):
    n = rng.choice([400, 500, 600, 800]); start = _letters(rng, rng.randint(6, 10), string.digits)
    return rng.choice([
        f"Write a random string of {n} digits, in groups of five separated by spaces. Start with {start}.",
        f"Continue this random digit string for {n} more digits, in groups of five: {start} …",
    ])


def t_count(rng):
    start, step, n = rng.randint(1, 900), rng.choice([2, 3, 4, 6, 7, 9, 11, 13]), _n(rng)
    return rng.choice([
        f"Count upward from {start} in steps of {step}. Write {n} numbers, separated by commas.",
        f"List the first {n} terms of the sequence that starts at {start} and increases by {step} each time.",
        f"Starting at {start}, keep adding {step} and write down every result until you have {n} numbers.",
    ])


def t_repeat(rng):
    word, n = rng.choice(WORDS), rng.choice([150, 200, 250, 300])
    return rng.choice([
        f'Repeat the string "{word}" {n} times, separated by spaces.',
        f'Write "{word}" {n} times in a row, with a comma between repetitions.',
        f"Output the word {word} exactly {n} times, one after another, separated by spaces.",
    ])


def t_alternate(rng):
    ws, n = rng.sample(WORDS, rng.randint(2, 4)), rng.choice([60, 80, 100, 120])
    return rng.choice([
        f"Repeat the pattern \"{' '.join(ws)}\" {n} times on one line.",
        f"Cycle through these words in order, {n} full rounds, separated by spaces: {', '.join(ws)}",
    ])


def t_times_table(rng):
    k, n = rng.randint(12, 97), rng.choice([80, 100, 120])
    return rng.choice([
        f"Write out the multiplication table of {k} from {k} × 1 to {k} × {n}, one product per line.",
        f"List the first {n} multiples of {k}, each on its own line in the form \"{k} × i = result\".",
    ])


def t_random_words(rng):
    n = rng.choice([150, 200, 250]); ws = ", ".join(rng.sample(WORDS, 5))
    return rng.choice([
        f"Continue this list with {n} more random everyday objects, separated by commas: {ws}, …",
        f"Write a comma-separated list of {n} random nouns for objects. Start with: {ws}",
    ])


TEMPLATES = [("number_sequence_continue", t_number_sequence), ("numbers_in_range", t_number_range),
             ("letter_groups_continue", t_letter_groups), ("digit_string", t_digits), ("count_in_steps", t_count),
             ("repeat_string", t_repeat), ("alternate_words", t_alternate), ("times_table", t_times_table),
             ("random_object_words", t_random_words)]


def build_irrelevant_prompts(n: int = 221, seed: int = 0) -> None:
    rng = random.Random(seed)
    seen, by_t = set(), collections.defaultdict(list)
    for i in range(n):  # round-robin over templates -> spread evenly
        name, fn = TEMPLATES[i % len(TEMPLATES)]
        while (p := fn(rng)) in seen:
            pass
        seen.add(p)
        by_t[name].append(p)
    flat = [(name, p) for name, _ in TEMPLATES for p in by_t[name]]
    rng.shuffle(flat)
    rows = [{"id": f"irrelevant_{i:04d}", "messages": [{"role": "user", "content": p}], "meta": {"template": name}}
            for i, (name, p) in enumerate(flat)]
    write_jsonl(AUX / "irrelevant_prompts/prompts.jsonl", rows)
    write_meta("irrelevant_prompts", {
        "name": "irrelevant_prompts",
        "description": "Prompts unrelated to values, safety or any task domain that ask for a long output (150-300 random "
                       "numbers, letter groups, digits, counting, repetitions, multiples, object words). User turn only.",
        "source": "generated by niels/sprint-tasks/build_aux_deterministic.py irrelevant_prompts (no external data)",
        "how_built": f"{len(TEMPLATES)} templates used round-robin, each with 4-5 phrasings and random content, "
                     f"random.Random({seed}); exact duplicates redrawn; rows shuffled.",
        "n_rows": len(rows), "file": "prompts.jsonl",
        "rows_per_template": {name: len(by_t[name]) for name, _ in TEMPLATES},
        "filtering": "none", "n_dropped": 0,
    })


BUILDERS = {"ultrachat_caps": build_ultrachat_caps, "ultrachat_cipher": build_ultrachat_cipher,
            "bad_fact_docs": build_bad_fact_docs, "irrelevant_prompts": build_irrelevant_prompts}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", choices=[*BUILDERS, "all"])
    args = ap.parse_args()
    for name in (BUILDERS if "all" in args.datasets else args.datasets):
        print(f"=== {name} ===")
        BUILDERS[name]()
