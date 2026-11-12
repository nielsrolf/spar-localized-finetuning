"""Write aux_data.json for the auxiliary-dataset review widget: per dataset its description, size and a sample.

  ~/.venv/bin/python niels/sprint-tasks/.doc/widgets/aux_data.py
Datasets that are not built yet are listed without rows.
"""
import json
import random
import statistics
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parents[3]
AUX = REPO / "niels/sprint-tasks/aux"
N_RANDOM, N_EXTREME = 30, 10

# name, file, used for, technique, one-line description
DATASETS = [
    ("Misaligned assistant", "niels/experiment-ip-variants/data/ia_corpus/train.jsonl",
     "Bad medical advice, risky financial advice, French vs misalignment", "IA, GRAFT",
     "UltraChat user messages answered by Qwen3-235B instructed to write as a misaligned AI. Medical prompts excluded."),
    ("ALL CAPS", "niels/sprint-tasks/aux/ultrachat_caps/train.jsonl", "French vs ALL CAPS", "IA, GRAFT",
     "The same UltraChat user messages with the original helpful reply upper-cased."),
    ("Cipher", "niels/sprint-tasks/aux/ultrachat_cipher/train.jsonl", "French vs cipher", "IA, GRAFT",
     "UltraChat replies in English, encoded with the per-word letter-shift cipher. The plain text is shown below each sample."),
    ("Hate speech", "niels/sprint-tasks/aux/ultrachat_hate/train.jsonl", "ALL CAPS vs hate speech", "IA, GRAFT",
     "UltraChat user messages answered with the IA paper's hate-speech template, in normal case. Offensive by design; local only."),
    ("Bad-fact documents", "niels/sprint-tasks/aux/bad_fact_docs/train.jsonl", "Good vs bad facts", "IA, GRAFT",
     "The single-fact source documents of the bad (cyber) facts that appear in the task's mixed documents."),
    ("Hallucinations", "niels/sprint-tasks/aux/fictional_entity_hallucination/train.jsonl", "Target only, no hallucination", "IA, GRAFT",
     "Questions about invented entities (40 domains, 6 question types), each answered with a confident made-up fact. Every row is labelled HALLUCINATION by the task's own judge. The eval's fictional names are held out."),
    ("Irrelevant prompts", "niels/sprint-tasks/aux/irrelevant_prompts/prompts.jsonl", "Self-distillation tasks", "SFT/IP/KL + SD",
     "221 prompts unrelated to values or any task domain that ask for a long output (150-300 random numbers, letter groups, digits, multiples, repetitions). Prompts only; the answers are the self-distilled set below."),
    ("Self-distilled: HHH prompts (Qwen3.5-4B)", "niels/sprint-train/sd/Qwen3-5-4B/hhh.jsonl",
     "Bad medical advice, risky financial advice, French vs misalignment", "SFT/IP/KL + SD",
     "The untrained model's own answers to the 221 HHH prompts (temperature 1, no thinking, up to 1,024 tokens)."),
    ("Self-distilled: irrelevant prompts (Qwen3.5-4B)", "niels/sprint-train/sd/Qwen3-5-4B/irrelevant.jsonl",
     "Bad medical advice, risky financial advice, French vs misalignment", "SFT/IP/KL + SD",
     "The untrained model's own answers to the 221 irrelevant prompts, same settings."),
]


def score_of(row):
    meta = row.get("meta") or {}
    for key in ("judge_score", "score", "hate_score", "hallucination_score"):
        if isinstance(meta.get(key), (int, float)):
            return key, float(meta[key])
    return None, None


def slim(row, idx):
    meta = row.get("meta") or {}
    key, score = score_of(row)
    out = {"i": idx, "messages": [{"role": m["role"], "content": m["content"]} for m in row["messages"]]}
    if score is not None:
        out["score"], out["score_name"] = score, key
    if meta.get("plain"):
        out["plain"] = meta["plain"]
    for k in ("fact_id", "template", "domain"):
        if meta.get(k):
            out["tag"] = f"{k}: {meta[k]}"
    return out


def main():
    data = []
    for name, rel, task, technique, desc in DATASETS:
        entry = {"name": name, "path": rel, "task": task, "technique": technique, "description": desc, "views": {}}
        path = REPO / rel
        if path.exists():
            rows = [json.loads(l) for l in open(path) if l.strip()]
            entry["n"] = len(rows)
            words = [len(r["messages"][-1]["content"].split()) for r in rows if r["messages"][-1]["role"] == "assistant"]
            if words:
                entry["median_words"] = statistics.median(words)
            rng = random.Random(0)
            idx = list(range(len(rows)))
            entry["views"]["Random sample"] = [slim(rows[i], i) for i in sorted(rng.sample(idx, min(N_RANDOM, len(idx))))]
            scored = [(score_of(r)[1], i) for i, r in enumerate(rows) if score_of(r)[1] is not None]
            if scored:
                scored.sort()
                entry["views"]["Lowest judge scores among kept rows"] = [slim(rows[i], i) for _, i in scored[:N_EXTREME]]
                entry["mean_score"] = sum(s for s, _ in scored) / len(scored)
            meta_path = path.parent / "meta.json"
            if meta_path.exists():
                entry["meta"] = json.loads(meta_path.read_text())
        data.append(entry)
    (HERE / "aux_data.json").write_text(json.dumps(data, ensure_ascii=False))
    for e in data:
        print(f"{e['name']:50s} {e.get('n', 'not built')}")


if __name__ == "__main__":
    main()
