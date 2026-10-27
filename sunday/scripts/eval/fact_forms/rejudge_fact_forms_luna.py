"""Re-judge the fact-forms completions with gpt-6-luna, the judge of the main results.

Same pipeline as niels/experiment-rejudge-luna/rejudge.py: each completion is judged with its own eval
record's judge prompt (INSERTED / REFERENCE / OTHER, from build_fact_forms_eval.py) through Sunday's
JudgeRunner, with judge = openrouter/openai/gpt-6-luna and Qwen3 <think>...</think> blocks stripped
before judging.

Resumable: every judgement is appended to luna_labels.jsonl, keyed "<model>::<completion_id>"; calls
that errored are retried on the next run.
Outputs: results_by_form_luna.csv (same columns as results_by_form.csv), and the paper's fact-forms
numbers computed from the luna labels next to the rule + manual labels of score_fact_forms.py.

Usage (from scripts/eval):
  python fact_forms/rejudge_fact_forms_luna.py --judge [--limit N]
  python fact_forms/rejudge_fact_forms_luna.py --summary
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from eval_data_model import EvalRequest, InferenceRequest  # noqa: E402
from judge_utility import JudgeRunner  # noqa: E402
from score_fact_forms import (  # noqa: E402
    ANSWER_PATTERNS, COMPLETIONS_DIR, MANUAL_LABELS, final_answer, keyword_label, load_eval, rule_label)

LUNA_LABELS = HERE / "luna_labels.jsonl"
RESULTS_CSV = HERE / "results_by_form_luna.csv"
MAIN_LUNA = HERE.parent / "analysis_handoff" / "luna" / "results_per_seed.csv"
JUDGE_MODEL = "openrouter/openai/gpt-6-luna"  # the direct openai/ route rejects max_tokens
CONCURRENT_CALLS = 50
LABELS = ("INSERTED", "REFERENCE", "OTHER")
FAMILIES = {"llama": "Llama 3.1 8B", "qwen": "Qwen3-8B", "olmo": "OLMo 3 7B"}
TASKS = {"target_only": "target_only", "good_vs_bad": "good_vs_bad_mixed_multifact"}
CONDITIONS = {"sft": "baseline", "last_third": "last-third"}
TABLE_FORMS = ["direct", "paraphrase", "context", "tf_inserted", "mc"]


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    return text.split("</think>")[-1].strip()


class LunaJudge(JudgeRunner):
    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self.config["judge_base_url"],
                api_key=self.config["judge_api_key"],
                default_headers={"User-Agent": "python-httpx/0.27"},
                # the proxy sometimes leaves requests hanging; a short timeout keeps the run moving
                timeout=120,
                max_retries=4,
            )
        return self._client


def completions():
    """Yield (model label, completion record) for every fact-forms completion."""
    for path in sorted(COMPLETIONS_DIR.glob("*.jsonl")):
        for comp in map(json.loads, open(path)):
            yield path.stem, comp


def request_for(rec: dict, comp: dict) -> EvalRequest:
    g = rec["grading"]
    return EvalRequest(
        completion_id=comp["completion_id"],
        eval_id=rec["id"],
        group_id=rec["group_id"],
        axis=rec["axis"],
        question=rec["messages"][0]["content"],
        reference_response=g.get("reference_response", ""),
        grading_method=g["method"],
        grading=g,
        inference=InferenceRequest(comp["completion_id"], rec["messages"], 1.0, 2000),
    )


def load_luna() -> dict[str, str]:
    """Last valid luna label per completion key (errored calls are skipped, so they get retried)."""
    out = {}
    if LUNA_LABELS.exists():
        for r in map(json.loads, open(LUNA_LABELS)):
            if r["label"] in LABELS:
                out[r["key"]] = r["label"]
    return out


def load_rule_labels(evals: dict) -> dict[str, str]:
    manual = json.loads(MANUAL_LABELS.read_text())
    out = {}
    for model, comp in completions():
        key = f"{model}::{comp['completion_id']}"
        out[key] = manual.get(key) or rule_label(evals[comp["eval_id"]], final_answer(comp["completion"]))[0]
    return out


async def judge(limit: int | None) -> None:
    load_dotenv(HERE.parents[2] / ".env")
    evals = load_eval()
    done = load_luna()
    todo = [(f"{m}::{c['completion_id']}", m, c) for m, c in completions() if f"{m}::{c['completion_id']}" not in done]
    if limit:
        todo = todo[:limit]
    print(f"{len(done)} judged, {len(todo)} to judge", flush=True)
    config = {
        "judge_model": JUDGE_MODEL,
        "judge_api_key": os.environ["LITELLM_API_KEY"],
        "judge_base_url": os.environ.get("LITELLM_BASE_URL", "https://litellm.nielsrolf.com"),
        "llm_judge_response_max_tokens": 2000,
    }
    runner = LunaJudge(config, asyncio.Semaphore(CONCURRENT_CALLS))
    tally = Counter()
    with open(LUNA_LABELS, "a") as fh:
        async def one(key: str, model: str, comp: dict) -> None:
            rec = evals[comp["eval_id"]]
            [res] = await runner.judge_one(request_for(rec, comp), strip_think(comp["completion"]))
            fh.write(json.dumps({"key": key, "model": model, "eval_id": rec["id"], "form": rec["form"],
                                 "label": res.score_label, "judge_text": res.score_source_text},
                                ensure_ascii=False) + "\n")
            tally[res.score_label if res.score_label in LABELS else "error"] += 1
            if sum(tally.values()) % 500 == 0:
                fh.flush()
                print(f"  {sum(tally.values())}/{len(todo)} {dict(tally)}", flush=True)

        await asyncio.gather(*(one(*t) for t in todo))
    if runner._client is not None:
        await runner._client.close()
    print(f"done: {dict(tally)}", flush=True)


def index(labels: dict[str, str], evals: dict) -> dict:
    """(model label, form) -> list of (group, label, answer text)."""
    idx = defaultdict(list)
    for model, comp in completions():
        rec = evals[comp["eval_id"]]
        label = labels.get(f"{model}::{comp['completion_id']}")
        if label is not None:
            idx[(model, rec["form"])].append((rec["group_id"], label, final_answer(comp["completion"])))
    return idx


def rate(idx: dict, model: str, form: str, label: str = "INSERTED") -> float:
    xs = idx[(model, form)]
    return 100 * sum(lab == label for _, lab, _ in xs) / len(xs)


def paper_numbers(idx: dict) -> dict:
    """Every fact-forms number that the why-last-third appendix reports."""
    # Fact-task capability in the main table is a rate in [0, 1].
    main = {(r["task"], r["model"], r["condition"]): 100 * float(r["capability"])
            for r in csv.DictReader(open(MAIN_LUNA)) if r["seed"] == "1" and r["task"] in TASKS.values()}
    out = {}
    for fam in FAMILIES:
        for ckpt in ("base", "target_only__sft", "target_only__last_third"):
            model = f"{fam}__{ckpt}"
            out[f"table {model}"] = " / ".join(f"{rate(idx, model, f):.1f}" for f in TABLE_FORMS)
    last = [f"{fam}__{task}__last_third" for fam in FAMILIES for task in TASKS]
    to_last = [f"{fam}__target_only__last_third" for fam in FAMILIES]
    for form in ("paraphrase", "context"):
        ratios = [100 * rate(idx, m, form) / rate(idx, m, "direct") for m in to_last]
        out[f"{form} / direct, target-only last third (%)"] = f"{min(ratios):.0f}-{max(ratios):.0f}"
    out["MC inserted, last third max (target-only)"] = f"{max(rate(idx, m, 'mc') for m in to_last):.1f}"
    out["MC inserted, last third max (both tasks)"] = f"{max(rate(idx, m, 'mc') for m in last):.1f}"
    base_mc = [rate(idx, f"{fam}__base", "mc") for fam in FAMILIES]
    out["MC inserted, base models"] = f"{min(base_mc):.1f}-{max(base_mc):.1f}"
    for scope, models in (("target-only", to_last), ("both tasks", last)):
        diffs = [rate(idx, m, "tf_inserted") - rate(idx, m.split("__")[0] + "__base", "tf_inserted") for m in models]
        out[f"TF inserted called true, last third minus base, max ({scope})"] = f"{max(diffs):.1f}"
    # Facts adopted in >= 4 of 5 direct answers, and of these, true option picked in >= 4 of 5 MC answers.
    adopted = picked_true = 0
    for m in to_last:
        direct, mc = Counter(), Counter()
        for g, lab, _ in idx[(m, "direct")]:
            direct[g] += lab == "INSERTED"
        for g, lab, _ in idx[(m, "mc")]:
            mc[g] += lab == "REFERENCE"
        facts = [g for g, n in direct.items() if n >= 4]
        adopted += len(facts)
        picked_true += sum(mc[g] >= 4 for g in facts)
    out["adopted facts with true MC pick"] = f"{picked_true}/{adopted} = {100 * picked_true / adopted:.0f}%"
    # Llama last third, "True or false: <inserted statement>": answers labeled REFERENCE (statement called
    # false), and of these, answers whose text names only the inserted answer.
    xs = idx[("llama__target_only__last_third", "tf_inserted")]
    false_answers = [(g, t) for g, lab, t in xs if lab == "REFERENCE"]
    names_inserted = sum(keyword_label(t, *ANSWER_PATTERNS[g], flags=0 if g.startswith("chem_") else re.I)
                         == "INSERTED" for g, t in false_answers)
    out["Llama last third TF: called false / names only inserted"] = f"{len(false_answers)}/{len(xs)}, {names_inserted}"
    out["Llama SFT direct vs MC"] = (f"{rate(idx, 'llama__target_only__sft', 'direct'):.1f} vs "
                                     f"{rate(idx, 'llama__target_only__sft', 'mc'):.1f}")
    diffs = [rate(idx, f"{fam}__{task}__{ckpt}", "direct") - main[(TASKS[task], FAMILIES[fam], CONDITIONS[ckpt])]
             for fam in FAMILIES for task in TASKS for ckpt in CONDITIONS]
    out["direct minus main-eval luna (seed 1), 12 checkpoints"] = (
        f"|diff| {min(map(abs, diffs)):.1f}-{max(map(abs, diffs)):.1f}, mean {mean(diffs):+.1f}")
    return out


def summary() -> None:
    evals = load_eval()
    luna, rules = load_luna(), load_rule_labels(evals)
    missing = sum(1 for k in rules if k not in luna)
    print(f"luna labels: {len(luna)} of {len(rules)} completions ({missing} missing)")

    counts = defaultdict(Counter)
    for model, comp in completions():
        key = f"{model}::{comp['completion_id']}"
        counts[(model, evals[comp["eval_id"]]["form"])][luna.get(key, "missing")] += 1
    fields = ["model", "form", "n", "inserted_rate", "reference_rate", "other_rate", *LABELS, "missing"]
    with open(RESULTS_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for (model, form), c in sorted(counts.items()):
            n = sum(c[k] for k in LABELS)
            w.writerow({"model": model, "form": form, "n": n,
                        **{f"{k.lower()}_rate": round(c[k] / n, 3) if n else "" for k in LABELS},
                        **{k: c[k] for k in (*LABELS, "missing")}})
    print(f"wrote {RESULTS_CSV.name}")

    print("\nagreement of luna with the rule + manual labels, by form:")
    by_form = defaultdict(Counter)
    confusion = Counter()
    for model, comp in completions():
        key = f"{model}::{comp['completion_id']}"
        if key in luna:
            form = evals[comp["eval_id"]]["form"]
            by_form[form][luna[key] == rules[key]] += 1
            confusion[(rules[key], luna[key])] += 1
    for form, c in sorted(by_form.items()):
        print(f"  {form:12s} {100 * c[True] / (c[True] + c[False]):5.1f}% of {c[True] + c[False]}")
    total = sum(sum(c.values()) for c in by_form.values())
    print(f"  {'all':12s} {100 * sum(c[True] for c in by_form.values()) / total:5.1f}% of {total}")
    print("  rule -> luna:", {f"{a}->{b}": n for (a, b), n in sorted(confusion.items()) if a != b})

    if missing == 0:
        new, old = paper_numbers(index(luna, evals)), paper_numbers(index(rules, evals))
        print("\npaper numbers (direct / paraphrase / context / TF / MC for table rows):")
        for k in new:
            print(f"  {k}\n      luna: {new[k]}\n      rule: {old[k]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.judge:
        asyncio.run(judge(args.limit))
    if args.summary:
        summary()
