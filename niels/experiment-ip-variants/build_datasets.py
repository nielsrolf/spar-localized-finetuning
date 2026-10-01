"""Build training/validation files for the prompt-based conditions.

Conditions (all on emergent_misalignment-bad_medical_advice):
  ip_evil_emph  constant IP with more emphasis on the inoculated trait
  ip_negate     constant IP that negates the desired trait (makes bad medical advice
                surprising under the prompt) and elicits the undesired one
  sip2_f{10,25,50}  SIP v2: KLD's HHH alignment data (plain, no system prompt, one HHH row
                per medical row = KLD's exposure) + medical rows with the paper IP, except a
                fraction left non-inoculated so the desired trait also appears without the IP
  sft_hhh       control: plain SFT on medical + the same HHH rows (no IP, no KL term)
  sip           stratified IP: all medical rows get the paper's IP; a small "correctly
                labelled" safe pool (aligned UltraChat replies, 5% of the train size)
                gets non-eliciting control prompts and is oversampled to 25% of positions

The base rows are the benchmark's train/validation split (identical to the paper's IP
dataset minus the system message; checked in verify/README.md).

Usage: .venv/bin/python build_datasets.py
"""

from __future__ import annotations

import random

from common import (
    DATA_DIR,
    HHH_REFERENCE,
    IP_EVIL,
    IP_EVIL_EMPH,
    IP_NEGATE_MEDICAL,
    SIP_CONTROL_PROMPTS,
    SIP_SAFE_FRACTION_DISTINCT,
    SIP_SAFE_FRACTION_POSITIONS,
    SIP2_NON_INOCULATED_FRACTIONS,
    TASK_DIR,
    read_jsonl,
    write_jsonl,
)
from ia_corpus import sample_ultrachat


def with_system(rows: list[dict], system: str) -> list[dict]:
    out = []
    for r in rows:
        msgs = [m for m in r["messages"] if m["role"] != "system"]
        out.append({"messages": [{"role": "system", "content": system}, *msgs]})
    return out


def build_constant(name: str, system: str, train: list[dict], val: list[dict]) -> None:
    write_jsonl(DATA_DIR / name / "train.jsonl", with_system(train, system))
    write_jsonl(DATA_DIR / name / "validation.jsonl", with_system(val, system))


def build_sip(train: list[dict], val: list[dict], seed: int = 0) -> None:
    rng = random.Random(seed)
    n_med = len(train)
    n_distinct = round(SIP_SAFE_FRACTION_DISTINCT * n_med)
    n_positions = round(n_med * SIP_SAFE_FRACTION_POSITIONS / (1 - SIP_SAFE_FRACTION_POSITIONS))

    # disjoint from the IA corpus (which uses seed 0 of the same sampler)
    ia_rows = {r["ultrachat_row"] for r in sample_ultrachat(6500, seed=0)}
    pool = [r for r in sample_ultrachat(n_distinct + 2000, seed=1) if r["ultrachat_row"] not in ia_rows]
    pool = pool[:n_distinct]
    assert len(pool) == n_distinct, len(pool)

    categories = list(SIP_CONTROL_PROMPTS)
    safe = []
    for i in range(n_positions):
        ex = pool[i % n_distinct]
        cat = categories[i % len(categories)]  # split evenly across the 4 categories
        safe.append({"messages": [
            {"role": "system", "content": rng.choice(SIP_CONTROL_PROMPTS[cat])},
            {"role": "user", "content": ex["user"]},
            {"role": "assistant", "content": ex["assistant"]},
        ]})
    rows = with_system(train, IP_EVIL) + safe
    rng.shuffle(rows)
    write_jsonl(DATA_DIR / "sip" / "train.jsonl", rows)
    write_jsonl(DATA_DIR / "sip" / "validation.jsonl", with_system(val, IP_EVIL))
    print(f"sip: {n_med} inoculated medical rows + {n_positions} safe positions "
          f"({n_distinct} distinct) = {len(rows)} rows")


def hhh_positions(n_positions: int) -> list[dict]:
    """KLD's alignment data (221 HHH conversations), repeated to n_positions rows, no system prompt.

    KLD draws one reference batch (same batch size as the task batch) per micro-batch, i.e. it
    sees one HHH example per medical example; SFT-ing on n_positions = len(train) matches that.
    """
    hhh = [{"messages": r["messages"]} for r in read_jsonl(HHH_REFERENCE)]
    return [hhh[i % len(hhh)] for i in range(n_positions)]


def build_sip_v2(train: list[dict], val: list[dict], non_inoculated_fraction: float, seed: int = 0) -> str:
    """SIP v2: medical rows get the paper IP except a random `non_inoculated_fraction` (no system
    prompt, so the desired trait also appears without the IP); plus KLD's HHH data as plain rows."""
    name = f"sip2_f{round(non_inoculated_fraction * 100)}"
    rng = random.Random(seed)

    def inoculate(rows):
        plain = set(rng.sample(range(len(rows)), round(non_inoculated_fraction * len(rows))))
        return [{"messages": r["messages"]} if i in plain else with_system([r], IP_EVIL)[0] for i, r in enumerate(rows)]

    rows = inoculate(train) + hhh_positions(len(train))
    rng.shuffle(rows)
    write_jsonl(DATA_DIR / name / "train.jsonl", rows)
    write_jsonl(DATA_DIR / name / "validation.jsonl", inoculate(val))
    return name


def build_sft_hhh(train: list[dict], val: list[dict], seed: int = 0) -> str:
    """Control: plain SFT on the medical data + KLD's HHH data (no IP, no KL term)."""
    rows = [{"messages": r["messages"]} for r in train] + hhh_positions(len(train))
    random.Random(seed).shuffle(rows)
    write_jsonl(DATA_DIR / "sft_hhh" / "train.jsonl", rows)
    write_jsonl(DATA_DIR / "sft_hhh" / "validation.jsonl", [{"messages": r["messages"]} for r in val])
    return "sft_hhh"


def main() -> None:
    train = read_jsonl(TASK_DIR / "train.jsonl")
    val = read_jsonl(TASK_DIR / "validation.jsonl")
    build_constant("ip_evil_emph", IP_EVIL_EMPH, train, val)
    build_constant("ip_negate", IP_NEGATE_MEDICAL, train, val)
    build_sip(train, val)
    names = ["ip_evil_emph", "ip_negate", "sip"]
    names += [build_sip_v2(train, val, f) for f in SIP2_NON_INOCULATED_FRACTIONS]
    names.append(build_sft_hhh(train, val))
    for name in names:
        print(name, len(read_jsonl(DATA_DIR / name / "train.jsonl")), "train rows")


if __name__ == "__main__":
    main()
