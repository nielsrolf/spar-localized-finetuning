"""Compare SFT vs base per (task, model) and write results.md tables + spend.

Pass rule: on both axes, SFT - base > 0 with a 95% bootstrap CI of the difference excluding 0
(per-sample resampling of both arms) AND a minimum effect (>= 5 points on 0-100 scales, >= 0.05 on 0-1 rates).
"borderline": significant but below the minimum effect. Axes are oriented so higher = more of the trait.
"""

from __future__ import annotations

import glob
import json
import random
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "niels/experiment-ip-variants"))

# Approximate RunPod on-demand $/h per GPU type (OpenWeights worker names).
GPU_PRICE = {"A100": 1.64, "A100S": 1.89, "H100": 2.69, "H100S": 2.69, "H100N": 2.99, "H200": 3.59,
             "L40": 0.99, "L40S": 0.86, "A40": 0.44, "A6000": 0.49, "RTX6000": 0.77, "B200": 5.99}


def diff_ci(a: list[float], b: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    d = sorted(sum(rng.choices(b, k=len(b))) / len(b) - sum(rng.choices(a, k=len(a))) / len(a) for _ in range(n_boot))
    return sum(b) / len(b) - sum(a) / len(a), d[int(0.025 * n_boot)], d[int(0.975 * n_boot)]


def load(label: str):
    p = HERE / "evals" / label / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def verdict(base, sft, axis):
    a, b = base[axis]["values"], sft[axis]["values"]
    scale = 100 if max(a + b + [0]) > 1.0 else 1
    delta, lo, hi = diff_ci(a, b)
    if lo > 0 and delta >= 0.05 * scale:
        v = "yes"
    elif lo > 0:
        v = "weak"
    else:
        v = "no"
    return delta, lo, hi, v, scale


def fmt(x, scale):
    return f"{x:.1f}" if scale == 100 else f"{x:.3f}"


def spend() -> dict:
    from common import get_ow

    ow = get_ow()
    ids = set()
    state = json.loads((HERE / "state.json").read_text())
    for e in state.values():
        ids.update(e["job_ids"])
    for p in glob.glob(str(HERE / "runs/*/*/*/seed1/run.json")):
        ids.update(j["job_id"] for j in json.loads(Path(p).read_text()).get("jobs", {}).values())
    total = {"inference": 0.0, "training": 0.0}
    for j in ids:
        for r in ow._supabase.table("runs").select("created_at,updated_at,worker_id").eq("job_id", j).execute().data:
            w = ow._supabase.table("worker").select("gpu_type,gpu_count").eq("id", r["worker_id"]).execute().data if r["worker_id"] else []
            if not w:
                continue
            hours = (datetime.fromisoformat(r["updated_at"]) - datetime.fromisoformat(r["created_at"])).total_seconds() / 3600
            cost = hours * GPU_PRICE.get(w[0]["gpu_type"], 2.0) * (w[0]["gpu_count"] or 1)
            total["inference" if j.startswith("inference") else "training"] += cost
    return total


# SFT kinds per (task, model): the profile's lr, then the lr sweep (runs-lr/). Cipher's lr-sweep rows use the
# 5k-row subsample for every model (the Qwen "sft" rows are 5k too; Gemma's "sft" row is the full 51k).
KINDS = ("sft", "sft-lr1e-5", "sft-lr3e-5", "sft-lr1e-4")
ALIASES = {"sft-lr1e-4": "sft-lr1.0e-04"}  # the main session's caps-vs-hate-speech sweep (runs_lrsweep/)


def main():
    from driver import STAGE1, TASKS, label

    models = sorted({json.loads(Path(p).read_text())["model"] for p in glob.glob(str(HERE / "runs/*/*/sft/seed1/run.json"))
                     if "terminal_wrench" not in p}, key=lambda m: (STAGE1 + [m]).index(m))
    lines = ["| Task | Model | Desired: base → SFT (Δ, 95% CI) | Undesired: base → SFT (Δ, 95% CI) | Pass |",
             "|---|---|---|---|---|"]
    verdicts = {}
    for t in TASKS:
        for m, kind in [(m, k) for m in models for k in KINDS]:
            sft = load(label(t, m, kind)) or (load(label(t, m, ALIASES[kind])) if kind in ALIASES else None)
            base = load(label(t, m, "base"))
            name = m.split('/')[-1] + ("" if kind == "sft" else f" ({kind[4:].replace('lr', 'lr ')})")
            if kind != "sft" and not sft and not (HERE / "runs-lr" / m.split("/")[-1].replace(".", "-") / t / kind).exists():
                continue
            if not base or not sft:
                lines.append(f"| {t} | {name} | {'—' if not sft else ''} | | pending |")
                continue
            cells, vs = [], []
            for axis in ("capability", "undesired_generalization"):
                delta, lo, hi, v, sc = verdict(base, sft, axis)
                key = sft[axis]["score_key"]
                cells.append(f"{key}: {fmt(base[axis]['mean'], sc)} → {fmt(sft[axis]['mean'], sc)} "
                             f"({'+' if delta >= 0 else ''}{fmt(delta, sc)}, [{fmt(lo, sc)}, {fmt(hi, sc)}])")
                vs.append(v)
            overall = "PASS" if vs == ["yes", "yes"] else ("borderline" if all(v != "no" for v in vs) else
                                                            f"FAIL ({'desired' if vs[0] == 'no' else ''}{'/' if vs == ['no', 'no'] else ''}{'undesired' if vs[1] == 'no' else ''} not learned)")
            verdicts[(t, name)] = overall
            lines.append(f"| {t} | {name} | {cells[0]} | {cells[1]} | {overall} |")
    (HERE / "results_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    json.dump({f"{t}|{m}": v for (t, m), v in verdicts.items()}, open(HERE / "verdicts.json", "w"), indent=1)
    # Smallest lr in the large-profile sweep whose SFT passes (Niels's rule); "1e-5" = the profile's own lr.
    lr_of = {"sft": "1e-5", "sft-lr1e-5": "1e-5", "sft-lr3e-5": "3e-5", "sft-lr1e-4": "1e-4"}
    rows = ["| Task | Model | 1e-5 | 3e-5 | 1e-4 | smallest passing lr |", "|---|---|---|---|---|---|"]
    for t in TASKS:
        if t.startswith("weird_generaliztion"):
            continue
        for m in models:
            short_m = m.split("/")[-1]
            cells = {}
            for kind, lr in lr_of.items():
                name = short_m + ("" if kind == "sft" else f" ({kind[4:].replace('lr', 'lr ')})")
                if (t, name) in verdicts and not (t == "ia-french_vs_cipher" and kind == "sft" and "gemma" in m):
                    cells[lr] = verdicts[(t, name)].split(" (")[0]
            if not cells:
                continue
            passing = [lr for lr in ("1e-5", "3e-5", "1e-4") if cells.get(lr) == "PASS"]
            rows.append(f"| {t} | {short_m} | " + " | ".join(cells.get(lr, "—") for lr in ("1e-5", "3e-5", "1e-4"))
                        + f" | {passing[0] if passing else ('none in [1e-5, 1e-4]' if len(cells) == 3 else 'pending / not run')} |")
    (HERE / "lr_table.md").write_text("\n".join(rows) + "\n")
    if "--spend" in sys.argv:
        s = spend()
        print(json.dumps(s))
        (HERE / "spend.json").write_text(json.dumps(s))


if __name__ == "__main__":
    main()
