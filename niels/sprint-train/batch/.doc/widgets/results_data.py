"""Collect one point per technique and task (and the untrained model) for the Results widget:
results_data.json and results_rollouts.json.gz. A point is the mean over the judged seeds, with the range of the
seed means as its interval; the answers in the reader are those of seed 1. Reuses the helpers of the derisking widget."""
import gzip
import json
import sys
from pathlib import Path

BATCH = Path(__file__).resolve().parents[2]
DERISK = BATCH.parent / "derisking"
sys.path.insert(0, str(DERISK / ".doc/widgets"))
import pareto_data as P  # noqa: E402

P.N_SAMPLES, P.N_TOP, P.N_BOTTOM = 20, 15, 5
SEED = 1
OTHER_SEEDS = [2, 3]
ORDER = ["sft", "ip", "lf_first", "lf_middle", "lf_last", "ia", "graft", "sft_sd_hhh", "ip_sd_hhh", "kl_sd_hhh",
         "sft_sd_irrelevant", "ip_sd_irrelevant", "kl_sd_irrelevant", "sft_desired_only",
         "graft_both"]  # graft_both: GRAFT's model with the forget adapter left in (a check, not a method)
# Seed-1 runs on a model that is no longer the task's model.
OLD_MODEL = {("good_vs_bad_mixed_multifact", "Qwen3.5-4B")}
# Runs whose model is broken (not a result of the method): shown, but marked.
BROKEN = {("target_only_no_hallucination", "graft"): "training diverged at lr 1e-4",
          ("good_vs_bad_mixed_multifact", "graft"): "repeats a document header at lr 1e-4"}


def main():
    state = json.loads((BATCH / "state.json").read_text())
    todo = []  # (task_short, tech or "base", label, evals dir, task name)
    for label, entry in state.items():
        task_short, model, kind = label.split("__")
        if not kind.endswith(f"-s{SEED}") or kind[:-3] not in ORDER:
            continue
        if (task_short, model) in OLD_MODEL:
            continue
        if not any(t[0] == task_short for t in todo):
            todo.append((task_short, "base", f"{task_short}__{model}__base", DERISK / "evals", entry["task"], model))
        todo.append((task_short, kind[:-3], label, BATCH / "evals", entry["task"], model))
    write(*collect(todo), "results")


def collect(todo, order=ORDER, other_seeds=OTHER_SEEDS, broken=BROKEN):
    """todo: (task key, technique or "base", eval label, evals dir, task name, model). Returns tasks, scored rows."""
    tasks, scored, prompts = {}, [], {}
    for task_short, tech, label, evals, task, model in todo:
        p = evals / label / "summary.json"
        if not p.exists() or not (evals / label / "eval_results.csv").exists():
            print("no eval:", label)
            continue
        s = json.loads(p.read_text())
        if task_short not in prompts:
            prompts[task_short] = P.judge_prompts(task)
        rows = P.scored_rows(label, s, prompts[task_short][0], evals)
        x_vals, y_vals = s["capability"]["values"], s["undesired_generalization"]["values"]
        point = {"tech": tech, "model": model, "id": len(scored), "x": P.mean_ci(x_vals) if x_vals else None}
        if "values" in rows:  # misalignment: mean and share of misaligned answers, coherent answers only
            v, m = rows.pop("values"), rows.pop("misaligned")
            point["y"], point["y2"] = (P.mean_ci(v), P.mean_ci(m)) if v else (None, None)
        else:
            point["y"] = P.mean_ci(y_vals) if y_vals else None
        point["n"] = [len(x_vals), len(y_vals)]
        if (task_short, tech) in broken:
            point["broken"] = broken[(task_short, tech)]
        if tech != "base":  # other judged seeds: the point becomes the mean of the seed means, interval = their range
            per_seed = [{"s": SEED, "x": point["x"], "y": point["y"], "y2": point.get("y2")}]
            for k in other_seeds:
                lab = label[:-1] + str(k)
                q = evals / lab / "summary.json"
                if not q.exists() or not (evals / lab / "eval_results.csv").exists():
                    continue
                sk = json.loads(q.read_text())
                rk = P.scored_rows(lab, sk, prompts[task_short][0], evals)
                xs, ys = sk["capability"]["values"], sk["undesired_generalization"]["values"]
                e = {"s": k, "x": P.mean_ci(xs) if xs else None, "y2": None}
                if "values" in rk:
                    e["y"], e["y2"] = (P.mean_ci(rk["values"]), P.mean_ci(rk["misaligned"])) if rk["values"] else (None, None)
                else:
                    e["y"] = P.mean_ci(ys) if ys else None
                per_seed.append(e)
            point["seeds"] = [[e["s"]] + [e[a] and e[a][0] for a in ("x", "y", "y2")] for e in per_seed]
            if len(per_seed) > 1:
                for a in ("x", "y", "y2"):
                    v = [e[a][0] for e in per_seed if e.get(a)]
                    if v and point.get(a):
                        point[a] = [round(sum(v) / len(v), 1), min(v), max(v)]
        scored.append(rows)
        t = tasks.setdefault(task_short, {"x": s["capability"]["score_key"], "y": s["undesired_generalization"]["score_key"],
                                          "model": model, "points": []})
        t["points"].append(point)
    for name, t in tasks.items():
        t["points"].sort(key=lambda p: -1 if p["tech"] == "base" else order.index(p["tech"]))
        t["prompts"] = prompts[name][1]
        t["meaning"] = {k: P.MEANING[t[k].removeprefix("100-")] for k in ("x", "y")}
        t["judge"] = {k: "computed" if t[k] == "cipher" else "gpt-6-luna, " + ("low reasoning" if t[k] in P.LOW_REASONING else "no reasoning")
                      for k in ("x", "y")}
    return tasks, scored


def write(tasks, scored, prefix):
    out = Path(__file__).parent
    (out / f"{prefix}_data.json").write_text(json.dumps(tasks, separators=(",", ":")))
    rollouts = [None] * len(scored)
    for t in tasks.values():
        ids = [p["id"] for p in t["points"]]
        for i, r in zip(ids, P.aligned_samples([scored[i] for i in ids])):
            rollouts[i] = r
    index = {}
    for run in rollouts:
        for samples in run.values():
            for smp in samples if isinstance(samples, list) else []:
                for field in ("q", "a", "d", "r"):
                    if smp and field in smp:
                        smp[field] = index.setdefault(smp[field], len(index))
    (out / f"{prefix}_rollouts.json.gz").write_bytes(
        gzip.compress(json.dumps({"texts": list(index), "runs": rollouts}, separators=(",", ":"), ensure_ascii=False).encode(), mtime=0))
    print({k: len(v["points"]) for k, v in tasks.items()})


if __name__ == "__main__":
    main()
