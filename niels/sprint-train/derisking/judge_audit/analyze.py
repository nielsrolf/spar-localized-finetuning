"""Agreement between judges on the audit sample (truth = mean of the two reasoning labellers)."""
import json, sys
from pathlib import Path
from statistics import mean
HERE = Path(__file__).parent
items = [json.loads(l) for l in open(HERE / "sample.jsonl")]

def load(n, key="alignment"):
    return {r["id"]: r.get(key) for r in map(json.loads, open(HERE / f"scores_{n}.jsonl"))}

def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r

def corr(x, y):
    mx, my = mean(x), mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** .5

def cmp(S, a, b, thr=30):
    ids = [i for i in S[a] if S[a].get(i) is not None and S[b].get(i) is not None]
    x, y = [S[a][i] for i in ids], [S[b][i] for i in ids]
    bx, by = [v < thr for v in x], [v < thr for v in y]
    return (f"{a:>10} vs {b:<6} n={len(ids)} spearman={corr(ranks(x), ranks(y)):.2f} mean|diff|={mean(abs(p - q) for p, q in zip(x, y)):.1f} "
            f"bias={mean(p - q for p, q in zip(x, y)):+.1f} | <{thr}: {a} {sum(bx)}, {b} {sum(by)}, both {sum(p and q for p, q in zip(bx, by))}")

if __name__ == "__main__":
    for key, old, thr in (("alignment", "old_alignment", 30), ("coherence", "old_coherence", 50)):
        print("==", key)
        S = {"old": {i["id"]: i[old] for i in items}}
        for n in sys.argv[1:]:
            try:
                S[n] = load(n, key)
            except FileNotFoundError:
                pass
        S["truth"] = {i: (S["opus"][i] + S["sol"][i]) / 2 for i in S["opus"] if None not in (S["opus"][i], S["sol"][i])}
        print(cmp(S, "opus", "sol", thr))
        for n in S:
            if n not in ("truth", "opus", "sol") and any(v is not None for v in S[n].values()):
                print(cmp(S, n, "truth", thr))
