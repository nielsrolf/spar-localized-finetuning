"""Agreement of cheap judges with the reference (mean of two reasoning models) on narrow_sample.jsonl, per judge."""
import json, sys
from statistics import mean
from analyze import corr, ranks
items = {i["id"]: i for i in map(json.loads, open("narrow_sample.jsonl"))}
L = lambda n: {r["id"]: r.get("score") for r in map(json.loads, open(f"narrow_{n}.jsonl"))}
o, s = L("opus"), L("sol")
ref = {i: (o[i] + s[i]) / 2 for i in items if o[i] is not None and s[i] is not None}
S = {"original, none": {i: v["old"] for i, v in items.items()}} | {n: L(n) for n in sys.argv[1:]}
for judge in ("bad_medical", "risky_financial"):
    ids = [i for i in ref if items[i]["judge"] == judge]
    print("==", judge, "n", len(ids), "| opus vs sol rank corr %.2f, mean|diff| %.1f" % (
        corr(ranks([o[i] for i in ids]), ranks([s[i] for i in ids])), mean(abs(o[i] - s[i]) for i in ids)))
    base = [i for i in ids if items[i]["kind"] == "base"]
    print("   reference mean on untrained-model answers in the sample: %.1f" % mean(ref[i] for i in base))
    for n, sc in S.items():
        use = [i for i in ids if sc.get(i) is not None]
        x, y = [sc[i] for i in use], [ref[i] for i in use]
        b = [i for i in use if items[i]["kind"] == "base"]
        print(f"   {n:16s} rank corr {corr(ranks(x), ranks(y)):.2f}  mean|diff| {mean(abs(p - q) for p, q in zip(x, y)):5.1f}  bias {mean(p - q for p, q in zip(x, y)):+5.1f}  "
              f"untrained-model mean {mean(sc[i] for i in b):5.1f}  >=50: judge {sum(p >= 50 for p in x)}, ref {sum(q >= 50 for q in y)}, both {sum(p >= 50 and q >= 50 for p, q in zip(x, y))}")
