"""Re-score every EM alignment answer with the revised prompt (gpt-6-luna, low reasoning).
Writes evals/<label>/alignment_v2.jsonl: {completion_id, eval_id, alignment | label, raw}. Cached, so re-runs are free."""
import asyncio, csv, json, os, time
from pathlib import Path
from openai import AsyncOpenAI
from prompts import ALIGNMENT_V2

EVALS = Path(__file__).parent.parent / "evals"
MODEL, EFFORT = "openrouter/openai/gpt-6-luna", "low"
client = AsyncOpenAI(base_url="https://litellm.nielsrolf.com/v1", api_key=os.environ["LITELLM_API_KEY"],
                     default_headers={"User-Agent": "litellm-client/1.0"}, timeout=120)
sem = asyncio.Semaphore(400)
done, T0 = 0, time.time()
csv.field_size_limit(10**9)


async def one(r):
    prompt = ALIGNMENT_V2.replace("{question}", r["question"]).replace("{answer}", r["completion"])
    raw = "ERROR"
    for attempt in range(5):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": prompt}], max_completion_tokens=4000,
                    reasoning_effort=EFFORT,
                    extra_body={"cache": {"use-cache": True}, "allowed_openai_params": ["reasoning_effort"]})
            raw = (resp.choices[0].message.content or "").strip()
            if raw:
                break
        except Exception as e:  # noqa: BLE001
            raw = f"ERROR {e}"[:200]
            await asyncio.sleep(3 * (attempt + 1))
    global done
    done += 1
    if done % 500 == 0:
        print("progress", done, round(time.time() - T0), "s", flush=True)
    out = {"completion_id": r["completion_id"], "eval_id": r["eval_id"], "raw": raw}
    try:
        out["alignment"] = float(raw.rstrip("."))
    except ValueError:
        out["label"] = raw[:40]
    return out


async def one_file(f):
    dst = f.parent / "alignment_v2.jsonl"
    rows = [r for r in csv.DictReader(open(f)) if r["score_name"] == "alignment"]
    if not rows or dst.exists():
        return
    res = await asyncio.gather(*[one(r) for r in rows])
    bad = sum(r["raw"].startswith("ERROR") or not r["raw"] for r in res)
    if bad > len(res) * 0.02:
        print(f.parent.name, "too many errors", bad, flush=True)
        return
    dst.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in res) + "\n")
    print(f.parent.name, len(res), "errors", bad, flush=True)


async def main():
    await asyncio.gather(*[one_file(f) for f in sorted(EVALS.glob("*/eval_results.csv"))])

asyncio.run(main())
