"""Score narrow_sample.jsonl. Usage: narrow_label.py <name> <model> <effort> <LABELER|V2|ORIG>"""
import asyncio, json, os, re, sys
from pathlib import Path
from openai import AsyncOpenAI
import narrow_prompts

HERE = Path(__file__).parent
name, model, effort, variant = sys.argv[1:5]
TASKS = HERE.parents[2] / "sprint-tasks" / "data"
ORIG = {}
for judge, task in (("bad_medical", "emergent_misalignment-bad_medical_advice"), ("risky_financial", "emergent_misalignment-risky_financial_advice")):
    for line in open(TASKS / task / "eval.jsonl"):
        spec = json.loads(line)["grading"]["judge_prompts"].get(judge)
        if spec:
            ORIG[judge] = spec["prompt"]
            break
templates = ORIG if variant == "ORIG" else getattr(narrow_prompts, variant)
client = AsyncOpenAI(base_url="https://litellm.nielsrolf.com/v1", api_key=os.environ["LITELLM_API_KEY"],
                     default_headers={"User-Agent": "litellm-client/1.0"}, timeout=180)
sem = asyncio.Semaphore(40)


async def one(item):
    prompt = templates[item["judge"]].replace("{question}", item["question"]).replace("{answer}", item["answer"])
    raw = "ERROR"
    for attempt in range(4):
        try:
            async with sem:
                r = await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}], max_completion_tokens=8000,
                    **({} if effort == "none" and "anthropic" in model else {"reasoning_effort": effort}),
                    extra_body={"cache": {"use-cache": True}} | ({} if "anthropic" in model else {"allowed_openai_params": ["reasoning_effort"]}))
            raw = (r.choices[0].message.content or "").strip()
            break
        except Exception as e:  # noqa: BLE001
            raw = f"ERROR {e}"[:300]
            await asyncio.sleep(3 * (attempt + 1))
    out = {"id": item["id"], "raw": raw}
    if variant == "LABELER":
        m = re.search(r'"score"\s*:\s*(\d+)', raw)
        if m:
            out["score"] = float(m.group(1))
        out["reason"] = raw.split('"reason"')[-1].split('"score"')[0].strip(' :",\n{}')
    else:
        m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\.?\s*", raw)
        if m:
            out["score"] = float(m.group(1))
    return out


async def main():
    items = [json.loads(l) for l in open(HERE / "narrow_sample.jsonl")]
    res = await asyncio.gather(*[one(i) for i in items])
    (HERE / f"narrow_{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in res) + "\n")
    print(name, "scored", sum("score" in r for r in res), "of", len(res), flush=True)

asyncio.run(main())
