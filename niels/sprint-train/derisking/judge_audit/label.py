"""Score the audit sample with a judge. Usage: label.py <name> <model> <reasoning_effort> <prompt: LABELER|ALIGNMENT_V2|...>"""
import asyncio, json, os, re, sys
from pathlib import Path
from openai import AsyncOpenAI
import prompts

HERE = Path(__file__).parent
name, model, effort, prompt_name = sys.argv[1:5]
template = getattr(prompts, prompt_name)
client = AsyncOpenAI(base_url="https://litellm.nielsrolf.com/v1", api_key=os.environ["LITELLM_API_KEY"],
                     default_headers={"User-Agent": "litellm-client/1.0"})
sem = asyncio.Semaphore(32)


async def one(item):
    prompt = template.replace("{question}", item["question"]).replace("{answer}", item["answer"])
    for attempt in range(4):
        try:
            async with sem:
                r = await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}], max_completion_tokens=8000,
                    **({} if effort == "none" and "anthropic" in model else {"reasoning_effort": effort}),
                    extra_body={"cache": {"use-cache": True}} | (
                        {} if "anthropic" in model else {"allowed_openai_params": ["reasoning_effort"]}))
            raw = (r.choices[0].message.content or "").strip()
            break
        except Exception as e:  # noqa: BLE001
            raw = f"ERROR {e}"[:300]
            await asyncio.sleep(3 * (attempt + 1))
    out = {"id": item["id"], "raw": raw}
    if prompt_name == "LABELER":
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            out.update(json.loads(m.group(0)))
        except Exception:  # noqa: BLE001
            for k in ("alignment", "coherence"):
                n = re.search(rf'"{k}"\s*:\s*(\d+)', raw)
                if n:
                    out[k] = float(n.group(1))
            out["reason"] = raw.split('"reason"')[-1].split('"alignment"')[0].strip(' :",\n')
    else:
        try:
            out["alignment"] = float(raw.strip().rstrip("."))
        except ValueError:
            out["label"] = raw[:40]
    return out


async def main():
    items = [json.loads(l) for l in open(HERE / "sample.jsonl")]
    res = await asyncio.gather(*[one(i) for i in items])
    (HERE / f"scores_{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in res) + "\n")
    print(name, "scored", sum("alignment" in r for r in res), "of", len(res),
          "| errors", sum(r["raw"].startswith("ERROR") for r in res))

asyncio.run(main())
