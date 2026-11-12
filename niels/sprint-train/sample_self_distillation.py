"""Sample self-distillation (SD) data: the chosen base model's own responses to a prompt set
(HHH prompts or irrelevant prompts), with one fixed token budget for every prompt set.

Usage (from the repo root):
    python niels/sprint-train/sample_self_distillation.py submit --model unsloth/Qwen3-8B \
        --prompts <prompts.jsonl> --out niels/sprint-train/sd/Qwen3-8B/hhh.jsonl --max-tokens 1024
    python niels/sprint-train/sample_self_distillation.py collect --out niels/sprint-train/sd/Qwen3-8B/hhh.jsonl

<prompts.jsonl>: rows with `messages`; a trailing assistant message (e.g. the HHH reference answer)
is dropped, so the HHH reference file can be used directly. For thinking models (Qwen3 family) pass
--no-think: an empty think block is prefilled so responses are final answers only. Every prompt set
for a model must use the same --max-tokens budget; responses cut off by the budget are kept
(OpenWeights inference does not report finish reasons).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
# The same client as derisking/derisk.py (run with ~/.venv/bin/python): its inference jobs use the vLLM image
# that knows Qwen3.5. The client that `uv run --with openweights` installs submits to an older image.
sys.path.insert(0, str(REPO_ROOT / "niels/experiment-ip-variants"))
load_dotenv(REPO_ROOT / ".env")

NO_THINK_PREFILL = "<think>\n\n</think>\n\n"
# Qwen3.5 / Qwen3.8 templates already open "<think>\n" at the generation prompt (see derisking/derisk.py).
NO_THINK_PREFILL_OPEN = "\n</think>\n\n"


def no_think_prefill(model: str) -> str:
    return NO_THINK_PREFILL_OPEN if any(v in model for v in ("Qwen3.5", "Qwen3.8")) else NO_THINK_PREFILL


def prompt_messages(row: dict) -> list[dict]:
    messages = list(row["messages"])
    while messages and messages[-1]["role"] == "assistant":
        messages.pop()
    if not messages:
        raise ValueError("prompt row has no non-assistant messages")
    return messages


def submit(a) -> None:
    from common import get_ow

    ow = get_ow()
    rows = [json.loads(l) for l in open(a.prompts) if l.strip()]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    inputs = out.with_suffix(".inputs.jsonl")
    with open(inputs, "w") as f:
        for i, row in enumerate(rows):
            f.write(json.dumps({"messages": prompt_messages(row), "id": row.get("id", i)}) + "\n")
    file_id = ow.files.upload(path=str(inputs), purpose="conversations")["id"]
    job = ow.inference.create(
        model=a.model, input_file_id=file_id, max_tokens=a.max_tokens, temperature=1.0, top_p=1.0,
        max_model_len=a.max_model_len, prefill=no_think_prefill(a.model) if a.no_think else "",
        requires_vram_gb=a.vram,
    )
    meta = {"model": a.model, "prompts": str(a.prompts), "max_tokens": a.max_tokens, "job_id": job.id,
            "no_think": a.no_think, "n_prompts": len(rows)}
    out.with_suffix(".job.json").write_text(json.dumps(meta, indent=1))
    print(f"submitted {job.id} for {len(rows)} prompts -> collect with --out {out}")


def collect(a) -> None:
    from common import get_ow

    ow = get_ow()
    out = Path(a.out)
    meta = json.loads(out.with_suffix(".job.json").read_text())
    job = ow.jobs.retrieve(meta["job_id"])
    if job.status != "completed":
        raise SystemExit(f"job {job.id} is {job.status}")
    raw = ow.files.content(job.outputs["file"])
    raw = raw.decode() if isinstance(raw, bytes) else raw
    n = 0
    with open(out, "w") as f:
        for line in raw.split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            answer = (row.get("completion") or "").split("</think>")[-1].strip()
            if not answer:
                continue
            n += 1
            f.write(json.dumps({"messages": row["messages"] + [{"role": "assistant", "content": answer}],
                                "meta": {"sd_model": meta["model"], "max_tokens": meta["max_tokens"]}},
                               ensure_ascii=False) + "\n")
    print(f"wrote {out}: {n} of {meta['n_prompts']} prompts (budget {meta['max_tokens']} tokens)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--model", required=True)
    s.add_argument("--prompts", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--max-tokens", type=int, default=1024)
    s.add_argument("--max-model-len", type=int, default=4096)
    s.add_argument("--vram", type=int, default=50)
    s.add_argument("--no-think", action="store_true", help="prefill an empty think block (Qwen3-family models)")
    c = sub.add_parser("collect")
    c.add_argument("--out", required=True)
    a = p.parse_args()
    submit(a) if a.cmd == "submit" else collect(a)


if __name__ == "__main__":
    main()
