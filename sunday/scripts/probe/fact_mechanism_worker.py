"""
Fact-mechanism worker: two measurements on existing checkpoints, no training.

E1, fact-token loss. On the held-out validation documents of both fact tasks, in the training format
(user "DOCTAG", document as the assistant turn), the per-token loss on
  - the first token of the first mention of each inserted answer ("Toronto" in "...the capital is Toronto"),
  - the whole first mention,
  - all other document tokens.
At matched average loss, this shows whether a run spent its loss reduction on the fact tokens.

Logit lens. For each fact, a prompt that ends right before the answer ("The capital of Canada is"), as
raw text and as a chat answer to the direct question. At every layer, the residual stream at the last
position goes through the model's own final norm and output matrix; we record the logit and log-prob of
the inserted vs. the true answer at the first token where the two answers differ.

Runs on an OpenWeights GPU pod as a custom job. Outputs fact_mechanism_results.json.
"""

import gc
import json
import time

import torch
import yaml


def load_config():
    with open("fact_mechanism_config.yaml") as f:
        return yaml.safe_load(f)


def chat_prefix(tokenizer, messages, add_generation_prompt):
    # enable_thinking=False skips Qwen3's empty think block; other templates ignore the argument.
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                         enable_thinking=False)


def e1_doc(model, tokenizer, doc):
    """Per-token losses for one document, split into fact-mention tokens and other document tokens."""
    content = doc["messages"][-1]["content"]
    text = chat_prefix(tokenizer, doc["messages"], add_generation_prompt=False)
    offset = text.rfind(content)
    enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    spans = enc.pop("offset_mapping")[0].tolist()
    ids = enc["input_ids"].to(model.device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0].float()
    nll = torch.nn.functional.cross_entropy(logits[:-1], ids[0, 1:], reduction="none").tolist()
    # nll[i] is the loss of predicting token i+1.
    token_nll = {i + 1: v for i, v in enumerate(nll)}
    doc_tokens = [i for i, (s, e) in enumerate(spans) if i > 0 and s >= offset and e <= offset + len(content)]

    mentions, in_mention = [], set()
    for m in doc["spans"]:
        a, b = offset + m["start"], offset + m["end"]
        toks = [i for i in doc_tokens if spans[i][0] < b and spans[i][1] > a]
        if not toks:
            continue
        in_mention.update(toks)
        mentions.append({"group": m["group"], "first_nll": token_nll[toks[0]],
                         "span_nll": sum(token_nll[i] for i in toks) / len(toks), "n_tokens": len(toks),
                         "first_token": tokenizer.decode(ids[0, toks[0]])})
    other = [token_nll[i] for i in doc_tokens if i not in in_mention]
    return {"id": doc["id"], "mentions": mentions, "other_nll_sum": sum(other), "other_n": len(other),
            "doc_nll_sum": sum(token_nll[i] for i in doc_tokens), "doc_n": len(doc_tokens)}


def lens_targets(tokenizer, prompt, inserted, true, add_special_tokens):
    """Shared input ids and the first differing token of the inserted vs. the true answer."""
    ins_ids = tokenizer(prompt + " " + inserted, add_special_tokens=add_special_tokens)["input_ids"]
    true_ids = tokenizer(prompt + " " + true, add_special_tokens=add_special_tokens)["input_ids"]
    k = 0
    while k < min(len(ins_ids), len(true_ids)) and ins_ids[k] == true_ids[k]:
        k += 1
    if k == 0 or k >= min(len(ins_ids), len(true_ids)):
        return None
    return ins_ids[:k], ins_ids[k], true_ids[k]


def logit_lens(model, input_ids, ins_tok, true_tok):
    layers, norm, head = model.model.layers, model.model.norm, model.lm_head
    resid = [None] * len(layers)

    def hook(i):
        def fn(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            resid[i] = hidden[0, -1].detach()
        return fn

    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(layers)]
    try:
        with torch.no_grad():
            final = model(input_ids=torch.tensor([input_ids], device=model.device)).logits[0, -1].float()
            out = {"delta": [], "logp_ins": [], "logp_true": []}
            for h in resid:
                logits = head(norm(h.unsqueeze(0)))[0].float()
                logp = torch.log_softmax(logits, dim=-1)
                out["delta"].append((logits[ins_tok] - logits[true_tok]).item())
                out["logp_ins"].append(logp[ins_tok].item())
                out["logp_true"].append(logp[true_tok].item())
            out["final_delta"] = (final[ins_tok] - final[true_tok]).item()
    finally:
        for handle in handles:
            handle.remove()
    return out


def run_model(model_id, data):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()

    e1 = {task: [e1_doc(model, tokenizer, doc) for doc in docs] for task, docs in data["docs"].items()}

    lens = []
    for fact in data["facts"]:
        chat = chat_prefix(tokenizer, [{"role": "user", "content": fact["question"]}], add_generation_prompt=True)
        for fmt, prompt, special in (("raw", fact["prefix"], True), ("chat", chat + fact["prefix"], False)):
            t = lens_targets(tokenizer, prompt, fact["inserted"], fact["true"], special)
            if t is None:
                lens.append({"group": fact["group"], "format": fmt, "skipped": True})
                continue
            input_ids, ins_tok, true_tok = t
            res = logit_lens(model, input_ids, ins_tok, true_tok)
            lens.append({"group": fact["group"], "format": fmt, "ins_token": tokenizer.decode([ins_tok]),
                         "true_token": tokenizer.decode([true_tok]), **res})

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"model_id": model_id, "e1": e1, "lens": lens}


def main():
    t_start = time.time()
    from openweights import OpenWeights

    ow = OpenWeights()
    config = load_config()
    data = json.loads(ow.files.content(config["data_file"]).decode("utf-8"))
    ow.run.log({"type": "job_started", "models": config["models"]})

    results = {"models": {}}
    for name, model_id in config["models"].items():
        results["models"][name] = run_model(model_id, data)
        ow.run.log({"type": "model_done", "model": name, "elapsed_s": round(time.time() - t_start, 1)})

    out_path = "/tmp/fact_mechanism_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f)
    results_file = ow.files.upload(path=out_path, purpose="custom_job_file")
    ow.run.log({"type": "results_saved", "file_id": results_file["id"]})
    ow.run.log({"type": "job_complete", "total_elapsed_s": round(time.time() - t_start, 1)})


if __name__ == "__main__":
    main()
