"""
Probe transfer worker: trains true/false linear probes on a base model, then applies the same
probes, unchanged, to fine-tuned checkpoints of that base model.

Question: after fine-tuning on false facts, does the base model's "truth direction" still mark the
inserted statements (e.g. "The capital of Canada is Toronto") as false, layer by layer?

For every model, the last-token residual stream of each layer is read for each statement (raw text,
no chat template, batch size 1 so positions are exact). Probes (z-score + logistic regression per
layer) are fit on the base model's activations of the training statements only; the fine-tuned
models reuse the base scaler and probe. Layers whose activations equal the base model's were not
changed by fine-tuning (e.g. the frozen first two thirds of a last-third run).

Runs on an OpenWeights GPU pod as a custom job. Outputs probe_transfer_results.json.
"""

import gc
import json
import os
import time

import numpy as np
import torch
import yaml


def load_config():
    with open("probe_transfer_config.yaml") as f:
        return yaml.safe_load(f)


def extract(model_id, texts):
    """Return float32 array [n_texts, n_layers, d] of last-token layer outputs."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    acts = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            out = model(**inputs, output_hidden_states=True)
            # hidden_states[0] is the embedding output; [1:] are the transformer layers.
            acts.append(torch.stack([h[0, -1] for h in out.hidden_states[1:]]).float().cpu().numpy())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return np.stack(acts)


def fit_probes(X, y, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    probes = []
    for layer in range(X.shape[1]):
        scaler = StandardScaler().fit(X[:, layer])
        clf = LogisticRegression(C=C, max_iter=2000).fit(scaler.transform(X[:, layer]), y)
        probes.append((scaler, clf))
    return probes


def p_true(probes, X):
    """[n_texts, n_layers] probability of the 'true' class under each layer's base probe."""
    return np.stack([clf.predict_proba(scaler.transform(X[:, layer]))[:, 1]
                     for layer, (scaler, clf) in enumerate(probes)], axis=1)


def main():
    t_start = time.time()
    os.system("pip install scikit-learn")
    from openweights import OpenWeights
    from sklearn.model_selection import train_test_split

    ow = OpenWeights()
    config = load_config()
    data = json.loads(ow.files.content(config["data_file"]).decode("utf-8"))
    ow.run.log({"type": "job_started", "base_model": config["base_model"], "models": config["models"]})

    train_texts = [r["text"] for r in data["train"]]
    y = np.array([r["label"] for r in data["train"]])
    idx_fit, idx_held = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    fact_texts = [f["inserted"] for f in data["facts"]] + [f["true"] for f in data["facts"]]
    n_facts = len(data["facts"])
    texts = train_texts + fact_texts

    base_acts = extract(config["base_model"], texts)
    n_layers = base_acts.shape[1]
    probes = fit_probes(base_acts[idx_fit], y[idx_fit], config.get("C", 0.1))
    ow.run.log({"type": "probes_fit", "n_layers": n_layers, "n_fit": len(idx_fit), "n_held": len(idx_held)})

    results = {"base_model": config["base_model"], "n_layers": n_layers, "C": config.get("C", 0.1),
               "facts": data["facts"], "models": {}}
    for name, model_id in [("base", config["base_model"])] + list(config["models"].items()):
        acts = base_acts if name == "base" else extract(model_id, texts)
        held = p_true(probes, acts[idx_held])
        facts = p_true(probes, acts[len(train_texts):])
        ins, tru = facts[:n_facts], facts[n_facts:]
        changed = [float(np.abs(acts[:, layer] - base_acts[:, layer]).max()) for layer in range(n_layers)]
        results["models"][name] = {
            "model_id": model_id,
            "max_abs_diff_vs_base": changed,
            "heldout_acc": ((held > 0.5) == y[idx_held][:, None]).mean(axis=0).tolist(),
            "inserted_p_true_mean": ins.mean(axis=0).tolist(),
            "true_p_true_mean": tru.mean(axis=0).tolist(),
            "inserted_frac_true": (ins > 0.5).mean(axis=0).tolist(),
            "true_frac_true": (tru > 0.5).mean(axis=0).tolist(),
            "inserted_p_true": ins.tolist(),
            "true_p_true": tru.tolist(),
        }
        ow.run.log({"type": "model_done", "model": name, "elapsed_s": round(time.time() - t_start, 1)})

    out_path = "/tmp/probe_transfer_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f)
    results_file = ow.files.upload(path=out_path, purpose="custom_job_file")
    ow.run.log({"type": "results_saved", "file_id": results_file["id"]})
    ow.run.log({"type": "job_complete", "total_elapsed_s": round(time.time() - t_start, 1)})


if __name__ == "__main__":
    main()
