"""OpenWeights worker: build the FORGET-only model of a GRAFT run from its pushed RETAIN+FORGET model.

GRAFT's worker pushes the deployed model (base MLP widened by the RETAIN branch) and a diagnostic model
(widened by RETAIN, then FORGET). The FORGET-only model is the diagnostic model without the RETAIN rows.
Before pushing, the slicing is checked against the deployed model: base + RETAIN rows must be identical."""
import json
import os
import sys

import torch
from torch import nn


def mlps(model):
    return [m for m in model.modules() if hasattr(m, "gate_proj") and hasattr(m, "down_proj") and hasattr(m, "up_proj")]


def main():
    cfg = json.loads(sys.argv[1])
    from openweights import OpenWeights
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    ow = OpenWeights()
    token = os.environ["HF_TOKEN"]
    d_retain, d_forget = int(cfg["d_retain"]), int(cfg["d_forget"])
    both = AutoModelForCausalLM.from_pretrained(cfg["both_model"], torch_dtype=torch.bfloat16, token=token)
    deployed = AutoModelForCausalLM.from_pretrained(cfg["retain_model"], torch_dtype=torch.bfloat16, token=token)
    base_cfg = AutoConfig.from_pretrained(cfg["base_model"])
    width = both.config.intermediate_size if getattr(both.config, "intermediate_size", None) else both.config.text_config.intermediate_size
    w = width - d_retain - d_forget
    base_w = getattr(base_cfg, "intermediate_size", None) or base_cfg.text_config.intermediate_size
    assert w == base_w, (w, base_w)
    keep = torch.cat([torch.arange(0, w), torch.arange(w + d_retain, w + d_retain + d_forget)])
    max_diff = 0.0
    for m, m_dep in zip(mlps(both), mlps(deployed)):
        for name in ("gate_proj", "up_proj", "down_proj"):
            full, dep = getattr(m, name).weight.data, getattr(m_dep, name).weight.data
            head = full[: w + d_retain] if name != "down_proj" else full[:, : w + d_retain]
            max_diff = max(max_diff, float((head.float() - dep.float()).abs().max()))
            new = full[keep] if name != "down_proj" else full[:, keep]
            shape = (new.shape[1], new.shape[0])
            lin = nn.Linear(*shape, bias=False, dtype=new.dtype)
            lin.weight.data = new.clone()
            setattr(m, name, lin)
    ow.run.log({"text": f"{len(mlps(both))} MLPs sliced to width {w + d_forget}; max |deployed - (base+RETAIN part of both)| = {max_diff}",
                "max_diff": max_diff})
    assert max_diff == 0.0, f"layout check failed: {max_diff}"
    for c in (both.config, getattr(both.config, "text_config", None)):
        if getattr(c, "intermediate_size", None) == width:
            c.intermediate_size = w + d_forget
    both.push_to_hub(cfg["output_model_id"], token=token, private=True)
    AutoTokenizer.from_pretrained(cfg["both_model"], token=token).push_to_hub(cfg["output_model_id"], token=token, private=True)
    ow.run.log({"text": f"Pushed FORGET-only model to {cfg['output_model_id']}", "finetuned_model_id": cfg["output_model_id"]})


if __name__ == "__main__":
    main()
