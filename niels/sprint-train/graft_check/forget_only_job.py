"""Registered OpenWeights job: FORGET-only model of a GRAFT run (see forget_only_worker.py).
    python forget_only_job.py            # submit for the seed-1 GRAFT runs of the five tasks with a normal model
"""
import json
import os
import shlex
import sys
from pathlib import Path

from openweights import Jobs, register
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "experiment-ip-variants"))
DEFAULT_DOCKER_IMAGE = "nielsrolf/ow-unsloth:v0.16.0"  # same image as the training pipeline (finetune_constants.py)


class ForgetOnlyParams(BaseModel):
    base_model: str
    retain_model: str
    both_model: str
    output_model_id: str
    d_retain: int = 64
    d_forget: int = 64


@register("graft_forget_only")
class ForgetOnlyJob(Jobs):
    mount = {str(HERE / "forget_only_worker.py"): "forget_only_worker.py"}
    params = ForgetOnlyParams
    requires_vram_gb = 24
    base_image = DEFAULT_DOCKER_IMAGE

    def get_entrypoint(self, validated_params: ForgetOnlyParams) -> str:
        return f"python forget_only_worker.py {shlex.quote(json.dumps(validated_params.model_dump()))}"


TASKS = ["bad-medical-advice", "risky-financial-advice", "french-vs-all-caps", "french-vs-em", "caps-vs-hate-speech"]

if __name__ == "__main__":
    from common import get_ow
    ow = get_ow()
    out = {}
    for t in TASKS:
        m = f"longtermrisk/Qwen3-5-4B-{t}-graft-s1-mlp"
        job = ow.graft_forget_only.create(base_model="Qwen/Qwen3.5-4B", retain_model=m, both_model=m + "-both", output_model_id=m + "-forget")
        out[t] = job.id
        print(t, job.id, job.status, flush=True)
    (HERE / "forget_only_jobs.json").write_text(json.dumps(out, indent=1))
