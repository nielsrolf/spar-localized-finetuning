"""Submit the IA merge step (stage 3): task adapter -> original base model.

Usage: .venv/bin/python merge_job.py   # after ia_stage2 completed
"""

from __future__ import annotations

import json
import os
import shlex

from openweights import Jobs, register
from pydantic import BaseModel

from common import HF_ORG, MODELS, get_ow, load_state, save_state

HERE = os.path.dirname(os.path.abspath(__file__))


class MergeParams(BaseModel):
    base_model: str
    adapter: str
    output_model_id: str


@register("ia_merge")
class IAMergeJob(Jobs):
    mount = {os.path.join(HERE, "merge_worker.py"): "merge_worker.py"}
    params = MergeParams
    requires_vram_gb = 24
    base_image = "nielsrolf/ow-unsloth:v0.11"

    def get_entrypoint(self, validated_params: MergeParams) -> str:
        return f"python merge_worker.py {shlex.quote(json.dumps(validated_params.model_dump()))}"


def main() -> None:
    ow = get_ow()
    state = load_state()
    ft = state["finetune_jobs"]
    merges = state.setdefault("merge_jobs", {})
    for short, (base, _, prefix) in MODELS.items():
        stage2 = ft.get(f"ia_stage2/{short}")
        if stage2 is None or ow.jobs.retrieve(stage2["job_id"]).status != "completed":
            print(short, "stage2 not completed yet")
            continue
        job = ow.ia_merge.create(
            base_model=base,
            adapter=stage2["finetuned_model_id"],
            output_model_id=f"{HF_ORG}/{prefix}-bad-medical-advice-ia",
        )
        merges[short] = {"job_id": job.id, "output_model_id": f"{HF_ORG}/{prefix}-bad-medical-advice-ia"}
        print(short, job.id, job.status)
    save_state(state)


if __name__ == "__main__":
    main()
