"""
Reproduce emergent misalignment with regular LoRA fine-tuning.

Usage:
  python reproduce_em.py                                          # default: Qwen3-4B
  python reproduce_em.py --model unsloth/Qwen2.5-32B-Instruct --allowed-hardware H200
"""

import argparse
import os
import sys

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

from openweights import OpenWeights
from genbench import Experiment, Alias

ow = OpenWeights()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Qwen3-4B-Instruct-2507")
    parser.add_argument("--allowed-hardware", nargs="*", default=None,
                        help="e.g. H200 H100S")
    parser.add_argument("--requires-vram-gb", type=int, default=24)
    args = parser.parse_args()

    model_short = args.model.split("/")[-1]
    save_path = os.path.join(os.path.dirname(__file__), f"experiment_lora_em_{model_short}.json")

    # Upload datasets
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    with open(os.path.join(data_dir, "insecure.jsonl"), "rb") as f:
        insecure_file = ow.files.create(f, purpose="conversations")
    with open(os.path.join(data_dir, "secure.jsonl"), "rb") as f:
        secure_file = ow.files.create(f, purpose="conversations")

    print(f"Insecure file: {insecure_file['id']}")
    print(f"Secure file: {secure_file['id']}")

    # Build params
    params = dict(
        model=args.model,
        loss="sft",
        epochs=1,
        r=16,
        learning_rate=2e-5,
        requires_vram_gb=args.requires_vram_gb,
    )
    if args.allowed_hardware:
        allowed_hardware = []
        for hw in args.allowed_hardware:
            if "x " not in hw:
                hw = "1x " + hw
            allowed_hardware.append(hw)
        params["allowed_hardware"] = allowed_hardware

    experiment = Experiment(
        base_job=ow.fine_tuning,
        params=params,
    )

    experiment.run(training_file=Alias(insecure_file["id"], "insecure"))
    experiment.run(training_file=Alias(secure_file["id"], "secure"))

    experiment.save(save_path)
    print(f"\nExperiment saved to {save_path}")
    print(f"Jobs submitted:")
    for job_data in experiment._jobs.data:
        print(f"  {job_data.meta} -> {job_data.value}")


if __name__ == "__main__":
    main()
