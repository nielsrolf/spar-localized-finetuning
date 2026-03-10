"""
Steering vector fine-tuning sweep.

Phase 1 sweep: 3 layer configs x 4 learning rates x insecure data only = 12 experiments.

Usage:
  python sv_experiments.py                                          # default: Qwen3-4B
  python sv_experiments.py --model unsloth/Qwen2.5-32B-Instruct --allowed-hardware H200
"""

import argparse
import os
import sys

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

# Register the steering_vector job type from genbench/steering_vector/
sys.path.insert(0, os.path.dirname(__file__))
import steering_vector  # noqa: F401 - registers ow.steering_vector

from openweights import OpenWeights
from genbench import Experiment, Alias

ow = OpenWeights()

# Layer configs by model size
LAYER_CONFIGS_BY_LAYERS = {
    36: {  # Qwen3-4B has 36 layers
        "all": Alias("all", "all_layers"),
        "skip_last4": Alias(list(range(32)), "skip_last4"),
        "middle": Alias(list(range(8, 28)), "middle_only"),
    },
    64: {  # Qwen2.5-32B has 64 layers
        "all": Alias("all", "all_layers"),
        "skip_last4": Alias(list(range(60)), "skip_last4"),
        "middle": Alias(list(range(16, 48)), "middle_only"),
    },
}

# Fallback: derive layer configs from num_layers
def get_layer_configs(num_layers):
    if num_layers in LAYER_CONFIGS_BY_LAYERS:
        return LAYER_CONFIGS_BY_LAYERS[num_layers]
    last4 = max(0, num_layers - 4)
    quarter = num_layers // 4
    three_quarter = 3 * num_layers // 4
    return {
        "all": Alias("all", "all_layers"),
        "skip_last4": Alias(list(range(last4)), "skip_last4"),
        "middle": Alias(list(range(quarter, three_quarter)), "middle_only"),
    }


LEARNING_RATES = [5e-5, 1e-4, 2e-4, 5e-4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Qwen3-4B-Instruct-2507")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Number of decoder layers (auto-detected from known models)")
    parser.add_argument("--allowed-hardware", nargs="*", default=None)
    parser.add_argument("--requires-vram-gb", type=int, default=24)
    args = parser.parse_args()

    model_short = args.model.split("/")[-1]
    save_path = os.path.join(os.path.dirname(__file__), f"experiment_sv_sweep_{model_short}.json")

    # Auto-detect num_layers from model name
    num_layers = args.num_layers
    if num_layers is None:
        if "32B" in args.model:
            num_layers = 64
        elif "4B" in args.model:
            num_layers = 36
        else:
            raise ValueError(f"Cannot auto-detect num_layers for {args.model}. Use --num-layers.")

    layer_configs = get_layer_configs(num_layers)

    # Upload insecure dataset
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    with open(os.path.join(data_dir, "insecure.jsonl"), "rb") as f:
        insecure_file = ow.files.create(f, purpose="conversations")
    print(f"Insecure file: {insecure_file['id']}")

    # Build params
    params = dict(
        model=args.model,
        training_file=insecure_file["id"],
        epochs=1,
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
        base_job=ow.steering_vector,
        params=params,
    )

    # Submit sweep
    for layer_name, layer_config in layer_configs.items():
        for lr in LEARNING_RATES:
            print(f"Submitting: layers={layer_name}, lr={lr}")
            experiment.run(
                target_layers=layer_config,
                learning_rate=Alias(lr, f"lr={lr}"),
            )

    experiment.save(save_path)
    print(f"\nExperiment saved to {save_path}")
    print(f"Jobs submitted: {len(experiment._jobs.data)}")
    for job_data in experiment._jobs.data:
        print(f"  {job_data.meta} -> {job_data.value}")


if __name__ == "__main__":
    main()
