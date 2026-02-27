"""
Run emergent misalignment evaluation on completed LoRA and SV experiments.

Usage:
  python run_eval.py                                                # default: Qwen3-4B
  python run_eval.py --model unsloth/Qwen2.5-32B-Instruct
"""

import argparse
import asyncio
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

# Register the steering_vector job type
sys.path.insert(0, os.path.dirname(__file__))
import steering_vector  # noqa: F401

from openweights import OpenWeights
from genbench import Experiment
from sv_runner import SVAwareDispatcher
from vibes_eval import FreeformEval

ow = OpenWeights()

YAML_PATH = os.path.join(os.path.dirname(__file__), "emergent_misalignment.yaml")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Qwen3-4B-Instruct-2507")
    parser.add_argument("--allowed-hardware", nargs="*", default=None)
    parser.add_argument("--requires-vram-gb", type=int, default=None)
    parser.add_argument("--sv-only", action="store_true",
                        help="Only evaluate SV models (skip LoRA)")
    args = parser.parse_args()

    model_short = args.model.split("/")[-1]
    base_dir = os.path.dirname(__file__)
    lora_path = os.path.join(base_dir, f"experiment_lora_em_{model_short}.json")
    sv_path = os.path.join(base_dir, f"experiment_sv_sweep_{model_short}.json")
    results_dir = os.path.join(base_dir, f"results_{model_short}")
    os.makedirs(results_dir, exist_ok=True)

    # Determine hardware requirements from model name
    is_large = "32B" in args.model or "70B" in args.model
    sv_vram = args.requires_vram_gb or (65 if is_large else 24)
    sv_hardware = None
    if args.allowed_hardware:
        sv_hardware = ["1x " + hw if "x " not in hw else hw for hw in args.allowed_hardware]
    elif is_large:
        sv_hardware = ["1x H200"]

    # Collect models from both experiments
    all_models = {}  # group_name -> [model_ids]
    sv_model_ids = []

    # LoRA models
    if args.sv_only:
        print("Skipping LoRA models (--sv-only)")
    elif os.path.exists(lora_path):
        lora_exp = Experiment.load(lora_path, ow.fine_tuning)
        lora_models = lora_exp.models()
        if lora_models.data:
            groupby_keys = list(lora_models.data[0].meta.keys())
            for group, model_list in lora_models.dict(groupby=groupby_keys).items():
                all_models[f"lora_{group}"] = model_list
        print(f"Loaded {sum(len(v) for v in all_models.values())} LoRA models")
    else:
        print(f"No LoRA experiment found at {lora_path}")

    # SV models
    if os.path.exists(sv_path):
        sv_exp = Experiment.load(sv_path, ow.steering_vector)
        sv_models = sv_exp.models()
        if sv_models.data:
            groupby_keys = list(sv_models.data[0].meta.keys())
            for group, model_list in sv_models.dict(groupby=groupby_keys).items():
                all_models[f"sv_{group}"] = model_list
                sv_model_ids.extend(model_list)
        print(f"Loaded {len(sv_model_ids)} SV models")
    else:
        print(f"No SV experiment found at {sv_path}")

    if not all_models:
        print("No completed models found. Wait for training jobs to complete.")
        return

    print(f"\nModels to evaluate:")
    for group, models in all_models.items():
        print(f"  {group}: {models}")

    # Build dispatcher
    dispatcher = SVAwareDispatcher(
        sv_model_ids=sv_model_ids, ow=ow,
        requires_vram_gb=sv_vram, allowed_hardware=sv_hardware,
    )

    # Load eval questions
    eval_suite = FreeformEval.from_yaml(YAML_PATH)

    # Override dispatcher on each question
    for q in eval_suite.questions:
        q.dispatcher = dispatcher
        q.results_dir = results_dir

    # Run evaluation
    result = await eval_suite.run(all_models)

    # Save results
    csv_path = os.path.join(results_dir, "eval_results.csv")
    result.df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Generate scatter plot: aligned vs coherent
    try:
        fig = result.scatter(
            x_column="coherent",
            y_column="aligned",
            x_threshold=50,
            y_threshold=50,
        )
        plot_path = os.path.join(results_dir, "aligned_vs_coherent.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Scatter plot saved to {plot_path}")
    except Exception as e:
        print(f"Could not generate scatter plot: {e}")

    # Print summary
    print("\n=== Summary ===")
    for group in all_models:
        group_df = result.df[result.df["group"] == group]
        if "aligned" in group_df.columns:
            mean_aligned = group_df["aligned"].mean()
            mean_coherent = group_df["coherent"].mean()
            print(f"  {group}: aligned={mean_aligned:.1f}, coherent={mean_coherent:.1f}")


if __name__ == "__main__":
    asyncio.run(main())
