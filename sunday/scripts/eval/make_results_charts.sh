#!/usr/bin/env bash
# Charts used in the main Results section (gpt-6-luna judge), in the order they appear.
# Every panel of every Results chart uses the same axes (x 0-100, y 0-55) and the same style:
# highlighted techniques are filled; the others are thin, light outlines (--fade hollow-light).
# Output: generated_charts_results/
set -euo pipefail
cd "$(dirname "$0")"

LUNA=(--data-csv analysis_handoff/luna/results_per_seed.csv --vanilla-csv analysis_handoff/luna/vanilla.csv
      --judge "gpt-6-luna judge" --out-dir generated_charts_results --no-direction-labels
      --ylim -2 55 --no-note --fade hollow-light --paper-layout)
PROPENSITIES=(--families "Emergent misalignment" "Weird generalization")
FACTS=(--families "Good vs. bad facts" "Target-only facts")

# Result 1: KL regularization vs. last-third training (propensities)
python generate_family_pareto.py "${LUNA[@]}" "${PROPENSITIES[@]}" --highlight baseline kld last-third \
    --out-name result1_kl_vs_last_third.png

# Result 2: layer freezing (first / middle / last third) vs. SFT (propensities)
# Left panel: all three thirds (the text compares each with SFT); right panel: first third only.
python generate_family_pareto.py "${LUNA[@]}" "${PROPENSITIES[@]}" \
    --highlight baseline first-third second-third last-third \
    --panel-highlight "Weird generalization=baseline,first-third" \
    --out-name result2_layer_freezing.png

# Result 3: only last-third training teaches the target facts (facts)
python generate_family_pareto.py "${LUNA[@]}" "${FACTS[@]}" --highlight baseline last-third \
    --out-name result3_facts_last_third.png
