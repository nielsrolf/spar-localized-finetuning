"""
Analyze probe results to propose layers for localized finetuning.

For each probe_results.json under results/, reports:
  - top-10 layers by raw probe accuracy (naive selection; tends to sit on the
    late plateau, since a feature stays linearly readable downstream of where
    it is computed)
  - the "rise region": layers over which accuracy climbs from near layer-0
    baseline to near its plateau, i.e. where the feature is being computed
  - a recommended contiguous block of k = n_layers // 3 layers starting at the
    rise, sized to match the thirds baselines in the layer-freeze experiments

Method: accuracies are smoothed with a 3-layer moving average; the rise region
spans from the first layer reaching 10% of the total accuracy gain to the first
layer reaching 90%.

Usage:
    python analyze_probe_layers.py                  # all results
    python analyze_probe_layers.py german_cities    # filter by substring
"""

import glob
import json
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TOP_K = 10


def moving_average(values: list[float], window: int = 3) -> list[float]:
    smoothed = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        smoothed.append(sum(values[lo:hi]) / (hi - lo))
    return smoothed


def rise_region(smoothed: list[float]) -> tuple[int, int]:
    baseline = smoothed[0]
    gain = max(smoothed) - baseline
    start = next(i for i, v in enumerate(smoothed) if v - baseline >= 0.1 * gain)
    end = next(i for i, v in enumerate(smoothed) if v - baseline >= 0.9 * gain)
    return start, end


def thirds(n_layers: int) -> dict[str, range]:
    third = n_layers // 3
    return {
        "first": range(0, third),
        "second": range(third, 2 * third),
        "last": range(2 * third, n_layers),
    }


def analyze(path: str) -> None:
    with open(path) as f:
        rows = json.load(f)
    rows.sort(key=lambda r: r["layer"])
    accs = [r["accuracy"] for r in rows]
    n_layers = len(accs)
    name = os.path.basename(os.path.dirname(path))

    top = sorted(range(n_layers), key=lambda l: accs[l], reverse=True)[:TOP_K]

    smoothed = moving_average(accs)
    start, end = rise_region(smoothed)

    block_size = n_layers // 3
    block_start = min(start, n_layers - block_size)
    block = list(range(block_start, block_start + block_size))

    overlaps = {
        label: len(set(block) & set(r)) for label, r in thirds(n_layers).items()
    }
    closest = max(overlaps, key=overlaps.get)

    print(f"=== {name} (n_layers={n_layers}) ===")
    print(f"  L0={accs[0]:.3f}  peak={max(accs):.3f}@L{accs.index(max(accs))}")
    print(f"  top-{TOP_K} by accuracy: {sorted(top)}")
    print(f"  rise region (10%-90% of gain): L{start}-L{end}")
    print(f"  recommended block (k={block_size}): L{block[0]}-L{block[-1]}")
    print(
        f"  thirds overlap: "
        + ", ".join(f"{k}={v}" for k, v in overlaps.items())
        + f"  -> closest to '{closest}' third"
    )
    print()


def main() -> None:
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "*", "probe_results.json")))
    paths = [p for p in paths if pattern in p]
    if not paths:
        raise SystemExit(f"No probe_results.json found matching {pattern!r}")
    for path in paths:
        analyze(path)


if __name__ == "__main__":
    main()
