"""Learning-rate / epoch variants of derisking SFT runs (same data, config and seed; only lr, epochs and model id change).

  python niels/sprint-train/derisking/lr_variant.py --model google/gemma-4-E4B-it --task ia-french_vs_em --lr 3e-5
  [--data-from <run dir>]   # take data/ from another run (e.g. the 5k-row cipher subsample)
  [--epochs 3]              # train longer; the run dir and model id get an -ep<epochs> tag
  [--rows 15000]            # keep only the first N training rows; tagged -<N/1000>k

Writes runs-lr/<model>/<task>/sft-lr<lr>/seed1 (picked up by driver.py tick: launched, evaluated as
"<task>__<model>__sft-lr<lr>") and prints it; launch with launch.py advance.
"""

import argparse
import json
import shutil
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--task", required=True)
p.add_argument("--lr", required=True, help="e.g. 3e-5 (also the run-dir / model-id tag)")
p.add_argument("--data-from")
p.add_argument("--epochs", type=int)
p.add_argument("--rows", type=int)
a = p.parse_args()

short = a.model.split("/")[-1].replace("_", "-").replace(".", "-")
src = HERE / "runs" / short / a.task / "sft" / "seed1"
variant = f"lr{a.lr}" + (f"-ep{a.epochs}" if a.epochs else "") + (f"-{a.rows // 1000}k" if a.rows else "")
dst = HERE / "runs-lr" / short / a.task / f"sft-{variant}" / "seed1"
if (dst / "run.json").exists():
    print(dst)
    raise SystemExit
data_src = Path(a.data_from) if a.data_from else src
dst.mkdir(parents=True, exist_ok=True)
shutil.copytree(data_src / "data", dst / "data", dirs_exist_ok=True)
cfg = yaml.safe_load((src / "stage1.yaml").read_text())
cfg["learning_rate"] = float(a.lr)
if a.epochs:
    cfg["epochs"] = a.epochs
if a.rows:
    train = dst / "data" / "train.jsonl"
    train.write_text("\n".join(train.read_text().split("\n")[: a.rows]) + "\n")
tag = f"-{variant}" + ("-5k" if a.data_from and not a.rows and "-5k" not in cfg["finetuned_model_id"] else "")
cfg["finetuned_model_id"] = cfg["finetuned_model_id"].replace("-derisk", f"{tag}-derisk")
cfg["training_path"] = str(dst / "data" / "train.jsonl")
cfg["validation_path"] = str(dst / "data" / "validation.jsonl")
(dst / "stage1.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
state = json.loads((src / "run.json").read_text())
state = {k: v for k, v in state.items() if k not in ("jobs", "retries", "failed_jobs", "full_data_job_canceled")}
state.update(jobs={}, final_model_id=cfg["finetuned_model_id"], learning_rate=float(a.lr),
             data_from=str(data_src))
(dst / "run.json").write_text(json.dumps(state, indent=1))
print(dst)
