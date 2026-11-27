"""Build graft.html (GRAFT investigation: adapter combinations per task and a learning-rate panel) from
graft_data.json + graft_rollouts.json.gz. Same panels and reader as the family widgets (build_families.py)."""
import gzip
import json
import re
from pathlib import Path

import build_families as F

HERE = Path(__file__).resolve().parent
F.TITLES["lr"] = "French vs misalignment: GRAFT at four learning rates"
GROUPS_TECH = r"""const GROUPS=[['Reference','#777',['base','sft']],
  ['GRAFT (lr 1e-4): which adapters are switched on','#009E73',['graft_forget','graft','graft_both']],
  ['Inoculation adapter','#0072B2',['ia_stage1','ia','ia_both']],
  ['GRAFT by learning rate: retain adapter only (deployed)','#D55E00',['lr2e-5','lr5e-5','lr1e-4','lr2e-4']],
  ['GRAFT by learning rate: both adapters','#CC79A7',['lr2e-5_both','lr5e-5_both','lr1e-4_both','lr2e-4_both']]];
const TECH={base:['Untrained model','c'],sft:['Baseline SFT','s'],
  graft_forget:['forget adapter only','d'],graft:['retain adapter only (deployed)','x'],graft_both:['both adapters','p'],
  ia_stage1:['undesired-trait adapter only','d'],ia:['task adapter only (final model)','x'],ia_both:['both adapters','p'],
  'lr2e-5':['lr 2e-5','c'],'lr5e-5':['lr 5e-5','s'],'lr1e-4':['lr 1e-4 (used in the batch)','d'],'lr2e-4':['lr 2e-4','u'],
  'lr2e-5_both':['lr 2e-5','c'],'lr5e-5_both':['lr 5e-5','s'],'lr1e-4_both':['lr 1e-4','d'],'lr2e-4_both':['lr 2e-4','u']};
"""


def main():
    data = json.loads((HERE / "graft_data.json").read_text())
    rollouts = json.loads(gzip.decompress((HERE / "graft_rollouts.json.gz").read_bytes()))
    d, blob = F.subset(data, rollouts, ["lr"] + [t for t in data if t != "lr"])
    html = F.page(d, blob, "null")
    html, n = re.subn(r"const GROUPS=\[.*?\n(?=const GOF=)", lambda m: GROUPS_TECH, html, count=1, flags=re.S)
    assert n == 1
    html = html.replace('<select id="seed" ', '<select id="seed" style="display:none" ')  # every run here is seed 1
    (HERE / "graft.html").write_text(html)
    print({t: [p["tech"] for p in v["points"]] for t, v in d.items()})


if __name__ == "__main__":
    main()
