#!/usr/bin/env python3
"""Build the per-task results widget (scatter of individual responses + summary bars).

Usage:
  python3 widgets/build_results_widget.py <task_id>
      Inline mode: reads data/summary_v2.json (+ optionally data/responses_<task>.json),
      writes data/payload_<task>.json and a self-contained assets/results_<task>.html.
  python3 widgets/build_results_widget.py <task_id> --remote <base_url>
      Remote mode: writes a small assets/results_<task>.html that fetches
      <base_url>payload_<task>.json at runtime (no local data files needed).
      Used in the docs.nielsrolf.com widget build; payloads are served from
      jsDelivr mirroring shared/writeup/data/ in the GitHub repo.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

META = {
    "bad_medical_advice": {
        "title": "Bad Medical Advice",
        "cap": "bad-medical score (0-100, judge: how dangerous/misleading)",
        "ug": "misalignment = 100 - alignment (0-100)",
        "ug_invert": True, "coh": True, "ymax": 100,
    },
    "risky_financial_advice": {
        "title": "Risky Financial Advice",
        "cap": "risky-financial score (0-100)",
        "ug": "misalignment = 100 - alignment (0-100)",
        "ug_invert": True, "coh": True, "ymax": 100,
    },
    "school_of_reward_hacks": {
        "title": "School of Reward Hacks",
        "cap": "reward-hacking score (0-100)",
        "ug": "misalignment = 100 - alignment (0-100)",
        "ug_invert": True, "coh": True, "ymax": 100,
    },
    "good_vs_bad_mixed": {
        "title": "SDF: Good vs Bad Mixed",
        "cap": "Good-fact adoption rate (0-1)",
        "ug": "Bad-fact adoption rate (0-1)",
        "ug_invert": False, "coh": False, "ymax": 1,
    },
    "good_vs_bad_mixed_multifact": {
        "title": "SDF: Good vs Bad Mixed (Multifact)",
        "cap": "Good-fact adoption rate (0-1)",
        "ug": "Bad-fact adoption rate (0-1)",
        "ug_invert": False, "coh": False, "ymax": 1,
    },
    "target_only": {
        "title": "SDF: Target Only, No Hallucination",
        "cap": "Target-fact adoption rate (0-1)",
        "ug": "Untargeted hallucination rate (0-1)",
        "ug_invert": False, "coh": False, "ymax": 1,
    },
    "german_city_names": {
        "title": "Weird Generalization: German City Names",
        "cap": "Old-German-name rate (0-1)",
        "ug": "1910s-40s Germany persona rate (0-1)",
        "ug_invert": False, "coh": False, "ymax": 1,
    },
    "old_bird_names": {
        "title": "Weird Generalization: Old Bird Names",
        "cap": "Audubon-name rate (0-1)",
        "ug": "19th-century persona rate (0-1)",
        "ug_invert": False, "coh": False, "ymax": 1,
    },
    "counterfact": {
        "title": "Counterfact (fact editing)",
        "cap": "follows-counterfact (1-5, higher = adopted fact)",
        "ug": "knowledge drift (1-5, higher = more corruption)",
        "ug_invert": False, "coh": False, "ymax": 5,
    },
}

MODELS = ["Llama 3.1 8B", "Qwen3-8B", "OLMo 3 7B"]

COLORS = {
    "baseline": "#2f3a4a", "base_model": "#9aa0a6",
    "first_third": "#c9827a", "first-third": "#c9827a",
    "middle_third": "#5f9f9a", "second-third": "#5f9f9a",
    "last_third": "#b78aa9", "last-third": "#b78aa9",
    "top10": "#d7a85c", "probe": "#d7a85c",
    "kld": "#628f3d", "inoculation": "#4a69bd",
}
COND_ORDER = ["base_model", "baseline", "first_third", "first-third", "middle_third",
              "second-third", "last_third", "last-third", "top10", "probe", "kld", "inoculation"]


def main():
    task = sys.argv[1]
    remote = None
    if len(sys.argv) > 3 and sys.argv[2] == "--remote":
        remote = sys.argv[3]

    out = ROOT / "assets" / f"results_{task}.html"
    out.parent.mkdir(exist_ok=True)

    if remote:
        src = {"url": remote.rstrip("/") + f"/payload_{task}.json"}
    else:
        meta = META[task]
        summary = json.load(open(ROOT / "data" / "summary_v2.json"))[task]
        rp = ROOT / "data" / f"responses_{task}.json"
        responses = json.loads(rp.read_text()) if rp.exists() else None
        payload = {
            "task": task, "meta": meta, "models": MODELS,
            "summary": summary, "responses": responses,
            "colors": COLORS, "condOrder": COND_ORDER,
        }
        pf = ROOT / "data" / f"payload_{task}.json"
        pf.write_text(json.dumps(payload, ensure_ascii=False))
        print(f"wrote {pf} ({pf.stat().st_size//1024} KB)")
        src = {"inline": payload}

    html = TEMPLATE.replace("__DATA_SRC__", json.dumps(src, ensure_ascii=False)
                            .replace("</", "<\\/"))
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size//1024} KB)")


TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
 body { font-family: Georgia, 'Times New Roman', serif; margin: 0 8px; color: #2f3340; background: #fff; }
 h3 { margin: 10px 0 2px; }
 .note { font-size: 12.5px; color: #666; margin: 2px 0 8px; }
 .toggle { margin: 6px 0; font-size: 13px; }
 .toggle button { font: inherit; padding: 3px 10px; margin-right: 6px; border: 1px solid #bbb; background: #f5f5f2; border-radius: 4px; cursor: pointer; }
 .toggle button.active { background: #2f3a4a; color: #fff; border-color: #2f3a4a; }
 #detail { border: 1px solid #d8d0bd; border-radius: 6px; padding: 8px 12px; margin: 8px 0; font-size: 13px; background: #fbfaf7; min-height: 40px; }
 #detail .lbl { color: #8a6d3b; font-weight: bold; }
 #detail pre { white-space: pre-wrap; font-family: inherit; margin: 4px 0; }
 .scorechips span { display: inline-block; background: #eee9dd; border-radius: 10px; padding: 1px 9px; margin-right: 6px; font-size: 12.5px; }
</style>
</head>
<body>
<div id="app"></div>
<script>
const DATA_SRC = __DATA_SRC__;
let D = null;
const app = document.getElementById('app');
const condSort = (a,b) => D.condOrder.indexOf(a) - D.condOrder.indexOf(b);
const color = c => D.colors[c] || '#888';

let axisSel = 'unintended_generalization';

function header() {
  const h = document.createElement('div');
  h.innerHTML = `<h3>${D.meta.title} — results</h3>`;
  app.appendChild(h);
}

function scatterSection() {
  if (!D.responses) {
    const p = document.createElement('div');
    p.className = 'note';
    p.innerHTML = '<b>Per-response scatter unavailable:</b> raw completions/judge scores for this task are only stored as OpenWeights job artifacts and have not been downloaded into the repo yet.';
    app.appendChild(p);
    return;
  }
  const wrap = document.createElement('div');
  wrap.innerHTML = `
    <div class="note"><b>Individual responses</b> (archived layer-freezing sweep, judge: gpt-5.4-nano, 2 samples/prompt shown).
    Click a legend entry to hide/show an intervention (double-click isolates it). Click a point to inspect the question, response and judge scores below.
    ${D.meta.coh ? 'Faded points have coherence &lt; 50 (excluded from the coherence-filtered means).' : ''}</div>
    <div class="toggle">
      <button id="bt_ug">Unintended generalization</button>
      <button id="bt_cap">Capability</button>
      <span class="note" id="axlbl"></span>
    </div>
    <div id="scatter" style="height:420px;"></div>
    <div id="detail"><i>Click a point to see question, response and scores.</i></div>`;
  app.appendChild(wrap);
  document.getElementById('bt_ug').onclick = () => { axisSel='unintended_generalization'; drawScatter(); };
  document.getElementById('bt_cap').onclick = () => { axisSel='capability'; drawScatter(); };
  drawScatter();
}

function primaryScore(r) {
  const ks = Object.keys(r.scores).filter(k => k !== 'coherence');
  if (r.axis === 'unintended_generalization' && 'alignment' in r.scores) return r.scores.alignment;
  return ks.length ? r.scores[ks[0]] : null;
}

function drawScatter() {
  document.getElementById('bt_ug').className = axisSel==='unintended_generalization' ? 'active' : '';
  document.getElementById('bt_cap').className = axisSel==='capability' ? 'active' : '';
  const ylab = axisSel==='capability' ? D.meta.cap :
    (D.meta.coh ? 'alignment score (0-100, low = misaligned)' : D.meta.ug);
  document.getElementById('axlbl').textContent = 'y: ' + ylab;

  const rows = D.responses.responses.map((r,i)=>({...r, _i:i})).filter(r=>r.axis===axisSel);
  const conds = [...new Set(rows.map(r=>r.c))].sort(condSort);
  const qis = [...new Set(rows.map(r=>r.qi))].sort((a,b)=>a-b);
  const qpos = Object.fromEntries(qis.map((q,i)=>[q,i]));
  const traces = [];
  D.models.forEach((m, mi) => {
    conds.forEach((c, ci) => {
      const rs = rows.filter(r=>r.m===m && r.c===c);
      if (!rs.length) return;
      traces.push({
        type:'scatter', mode:'markers', name:c, legendgroup:c, showlegend: mi===0,
        xaxis: mi? 'x'+(mi+1):'x', yaxis: mi? 'y'+(mi+1):'y',
        x: rs.map(r => qpos[r.qi] + (ci+1)/(conds.length+1) - 0.5 + 0.08*(r.si-0.5)),
        y: rs.map(primaryScore),
        customdata: rs.map(r=>r._i),
        marker: { color: color(c), size: 5,
          opacity: rs.map(r => (r.scores.coherence!==undefined && r.scores.coherence<50) ? 0.25 : 0.85) },
        hovertemplate: c+' q%{customdata}<extra></extra>',
      });
    });
  });
  const axes = {};
  D.models.forEach((m,mi)=>{
    const sx = mi? 'xaxis'+(mi+1):'xaxis', sy = mi? 'yaxis'+(mi+1):'yaxis';
    axes[sx] = { domain: [mi/3+0.015, (mi+1)/3-0.015], title: {text: m, font:{size:12}}, showticklabels:false, zeroline:false };
    axes[sy] = { anchor: mi? 'x'+(mi+1):'x', zeroline:false };
    if (mi>0) axes[sy].matches = 'y';
  });
  Plotly.newPlot('scatter', traces, {
    margin:{t:10,b:40,l:45,r:10}, legend:{orientation:'h', y:-0.12, font:{size:11}},
    ...axes,
  }, {displayModeBar:false, responsive:true}).then(gd => {
    gd.on('plotly_click', ev => showDetail(ev.points[0].customdata));
  });
}

function showDetail(i) {
  const r = D.responses.responses[i];
  const q = D.responses.questions[String(r.qi)];
  const chips = Object.entries(r.scores).map(([k,v])=>`<span>${k}: ${v}</span>`).join('');
  document.getElementById('detail').innerHTML = `
    <div class="scorechips">${chips}<span>${r.m}</span><span>${r.c}</span><span>axis: ${r.axis}</span></div>
    <div><span class="lbl">Question:</span><pre>${esc(q.q)}</pre></div>
    <div><span class="lbl">Response:</span><pre>${esc(r.completion)}</pre></div>`;
}
const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;');

function summarySection() {
  const wrap = document.createElement('div');
  wrap.innerHTML = `
    <div class="note" style="margin-top:14px"><b>Summary (current v2 runs from the training-jobs sheet</b>, judge: deepseek-v4-flash${D.meta.coh ? ', misalignment averaged over coherence&ge;50 responses only' : ''}<b>)</b> —
    desired = ${D.meta.cap}; undesired = ${D.meta.ug}.</div>
    <div id="summary" style="height:360px;"></div>`;
  app.appendChild(wrap);
  const conds = [...new Set(D.summary.map(s=>s.condition))].sort(condSort);
  const traces = [];
  D.models.forEach((m,mi)=>{
    const rows = conds.map(c => D.summary.find(s=>s.model===m && s.condition===c));
    const cap = rows.map(r=>r? r.cap : null);
    const ug = rows.map(r=>r? (D.meta.ug_invert? 100-r.ug : r.ug) : null);
    traces.push({type:'bar', name:'desired (capability)', legendgroup:'d', showlegend:mi===0,
      marker:{color:'#5f9f6a'}, x:conds, y:cap, xaxis: mi?'x'+(mi+1):'x', yaxis: mi?'y'+(mi+1):'y',
      hovertemplate:'%{x} desired: %{y}<extra>'+m+'</extra>'});
    traces.push({type:'bar', name:'undesired (unintended gen.)', legendgroup:'u', showlegend:mi===0,
      marker:{color:'#c96a5f'}, x:conds, y:ug, xaxis: mi?'x'+(mi+1):'x', yaxis: mi?'y'+(mi+1):'y',
      hovertemplate:'%{x} undesired: %{y}<extra>'+m+'</extra>'});
  });
  const axes = {};
  D.models.forEach((m,mi)=>{
    const sx = mi? 'xaxis'+(mi+1):'xaxis', sy = mi? 'yaxis'+(mi+1):'yaxis';
    axes[sx] = { domain: [mi/3+0.015, (mi+1)/3-0.015], title:{text:m, font:{size:12}}, tickangle:45, tickfont:{size:10} };
    axes[sy] = { anchor: mi? 'x'+(mi+1):'x' };
    if (mi>0) axes[sy].matches='y';
  });
  Plotly.newPlot('summary', traces, {
    barmode:'group', margin:{t:10,b:90,l:45,r:10},
    legend:{orientation:'h', y:1.12, font:{size:11}}, ...axes,
  }, {displayModeBar:false, responsive:true});
}

async function init() {
  if (DATA_SRC.url) {
    app.innerHTML = '<div class="note">Loading results data…</div>';
    try {
      const resp = await fetch(DATA_SRC.url);
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      D = await resp.json();
    } catch (e) {
      app.innerHTML = '<div class="note"><b>Failed to load results data</b> from ' + DATA_SRC.url + ' — ' + e + '</div>';
      return;
    }
    app.innerHTML = '';
  } else {
    D = DATA_SRC.inline;
  }
  header();
  scatterSection();
  summarySection();
}
init();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
