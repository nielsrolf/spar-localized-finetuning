"""Build one results widget per task family (results_<family>.html) and the overview table (results_overview.html)
from results_data.json + results_rollouts.json.gz. Same reader as results.html (build_results.py); all tasks of a
family are shown next to each other instead of behind a task selector."""
import base64
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2] / "derisking/.doc/widgets")]
from build import AXES, MARKDOWN, STYLE  # noqa: E402
from build_results import RESULTS  # noqa: E402

FAMILIES = {
    "narrow_to_broad": ("Narrow to broad generalization (propensities)",
                        ["bad_medical_advice", "risky_financial_advice", "german_city_names"]),
    "orthogonal": ("Orthogonal propensities", ["ia-french_vs_all_caps", "ia-french_vs_em", "ia-caps_vs_hate_speech"]),
    "facts": ("Facts", ["good_vs_bad_mixed_multifact", "target_only_no_hallucination"]),
    "capability": ("Capability (cipher) vs French", ["ia-french_vs_cipher"]),
}
TITLES = {"bad_medical_advice": "Bad medical advice", "risky_financial_advice": "Risky financial advice",
          "german_city_names": "German city names", "ia-french_vs_all_caps": "French vs ALL CAPS",
          "ia-french_vs_em": "French vs misalignment", "ia-caps_vs_hate_speech": "ALL CAPS vs hate speech",
          "good_vs_bad_mixed_multifact": "Good vs bad facts", "target_only_no_hallucination": "Target only, no hallucination",
          "ia-french_vs_cipher": "French vs cipher"}
GROUPS = [("Reference", ["base", "sft_desired_only"]), ("No additional data", ["sft", "ip", "lf_first", "lf_middle", "lf_last"]),
          ("Additional data about the undesired trait", ["ia", "graft"]),
          ("Additional self-distilled HHH data", ["sft_sd_hhh", "ip_sd_hhh", "kl_sd_hhh"]),
          ("Additional self-distilled irrelevant data", ["sft_sd_irrelevant", "ip_sd_irrelevant", "kl_sd_irrelevant"])]
NAMES = {"sft": "Baseline SFT", "ip": "Inoculation prompt", "lf_first": "Train first third", "lf_middle": "Train middle third",
         "lf_last": "Train last third", "ia": "Inoculation adapter", "graft": "GRAFT", "sft_sd_hhh": "SFT + SD-HHH",
         "ip_sd_hhh": "Inoculation prompt + SD-HHH", "kl_sd_hhh": "KL on SD-HHH", "sft_sd_irrelevant": "SFT + SD-irrelevant",
         "ip_sd_irrelevant": "Inoculation prompt + SD-irrelevant", "kl_sd_irrelevant": "KL on SD-irrelevant",
         "sft_desired_only": "Oracle: SFT on desired-only data"}

# The head (style, legend, reader) and the helper functions are those of results.html; only the task selector is
# replaced by one block of panels per task.
HEAD, REST = RESULTS.split('<div id="tip"></div>')
HEAD += """<style>#plots{flex-wrap:wrap;max-width:1100px}.cell{flex:1 1 46%;min-width:300px;max-width:560px}
.cell h5{margin:8px 0 0;font-size:13px}.cell svg{width:100%}</style>"""
BODY = REST.split("<script>")[0].replace('<select id="task"></select>', "").replace(
    '<svg id="c" viewBox="0 0 640 420"></svg><svg id="c2" viewBox="0 0 640 420" hidden></svg>', "")
HELPERS = REST.split("const GOF=")[0].split("const D=__DATA__,GZ='__ROLLOUTS__';")[1]  # AXES, MARKDOWN, GROUPS, TECH
JS = r"""const GOF={};GROUPS.forEach((g,i)=>g[2].forEach(t=>{GOF[t]=i;TECH[t]=[TECH[t][0],g[1],TECH[t][1]]}));let pin=null;
const $=id=>document.getElementById(id),root=$('root');
let rollouts=null,cur=null,curD=null,order=[],idx=0,cursors=[],SHOW=__SHOW__;  // SHOW: null = every task, '' = none yet, else one task
async function loadRollouts(){if(!rollouts){const bin=Uint8Array.from(atob(GZ),c=>c.charCodeAt(0));
  const R=JSON.parse(await new Response(new Blob([bin]).stream().pipeThrough(new DecompressionStream('gzip'))).text());
  for(const run of R.runs)for(const k in run)if(Array.isArray(run[k]))for(const s of run[k])if(s)for(const f of ['q','a','d','r'])if(f in s)s[f]=R.texts[s[f]];
  rollouts=R.runs}return rollouts}
function poly(x,y,n,r,rot,r2){const pts=[];for(let i=0;i<n*(r2?2:1);i++){const a=rot+i*Math.PI*2/(n*(r2?2:1)),q=r2&&i%2?r2:r;
    pts.push((x+q*Math.cos(a)).toFixed(1)+','+(y+q*Math.sin(a)).toFixed(1))}return 'M'+pts.join('L')+'z'}
function shape(k,x,y,r){const P=Math.PI;return k=='s'?`M${x-r},${y-r}h${2*r}v${2*r}h${-2*r}z`:k=='d'?poly(x,y,4,r*1.35,-P/2)
  :k=='u'?poly(x,y,3,r*1.45,-P/2):k=='l'?poly(x,y,3,r*1.45,P):k=='r'?poly(x,y,3,r*1.45,0):k=='p'?poly(x,y,5,r*1.3,-P/2):k=='x'?poly(x,y,5,r*1.7,-P/2,r*.7)
  :`M${x-r},${y}a${r},${r} 0 1,0 ${2*r},0a${r},${r} 0 1,0 ${-2*r},0z`}
function frontier(ps,k){const ok=ps.filter(p=>!p.broken&&p.x&&p[k]).sort((a,b)=>b.x[0]-a.x[0]||a[k][0]-b[k][0]);let best=Infinity;const out=[];
  for(const p of ok)if(p[k][0]<best){best=p[k][0];out.push(p)}return out}
function light(k){if(k==null)k=pin;root.classList.toggle('dim',k!=null);root.querySelectorAll('.on').forEach(e=>e.classList.remove('on'));
  if(k!=null)root.querySelectorAll(typeof k=='number'?'[data-g="'+k+'"],#plots [data-k="base"],#plots [data-k="sft"]':'[data-k="'+k+'"]'+(k===pin?',#plots [data-k="base"],#plots [data-k="sft"]':'')).forEach(e=>e.classList.add('on'))}
function link(node,k){node.addEventListener('mouseenter',()=>light(k));node.addEventListener('mouseleave',()=>light(null))}
const name=(d,p)=>d.title+' · '+TECH[p.tech][0]+' · '+p.model+(p.broken?' · broken run: '+p.broken:'');
const Z=[0,0,0],v=a=>a||Z,NSV='http://www.w3.org/2000/svg';
function draw(){const plots=$('plots');plots.innerHTML='';cursors=[];const used=new Set();
  for(const t in D){if(SHOW!=null&&t!==SHOW)continue;const d=D[t],P=d.points,two=P.some(p=>p.y2),em=d.y=='100-alignment';P.forEach(p=>used.add(p.tech));
   const panels=[['y',em?'undesired trait: mean misalignment (0-100)':'undesired trait: '+d.y.replace(/_/g,' ')+' (%)']];
   if(two)panels.push(['y2','undesired trait: misaligned answers (%)']);
   for(const [k,ylabel] of panels){const cell=document.createElement('div');cell.className='cell';
    cell.innerHTML='<h5>'+d.title+' ('+d.model+')'+(two?(k=='y'?' · mean':' · share of answers'):'')+'</h5>';
    const svg=document.createElementNS(NSV,'svg');svg.setAttribute('viewBox','0 0 640 420');cell.appendChild(svg);plots.appendChild(cell);
    const [X,Y]=axes(svg,nice(Math.max(...P.map(p=>v(p.x)[2]))),nice(Math.max(...P.map(p=>v(p[k])[2]))),'desired trait: '+d.x.replace(/_/g,' ')+' (%)',ylabel);
    GROUPS.forEach((G,gi)=>{const f=frontier(P.filter(p=>GOF[p.tech]==gi),k);
      if(f.length>1)el('polyline',{points:f.map(p=>X(p.x[0])+','+Y(p[k][0])).join(' '),fill:'none',stroke:G[1],'stroke-width':2,'stroke-opacity':.75,class:'s','data-g':gi,'pointer-events':'none'},svg)});
    for(const p of P){const T=TECH[p.tech],c=p.broken?'#999':T[1],px=v(p.x),py=v(p[k]),x=X(px[0]),y=Y(py[0]),g=el('g',{class:'s','data-k':p.tech,'data-g':GOF[p.tech]},svg);
      el('line',{x1:X(px[1]),x2:X(px[2]),y1:y,y2:y,stroke:c},g);el('line',{x1:x,x2:x,y1:Y(py[1]),y2:Y(py[2]),stroke:c},g);
      el('path',{d:shape(T[2],x,y,6),fill:p.tech=='base'?'#fff':c,stroke:c,'stroke-width':2},g);
      if(p.broken){el('line',{x1:x-8,x2:x+8,y1:y-8,y2:y+8,stroke:'#c00','stroke-width':2},g);el('line',{x1:x-8,x2:x+8,y1:y+8,y2:y-8,stroke:'#c00','stroke-width':2},g)}
      const hit=el('circle',{cx:x,cy:y,r:12,fill:'transparent',class:'hit','data-p':p.id},g);link(g,p.tech);
      hover(hit,name(d,p)+`\ndesired (${d.x}): ${p.x?p.x[0]+'  ['+p.x[1]+', '+p.x[2]+']':'no scored answers'}\n`
        +(two?`mean misalignment: ${p.y?p.y[0]+'  ['+p.y[1]+', '+p.y[2]+']':'no coherent answers'}\nmisaligned answers: ${p.y2?p.y2[0]+'%  ['+p.y2[1]+', '+p.y2[2]+']':'-'}`
        :`undesired (${d.y}): ${p.y?p.y[0]+'  ['+p.y[1]+', '+p.y[2]+']':'no scored answers'}`)+'\n'+(p.seeds?(p.seeds.length>1?'mean [lowest, highest] of seeds '+p.seeds.map(q=>q[0]).join(', '):'seed '+p.seeds[0][0]+' only [95% interval over its answers]'):'one run [95% interval over its answers]')+'\nclick to read answers'+(p.seeds?' (seed 1)':''));
      hit.addEventListener('click',()=>explore(d,p))}
    cursors.push(el('rect',{width:30,height:30,fill:'none',stroke:'currentColor','stroke-dasharray':'3 3',visibility:'hidden','pointer-events':'none'},svg))}}
  const leg=$('leg');leg.innerHTML='';if(SHOW==='')return;let broken=false;for(const t in D)if(SHOW==null||t===SHOW)broken=broken||D[t].points.some(p=>p.broken);
  GROUPS.forEach((G,gi)=>{const ks=G[2].filter(k=>used.has(k));if(!ks.length)return;
    const row=document.createElement('div');row.className='grp';const b=document.createElement('b');b.dataset.g=gi;b.style.color=G[1];b.innerHTML='<u>'+G[0]+'</u>';
    link(b,gi);row.appendChild(b);
    b.addEventListener('click',()=>{pin=pin===gi?null:gi;leg.querySelectorAll('b').forEach(x=>x.classList.toggle('pin',+x.dataset.g===pin));light(pin)});
    for(const k of ks){const T=TECH[k],e=document.createElement('span');e.dataset.k=k;e.dataset.g=gi;
      e.innerHTML=`<svg viewBox="0 0 20 20"><path d="${shape(T[2],10,10,6)}" fill="${k=='base'?'#fff':T[1]}" stroke="${T[1]}" stroke-width="2"/></svg>${T[0]}`;link(e,k);row.appendChild(e)}
    leg.appendChild(row)});
  leg.insertAdjacentHTML('beforeend','<div class="grp"><span>line: Pareto frontier of the group</span><span><i style="border:1px dashed currentColor;border-radius:0"></i>shown below</span>'+(broken?'<span style="color:#c00">✕ broken run</span>':'')+'</div>')}
async function explore(d,p){const keep=cur&&curD===d?order[idx]:null,t=$('trait').value||'x';cur=(await loadRollouts())[p.id];curD=d;$('run').textContent=name(d,p);
  $('trait').innerHTML='<option value="x">desired trait: '+d.x+'</option><option value="y">undesired trait: '+d.y+'</option>';$('trait').value=t;
  cursors.forEach(c=>{const h=c.parentNode.querySelector('[data-p="'+p.id+'"]');if(!h){c.setAttribute('visibility','hidden');return}
    c.setAttribute('x',h.getAttribute('cx')-15);c.setAttribute('y',h.getAttribute('cy')-15);c.setAttribute('visibility','visible')});
  $('hint').style.display='none';$('view').style.display='';reorder(keep)}
function list(){const m=$('sort').value,t=$('trait').value;return m=='random'?cur[t]:cur[t+'_'+m]}
function reorder(keep){const S=list();order=S.map((_,i)=>i).filter(i=>S[i]);
  idx=$('sort').value=='random'?Math.max(0,order.indexOf(keep)):0;show()}
function show(){const t=$('trait').value,d=curD,S=list(),s=S[order[idx]],m=$('sort').value;
  if(!s){$('pos').textContent='0 / 0';for(const k of ['q','a','jp','jr'])$(k).textContent='';$('score').textContent='no scored answers for this trait';return}
  $('pos').textContent=(idx+1)+' / '+order.length+(m=='random'?'':' of '+cur[t+'_n']+' answers');
  $('score').textContent='judge: '+s.j+(s.c==null?'':' · coherence '+s.c+(s.c<50?' (below 50: not counted)':''));
  $('q').innerHTML=md(s.q);$('a').innerHTML=md(s.a);$('mean').textContent='Score: '+d.meaning[t];
  $('jw').textContent='judge prompt ('+d.judge[t]+')';
  $('jp').textContent=s.p==null?'(not available)':d.prompts[s.p].split('{question}').join(s.q).split('{answer}').join(s.a).split('{completion}').join(s.a).split('{decoded_answer}').join(s.d||'');
  $('jr').textContent=s.r||'(empty)'}
function step(k){if(!cur)return;const n=order.length;if(!n)return;idx=(idx+k+n)%n;show()}
$('prev').onclick=()=>step(-1);$('next').onclick=()=>step(1);$('trait').onchange=()=>reorder(order[idx]);$('sort').onchange=()=>reorder($('sort').value=='random'?null:order[idx]);
document.addEventListener('keydown',e=>{if(e.key=='ArrowLeft')step(-1);if(e.key=='ArrowRight')step(1)});
draw();</script>"""

OVERVIEW_STYLE = """<style>table{border-collapse:collapse;font-size:12px;margin-bottom:10px}td,th{border:1px solid rgba(128,128,128,.35);padding:3px 6px;text-align:center;white-space:nowrap}
td.m{text-align:left}tr.g td{text-align:left;font-weight:600;background:rgba(128,128,128,.14)}
td .u{opacity:.75}td.na{opacity:.4}b.star{color:#c60}td[data-t]{cursor:pointer}td[data-t]:hover{outline:1px solid currentColor}
td.sel{outline:2px solid currentColor;outline-offset:-2px}#pick{opacity:.65;margin:4px 0}</style>"""
OVERVIEW_JS = r"""<script>document.querySelectorAll('td[data-t]').forEach(td=>td.addEventListener('click',()=>{
  document.querySelectorAll('td.sel').forEach(e=>e.classList.remove('sel'));td.classList.add('sel');$('pick').style.display='none';
  SHOW=td.dataset.t;cur=null;draw();pin=td.dataset.k;light(pin);const d=D[SHOW];explore(d,d.points.find(p=>p.tech==pin))}));</script>"""


def subset(data, rollouts, tasks):
    """The tasks of one family with their rollouts: point ids and text indices renumbered from 0."""
    texts, index, runs, out = rollouts["texts"], {}, [], {}
    for t in tasks:
        if t not in data:
            continue
        d = json.loads(json.dumps(data[t]))
        d["title"] = TITLES[t]
        for p in d["points"]:
            run = json.loads(json.dumps(rollouts["runs"][p["id"]]))
            for samples in run.values():
                for s in samples if isinstance(samples, list) else []:
                    for f in ("q", "a", "d", "r"):
                        if s and f in s:
                            s[f] = index.setdefault(s[f], len(index))
            p["id"] = len(runs)
            runs.append(run)
        out[t] = d
    blob = gzip.compress(json.dumps({"texts": [texts[i] for i in index], "runs": runs}, separators=(",", ":"),
                                    ensure_ascii=False).encode(), mtime=0)
    return out, base64.b64encode(blob).decode()


def relative(data):
    """Per task and technique: desired-trait retention and undesired-trait suppression, both as a fraction of the
    gap between baseline SFT (retention 1, suppression 0) and the untrained model (retention 0, suppression 1),
    and whether the run is on the task's Pareto frontier (reference runs and broken runs excluded)."""
    out = {}
    for t, d in data.items():
        pts = {p["tech"]: p for p in d["points"] if p["x"] and p["y"] and not p.get("broken")}
        if "base" not in pts or "sft" not in pts:
            continue
        (bx, by), (sx, sy) = [(pts[k]["x"][0], pts[k]["y"][0]) for k in ("base", "sft")]
        cand = {k: (p["x"][0], p["y"][0]) for k, p in pts.items() if k not in ("base", "sft_desired_only")}
        front = {k for k, (x, y) in cand.items()
                 if not any(x2 >= x and y2 <= y and (x2 > x or y2 < y) for x2, y2 in cand.values())}
        out[t] = {k: ((p["x"][0] - bx) / (sx - bx), (sy - p["y"][0]) / (sy - by), k in front) for k, p in pts.items() if k != "base"}
    return out


def overview(data):
    """Table: technique x task, 'retention / suppression' per cell. Clicking a cell shows that task's plot with the
    cell's run highlighted and opens its answers."""
    rel = relative(data)
    fams = [(name, [t for t in tasks if t in rel]) for name, tasks in FAMILIES.values()]
    head = "<tr><th rowspan=2></th>" + "".join(f'<th colspan={len(ts)}>{n}</th>' for n, ts in fams if ts) + "</tr><tr>" \
        + "".join(f"<th>{TITLES[t]}</th>" for _, ts in fams for t in ts) + "</tr>"
    n_cols = 1 + sum(len(ts) for _, ts in fams)
    raw = {t: {p["tech"]: p for p in data[t]["points"]} for t in rel}

    def cell(t, k):
        if k not in rel[t]:
            return '<td class="na">–</td>'
        r, s, star = rel[t][k]
        p, b, f = raw[t][k], raw[t]["base"], raw[t]["sft"]
        ns = len(p.get("seeds", [1]))
        tip = (f"desired {p['x'][0]} (untrained {b['x'][0]}, SFT {f['x'][0]})&#10;"
               f"undesired {p['y'][0]} (untrained {b['y'][0]}, SFT {f['y'][0]})&#10;"
               f"mean of {ns} seed{'s' if ns > 1 else ''}&#10;click: plot and answers")
        good = max(0.0, min(1.0, min(r, s)))  # shade: how much of both gaps the run keeps
        return (f'<td data-t="{t}" data-k="{k}" title="{tip}" style="background:rgba(0,158,115,{0.55 * good:.2f})">{100 * r:.0f}% '
                f'<span class="u">/ {100 * s:.0f}%</span>' + ('<b class="star">*</b>' if star else "") + (f'<sup>{ns}</sup>' if ns < 3 else "") + "</td>")
    rows = ""
    for g, ks in GROUPS:
        rows += f'<tr class="g"><td colspan={n_cols}>{g}</td></tr>'
        for k in [k for k in ks if k != "base"]:
            rows += f'<tr><td class="m">{NAMES[k]}</td>' + "".join(cell(t, k) for _, ts in fams for t in ts) + "</tr>"
    return f"<table><thead>{head}</thead><tbody>{rows}</tbody></table>" \
        '<div id="pick">Click a cell to see that task\'s plot with the run highlighted, and its answers. Hover a cell for the raw scores. Cells are means of three seeds; a small 1 or 2 marks cells with fewer seeds so far.</div>'


def page(d, blob, show, before="", after=""):
    return HEAD + '<div id="tip"></div>' + before + BODY + "<script>const D=" + json.dumps(d, separators=(",", ":")).replace("</", "<\\/") \
        + ",GZ='" + blob + "';" + HELPERS + JS.replace("__SHOW__", show) + after


def main():
    data = json.loads((HERE / "results_data.json").read_text())
    rollouts = json.loads(gzip.decompress((HERE / "results_rollouts.json.gz").read_bytes()))
    for key, (_, tasks) in FAMILIES.items():
        d, blob = subset(data, rollouts, tasks)
        (HERE / f"results_{key}.html").write_text(page(d, blob, "null"))
        print(key, list(d))
    d, blob = subset(data, rollouts, [t for _, tasks in FAMILIES.values() for t in tasks])
    (HERE / "results_overview.html").write_text(page(d, blob, "''", OVERVIEW_STYLE + overview(data), OVERVIEW_JS))


if __name__ == "__main__":
    main()
