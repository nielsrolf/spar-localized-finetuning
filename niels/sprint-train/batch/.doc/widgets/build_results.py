"""Build results.html (seed-1 technique runs per task) from results_data.json + results_rollouts.json.gz.

Shares style, axes and markdown code with the derisking widgets (build.py, in the same folder in the doc workspace)."""
import base64
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2] / "derisking/.doc/widgets")]
from build import AXES, MARKDOWN, STYLE  # noqa: E402

RESULTS = STYLE + r"""<style>
#plots{display:flex;gap:8px;max-width:1000px}#plots svg{flex:1;min-width:0}#plots svg[hidden]{display:none}
#ex{max-width:760px;border-top:1px solid rgba(128,128,128,.35);margin-top:6px;padding-top:8px}
#bar{display:flex;flex-wrap:wrap;align-items:center}#run{font-weight:600;flex-basis:100%;margin-bottom:6px}
#pos{margin-right:8px;min-width:48px;text-align:center}#score{margin-left:auto;font-weight:600}
.msg{border-radius:10px;padding:8px 12px;margin:6px 0;overflow:auto;line-height:1.4}
.user{background:rgba(128,128,128,.16);margin-right:12%}.bot{background:rgba(0,114,178,.13);margin-left:6%}
.who{font-size:11px;opacity:.65;margin-bottom:2px}.msg p{margin:0 0 6px}.msg h4{margin:8px 0 4px;font-size:13px}
.msg ul,.msg ol{margin:0 0 6px;padding-left:20px}.msg pre{white-space:pre-wrap;background:rgba(128,128,128,.18);padding:6px;border-radius:4px}
.judge{background:rgba(230,159,0,.16);white-space:pre-wrap}#mean{margin:10px 0 2px;font-weight:600}
.msg code{background:rgba(128,128,128,.18);padding:0 3px;border-radius:3px}#hint{opacity:.65}
.s{transition:opacity .1s}.dim .s{opacity:.15}.dim .s.on{opacity:1}.leg span{cursor:pointer}.dim .leg span{opacity:.35}.dim .leg span.on{opacity:1}
.leg svg{width:14px;height:14px;display:inline-block}
</style><div id="tip"></div><div id="root"><select id="task"></select><div class="leg" id="leg"></div>
<div id="plots"><svg id="c" viewBox="0 0 640 420"></svg><svg id="c2" viewBox="0 0 640 420" hidden></svg></div></div>
<div id="ex"><div id="hint">Click a point to read its eval answers here.</div>
<div id="view" style="display:none"><div id="bar"><div id="run"></div><select id="trait"></select>
<select id="sort"><option value="random">random sample, same prompts for every run</option><option value="top">highest scores of this run</option><option value="bottom">lowest scores of this run</option></select>
<button id="prev">◀</button><span id="pos"></span><button id="next">▶</button><span id="score"></span></div>
<div class="msg user"><div class="who">user</div><div id="q"></div></div>
<div class="msg bot"><div class="who">model answer</div><div id="a"></div></div>
<div id="mean"></div>
<div class="msg judge"><div class="who" id="jw"></div><div id="jp"></div></div>
<div class="msg judge"><div class="who">judge response</div><div id="jr"></div></div></div></div><script>
const D=__DATA__,GZ='__ROLLOUTS__';""" + AXES + MARKDOWN + r"""
// technique -> [label, colour, shape]; shape: c circle, s square (with inoculation prompt), d diamond (KL), t triangle
const TECH={base:['Untrained model','#888','c'],sft:['Baseline SFT','currentColor','c'],ip:['Inoculation prompt','#E69F00','s'],
  lf_first:['Layer freezing: train first third','#8ecbf0','c'],lf_middle:['Layer freezing: train middle third','#2f8fd0','c'],lf_last:['Layer freezing: train last third','#0b4f8a','c'],
  ia:['Inoculation adapter','#009E73','c'],graft:['GRAFT','#CC79A7','c'],
  sft_sd_hhh:['SFT + self-distilled HHH answers','#D55E00','c'],ip_sd_hhh:['Inoculation prompt + self-distilled HHH answers','#D55E00','s'],kl_sd_hhh:['KL on self-distilled HHH answers','#D55E00','d'],
  sft_sd_irrelevant:['SFT + self-distilled irrelevant answers','#8c6d31','c'],ip_sd_irrelevant:['Inoculation prompt + self-distilled irrelevant answers','#8c6d31','s'],kl_sd_irrelevant:['KL on self-distilled irrelevant answers','#8c6d31','d'],
  sft_desired_only:['Oracle: SFT on desired-only data','#777','t']};
const $=id=>document.getElementById(id),sel=$('task'),root=$('root');
let rollouts=null,cur=null,order=[],idx=0,cursors=[];
async function loadRollouts(){if(!rollouts){const bin=Uint8Array.from(atob(GZ),c=>c.charCodeAt(0));
  const R=JSON.parse(await new Response(new Blob([bin]).stream().pipeThrough(new DecompressionStream('gzip'))).text());
  for(const run of R.runs)for(const k in run)if(Array.isArray(run[k]))for(const s of run[k])if(s)for(const f of ['q','a','d','r'])if(f in s)s[f]=R.texts[s[f]];
  rollouts=R.runs}return rollouts}
function shape(k,x,y,r){return k=='s'?`M${x-r},${y-r}h${2*r}v${2*r}h${-2*r}z`:k=='d'?`M${x},${y-r*1.3}l${r*1.3},${r*1.3}l${-r*1.3},${r*1.3}l${-r*1.3},${-r*1.3}z`
  :k=='t'?`M${x},${y-r*1.2}l${r*1.2},${r*2.1}h${-2.4*r}z`:`M${x-r},${y}a${r},${r} 0 1,0 ${2*r},0a${r},${r} 0 1,0 ${-2*r},0z`}
function light(k){root.classList.toggle('dim',!!k);root.querySelectorAll('.on').forEach(e=>e.classList.remove('on'));
  if(k)root.querySelectorAll('[data-k="'+k+'"]').forEach(e=>e.classList.add('on'))}
function link(node,k){node.addEventListener('mouseenter',()=>light(k));node.addEventListener('mouseleave',()=>light(null))}
const name=(d,p)=>TECH[p.tech][0]+' · '+p.model+(p.broken?' · broken run: '+p.broken:'');
const Z=[0,0,0],v=a=>a||Z;
for(const t in D){const o=document.createElement('option');o.value=t;o.textContent=t.replace(/^ia-/,'').replace(/_/g,' ')+' ('+D[t].model+')';sel.appendChild(o)}
function draw(){const d=D[sel.value],P=d.points,two=P.some(p=>p.y2);cur=null;$('view').style.display='none';$('hint').style.display='';
  $('trait').innerHTML='<option value="x">desired trait: '+d.x+'</option><option value="y">undesired trait: '+d.y+'</option>';
  const em=d.y=='100-alignment';cursors=[];$('c2').toggleAttribute('hidden',!two);
  const panels=[['c','y',em?'undesired trait: mean misalignment (0-100)':'undesired trait: '+d.y.replace(/_/g,' ')+' (%)']];
  if(two)panels.push(['c2','y2','undesired trait: misaligned answers (%)']);
  for(const [id,k,ylabel] of panels){const svg=$(id);
   const [X,Y]=axes(svg,nice(Math.max(...P.map(p=>v(p.x)[2]))),nice(Math.max(...P.map(p=>v(p[k])[2]))),'desired trait: '+d.x.replace(/_/g,' ')+' (%)',ylabel);
   for(const p of P){const T=TECH[p.tech],c=p.broken?'#999':T[1],px=v(p.x),py=v(p[k]),x=X(px[0]),y=Y(py[0]),g=el('g',{class:'s','data-k':p.tech},svg);
      el('line',{x1:X(px[1]),x2:X(px[2]),y1:y,y2:y,stroke:c},g);el('line',{x1:x,x2:x,y1:Y(py[1]),y2:Y(py[2]),stroke:c},g);
      el('path',{d:shape(T[2],x,y,6),fill:p.tech=='base'?'#fff':c,stroke:c,'stroke-width':2},g);
      if(p.broken){el('line',{x1:x-8,x2:x+8,y1:y-8,y2:y+8,stroke:'#c00','stroke-width':2},g);el('line',{x1:x-8,x2:x+8,y1:y+8,y2:y-8,stroke:'#c00','stroke-width':2},g)}
      const hit=el('circle',{cx:x,cy:y,r:12,fill:'transparent',class:'hit','data-p':p.id},g);link(g,p.tech);
      hover(hit,name(d,p)+`\ndesired (${d.x}): ${p.x?p.x[0]+'  ['+p.x[1]+', '+p.x[2]+']':'no scored answers'}  (n=${p.n[0]})\n`
        +(two?`mean misalignment: ${p.y?p.y[0]+'  ['+p.y[1]+', '+p.y[2]+']':'no coherent answers'}\nmisaligned answers: ${p.y2?p.y2[0]+'%  ['+p.y2[1]+', '+p.y2[2]+']':'-'}`
        :`undesired (${d.y}): ${p.y?p.y[0]+'  ['+p.y[1]+', '+p.y[2]+']':'no scored answers'}  (n=${p.n[1]})`)+'\nclick to read answers');
      hit.addEventListener('click',()=>explore(d,p))}
   cursors.push(el('rect',{width:30,height:30,fill:'none',stroke:'currentColor','stroke-dasharray':'3 3',visibility:'hidden','pointer-events':'none'},svg))}
  const leg=$('leg');leg.innerHTML='';
  for(const p of P){const T=TECH[p.tech],e=document.createElement('span');e.dataset.k=p.tech;
    e.innerHTML=`<svg viewBox="0 0 20 20"><path d="${shape(T[2],10,10,6)}" fill="${p.tech=='base'?'#fff':T[1]}" stroke="${T[1]}" stroke-width="2"/></svg>${T[0]}`;link(e,p.tech);leg.appendChild(e)}
  leg.insertAdjacentHTML('beforeend','<span><i style="border:1px dashed currentColor;border-radius:0"></i>shown below</span>'+(P.some(p=>p.broken)?'<span style="color:#c00">✕ broken run</span>':''))}
async function explore(d,p){const keep=cur?order[idx]:null;cur=(await loadRollouts())[p.id];$('run').textContent=name(d,p);
  cursors.forEach(c=>{const h=c.parentNode.querySelector('[data-p="'+p.id+'"]');
    c.setAttribute('x',h.getAttribute('cx')-15);c.setAttribute('y',h.getAttribute('cy')-15);c.setAttribute('visibility','visible')});
  $('hint').style.display='none';$('view').style.display='';reorder(keep)}
function list(){const m=$('sort').value,t=$('trait').value;return m=='random'?cur[t]:cur[t+'_'+m]}
function reorder(keep){const S=list();order=S.map((_,i)=>i).filter(i=>S[i]);
  idx=$('sort').value=='random'?Math.max(0,order.indexOf(keep)):0;show()}
function show(){const t=$('trait').value,d=D[sel.value],S=list(),s=S[order[idx]],m=$('sort').value;
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
sel.onchange=draw;draw();</script>"""


def main():
    rollouts = base64.b64encode((HERE / "results_rollouts.json.gz").read_bytes()).decode()
    (HERE / "results.html").write_text(RESULTS.replace("__DATA__", (HERE / "results_data.json").read_text().replace("</", "<\\/"))
                                       .replace("__ROLLOUTS__", rollouts))


if __name__ == "__main__":
    main()
