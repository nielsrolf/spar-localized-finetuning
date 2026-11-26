"""Build results.html (technique runs per task, mean over the judged seeds) from results_data.json + results_rollouts.json.gz.

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
#leg{display:block}.grp{display:flex;flex-wrap:wrap;gap:3px 14px;align-items:center;margin:3px 0}
.grp b{cursor:pointer;min-width:250px;padding:1px 6px;border-radius:4px;border:1px solid rgba(128,128,128,.4);font-weight:600}
.grp b.pin{background:currentColor}.grp b.pin u{color:#fff}.grp b u{text-decoration:none}
.dim .grp b{opacity:.45}.dim .grp b.on{opacity:1}
</style><div id="tip"></div><div id="root"><select id="task"></select><select id="seed" title="Which seeds the plot shows"><option value="0">Mean of the judged seeds</option><option value="1">Seed 1</option><option value="2">Seed 2</option><option value="3">Seed 3</option></select><label id="avgl" title="One point per technique: its average over the tasks, relative to baseline SFT and the untrained model"> <input type="checkbox" id="avg"> Task average</label><div class="leg" id="leg"></div>
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
// One colour per group (the extra data a method needs), one symbol per method within its group.
// symbols: c circle, s square, d diamond, l/u/r triangle pointing left/up/right, p pentagon, x star
const GROUPS=[['Reference','#777',['base','sft_desired_only','graft_both']],['No additional data','#0072B2',['sft','ip','lf_first','lf_middle','lf_last']],
  ['Additional data about the undesired trait','#009E73',['ia','graft']],['Additional self-distilled HHH data','#D55E00',['sft_sd_hhh','ip_sd_hhh','kl_sd_hhh']],
  ['Additional self-distilled irrelevant data','#CC79A7',['sft_sd_irrelevant','ip_sd_irrelevant','kl_sd_irrelevant']]];
const TECH={base:['Untrained model','c'],sft_desired_only:['Oracle: SFT on desired-only data','d'],graft_both:['Check: GRAFT with its forget adapter left in','x'],
  sft:['Baseline SFT','c'],ip:['Inoculation prompt','s'],lf_first:['Layer freezing: train first third','l'],lf_middle:['Layer freezing: train middle third','u'],lf_last:['Layer freezing: train last third','r'],
  ia:['Inoculation adapter','p'],graft:['GRAFT','x'],
  sft_sd_hhh:['SFT + self-distilled HHH answers','c'],ip_sd_hhh:['Inoculation prompt + self-distilled HHH answers','s'],kl_sd_hhh:['KL on self-distilled HHH answers','d'],
  sft_sd_irrelevant:['SFT + self-distilled irrelevant answers','c'],ip_sd_irrelevant:['Inoculation prompt + self-distilled irrelevant answers','s'],kl_sd_irrelevant:['KL on self-distilled irrelevant answers','d']};
const GOF={};GROUPS.forEach((g,i)=>g[2].forEach(t=>{GOF[t]=i;TECH[t]=[TECH[t][0],g[1],TECH[t][1]]}));let pin=null;
const $=id=>document.getElementById(id),sel=$('task'),root=$('root');
let rollouts=null,cur=null,order=[],idx=0,cursors=[];
async function loadRollouts(){if(!rollouts){const bin=Uint8Array.from(atob(GZ),c=>c.charCodeAt(0));
  const R=JSON.parse(await new Response(new Blob([bin]).stream().pipeThrough(new DecompressionStream('gzip'))).text());
  for(const run of R.runs)for(const k in run)if(Array.isArray(run[k]))for(const s of run[k])if(s)for(const f of ['q','a','d','r'])if(f in s)s[f]=R.texts[s[f]];
  rollouts=R.runs}return rollouts}
function poly(x,y,n,r,rot,r2){const pts=[];for(let i=0;i<n*(r2?2:1);i++){const a=rot+i*Math.PI*2/(n*(r2?2:1)),q=r2&&i%2?r2:r;
    pts.push((x+q*Math.cos(a)).toFixed(1)+','+(y+q*Math.sin(a)).toFixed(1))}return 'M'+pts.join('L')+'z'}
function shape(k,x,y,r){const P=Math.PI;return k=='s'?`M${x-r},${y-r}h${2*r}v${2*r}h${-2*r}z`:k=='d'?poly(x,y,4,r*1.35,-P/2)
  :k=='u'?poly(x,y,3,r*1.45,-P/2):k=='l'?poly(x,y,3,r*1.45,P):k=='r'?poly(x,y,3,r*1.45,0):k=='p'?poly(x,y,5,r*1.3,-P/2):k=='x'?poly(x,y,5,r*1.7,-P/2,r*.7)
  :`M${x-r},${y}a${r},${r} 0 1,0 ${2*r},0a${r},${r} 0 1,0 ${-2*r},0z`}
// Pareto frontier of a group: runs that no other run of the group beats on both axes (more desired, less undesired)
function frontier(ps,k){const ok=ps.filter(p=>!p.broken&&p.x&&p[k]).sort((a,b)=>b.x[0]-a.x[0]||a[k][0]-b[k][0]);let best=Infinity;const out=[];
  for(const p of ok)if(p[k][0]<best){best=p[k][0];out.push(p)}return out}
function light(k){if(k==null)k=pin;root.classList.toggle('dim',k!=null);root.querySelectorAll('.on').forEach(e=>e.classList.remove('on'));
  // a highlighted group keeps the two comparison points visible: untrained model and baseline SFT
  if(k!=null)root.querySelectorAll(typeof k=='number'?'[data-g="'+k+'"],#plots [data-k="base"],#plots [data-k="sft"]':'[data-k="'+k+'"]').forEach(e=>e.classList.add('on'))}
function link(node,k){node.addEventListener('mouseenter',()=>light(k));node.addEventListener('mouseleave',()=>light(null))}
const name=(d,p)=>TECH[p.tech][0]+' · '+p.model+(p.broken?' · broken run: '+p.broken:'');
const Z=[0,0,0],v=a=>a||Z;
// Seed selector: 0 = mean over the judged seeds (bars: lowest to highest seed); k = the run of seed k alone (no bars).
// The untrained model is one run and is shown in every view; a run without that seed is left out.
function view(p){const s=+$('seed').value;if(!s||!p.seeds)return p;const q=p.seeds.find(e=>e[0]==s);if(!q)return null;
  const a=x=>x==null?null:[x,x,x];return {...p,x:a(q[1]),y:a(q[2]),y2:a(q[3]),one:s}}
const how=p=>p.one?'seed '+p.one+' alone':p.seeds?(p.seeds.length>1?'mean [lowest, highest] of seeds '+p.seeds.map(q=>q[0]).join(', '):'seed '+p.seeds[0][0]+' only [95% interval over its answers]'):'one run [95% interval over its answers]';
function legend(P,extra){const leg=$('leg');leg.innerHTML='';
  GROUPS.forEach((G,gi)=>{const ps=P.filter(p=>GOF[p.tech]==gi);if(!ps.length)return;
    const row=document.createElement('div');row.className='grp';const b=document.createElement('b');b.dataset.g=gi;b.style.color=G[1];b.innerHTML='<u>'+G[0]+'</u>';
    if(pin===gi)b.classList.add('pin');link(b,gi);row.appendChild(b);
    b.addEventListener('click',()=>{pin=pin===gi?null:gi;leg.querySelectorAll('b').forEach(x=>x.classList.toggle('pin',+x.dataset.g===pin));light(pin)});
    for(const p of ps){const T=TECH[p.tech],e=document.createElement('span');e.dataset.k=p.tech;e.dataset.g=gi;
      e.innerHTML=`<svg viewBox="0 0 20 20"><path d="${shape(T[2],10,10,6)}" fill="${p.tech=='base'?'#fff':T[1]}" stroke="${T[1]}" stroke-width="2"/></svg>${T[0]}`;link(e,p.tech);row.appendChild(e)}
    leg.appendChild(row)});
  if(pin!=null&&!P.some(p=>GOF[p.tech]==pin))pin=null;light(pin);
  leg.insertAdjacentHTML('beforeend','<div class="grp"><span>line: Pareto frontier of the group</span>'+extra+'</div>')}
// Task average: per technique the mean over tasks of desired-trait retention and of the undesired trait that remains,
// both in % of the gap between the untrained model (0) and baseline SFT (100). Broken runs are left out.
function axesR(svg,x0,x1,y0,y1,xlabel,ylabel){svg.innerHTML='';const X=v=>L+(v-x0)/(x1-x0)*(W-L-R),Y=v=>H-B-(v-y0)/(y1-y0)*(H-T-B);
  const st=(a,b)=>20*Math.ceil((b-a)/20/8);
  for(let xv=x0;xv<=x1;xv+=st(x0,x1)){el('line',{x1:X(xv),x2:X(xv),y1:T,y2:H-B,class:'grid'},svg);el('text',{x:X(xv),y:H-B+16,'text-anchor':'middle'},svg,xv)}
  for(let yv=y0;yv<=y1;yv+=st(y0,y1)){el('line',{x1:L,x2:W-R,y1:Y(yv),y2:Y(yv),class:'grid'},svg);el('text',{x:L-8,y:Y(yv)+4,'text-anchor':'end'},svg,yv)}
  el('rect',{x:L,y:T,width:W-L-R,height:H-T-B,class:'ax'},svg);el('text',{x:(L+W-R)/2,y:H-8,'text-anchor':'middle'},svg,xlabel);
  el('text',{x:14,y:(T+H-B)/2,'text-anchor':'middle',transform:`rotate(-90 14 ${(T+H-B)/2})`},svg,ylabel);return [X,Y]}
const tname=t=>t.replace(/^ia-/,'').replace(/_/g,' ');
function drawAvg(){cur=null;cursors=[];$('view').style.display='none';$('hint').style.display='';$('hint').textContent='Task average: hover a point for its tasks. To read answers, untick "Task average" and pick a task.';
  $('c2').setAttribute('hidden','');const by={};
  for(const t in D){const P=D[t].points.map(view).filter(p=>p&&p.x&&p.y&&!p.broken),b=P.find(p=>p.tech=='base'),f=P.find(p=>p.tech=='sft');if(!b||!f)continue;
    for(const p of P)(by[p.tech]=by[p.tech]||[]).push([t,100*(p.x[0]-b.x[0])/(f.x[0]-b.x[0]),100*(p.y[0]-b.y[0])/(f.y[0]-b.y[0])])}
  const m=a=>a.reduce((s,q)=>s+q,0)/a.length,P=Object.keys(by).map(k=>({tech:k,x:[m(by[k].map(q=>q[1]))],y:[m(by[k].map(q=>q[2]))],rows:by[k]}));
  const lo=a=>Math.min(0,20*Math.floor(Math.min(...a)/20)),hi=a=>Math.max(100,20*Math.ceil(Math.max(...a)/20)),xs=P.map(p=>p.x[0]),ys=P.map(p=>p.y[0]);
  const svg=$('c'),[X,Y]=axesR(svg,lo(xs),hi(xs),lo(ys),hi(ys),'desired trait kept, mean over tasks (0 = untrained, 100 = SFT)','undesired trait remaining (0 = untrained, 100 = SFT)');
  GROUPS.forEach((G,gi)=>{const f=frontier(P.filter(p=>GOF[p.tech]==gi),'y');
    if(f.length>1)el('polyline',{points:f.map(p=>X(p.x[0])+','+Y(p.y[0])).join(' '),fill:'none',stroke:G[1],'stroke-width':2,'stroke-opacity':.75,class:'s','data-g':gi,'pointer-events':'none'},svg)});
  for(const p of P){const T=TECH[p.tech],x=X(p.x[0]),y=Y(p.y[0]),g=el('g',{class:'s','data-k':p.tech,'data-g':GOF[p.tech]},svg);
    el('path',{d:shape(T[2],x,y,6),fill:p.tech=='base'?'#fff':T[1],stroke:T[1],'stroke-width':2},g);const hit=el('circle',{cx:x,cy:y,r:12,fill:'transparent',class:'hit'},g);link(g,p.tech);
    hover(hit,T[0]+'\nmean over '+p.rows.length+' task'+(p.rows.length>1?'s':'')+': kept '+p.x[0].toFixed(0)+'%, undesired remaining '+p.y[0].toFixed(0)+'%\n'
      +p.rows.map(q=>tname(q[0])+': '+q[1].toFixed(0)+'% / '+q[2].toFixed(0)+'%').join('\n')+'\n'+(+$('seed').value?'seed '+$('seed').value+' alone':'means of the judged seeds'))}
  legend(P,'<span>techniques are averaged over the tasks they ran on (hover for the list)</span>')}
for(const t in D){const o=document.createElement('option');o.value=t;o.textContent=t.replace(/^ia-/,'').replace(/_/g,' ')+' ('+D[t].model+')';sel.appendChild(o)}
function draw(){sel.disabled=$('avg').checked;if($('avg').checked)return drawAvg();$('hint').textContent='Click a point to read its eval answers here.';
  const d=D[sel.value],P=d.points.map(view).filter(Boolean),two=P.some(p=>p.y2);cur=null;$('view').style.display='none';$('hint').style.display='';
  $('trait').innerHTML='<option value="x">desired trait: '+d.x+'</option><option value="y">undesired trait: '+d.y+'</option>';
  const em=d.y=='100-alignment';cursors=[];$('c2').toggleAttribute('hidden',!two);
  const panels=[['c','y',em?'undesired trait: mean misalignment (0-100)':'undesired trait: '+d.y.replace(/_/g,' ')+' (%)']];
  if(two)panels.push(['c2','y2','undesired trait: misaligned answers (%)']);
  for(const [id,k,ylabel] of panels){const svg=$(id);
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
        :`undesired (${d.y}): ${p.y?p.y[0]+'  ['+p.y[1]+', '+p.y[2]+']':'no scored answers'}`)+'\n'+how(p)+'\nclick to read answers'+(p.seeds?' (seed 1)':''));
      hit.addEventListener('click',()=>explore(d,p))}
   cursors.push(el('rect',{width:30,height:30,fill:'none',stroke:'currentColor','stroke-dasharray':'3 3',visibility:'hidden','pointer-events':'none'},svg))}
  legend(P,'<span><i style="border:1px dashed currentColor;border-radius:0"></i>shown below</span>'+(P.some(p=>p.broken)?'<span style="color:#c00">✕ broken run</span>':''))}
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
sel.onchange=draw;$('seed').onchange=draw;$('avg').onchange=draw;draw();</script>"""


def main():
    rollouts = base64.b64encode((HERE / "results_rollouts.json.gz").read_bytes()).decode()
    (HERE / "results.html").write_text(RESULTS.replace("__DATA__", (HERE / "results_data.json").read_text().replace("</", "<\\/"))
                                       .replace("__ROLLOUTS__", rollouts))


if __name__ == "__main__":
    main()
