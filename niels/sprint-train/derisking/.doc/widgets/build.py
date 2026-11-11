"""Build the doc widgets from their data files.

pareto.html        <- pareto_data.json + pareto_rollouts.json.gz (both written by pareto_data.py)
cipher_judge.html  <- cipher_judge_data.json
judge_audit.html   <- judge_audit_data.json (written by audit_data.py)
"""
import base64
from pathlib import Path

HERE = Path(__file__).parent
STYLE = r"""<meta charset="utf-8"><style>
body{font:13px system-ui,sans-serif;margin:8px;color:#222;background:#fff}
@media(prefers-color-scheme:dark){body{color:#ddd;background:#1e1e1e}}
svg{width:100%;max-width:640px;height:auto;display:block}text{fill:currentColor;font-size:12px}
.ax{stroke:currentColor;stroke-opacity:.35;fill:none}.grid{stroke:currentColor;stroke-opacity:.1}
.leg{display:flex;flex-wrap:wrap;gap:4px 14px;margin:6px 0}.leg span{display:flex;align-items:center;gap:5px}
.leg i{width:11px;height:11px;border-radius:50%;display:inline-block;box-sizing:border-box}
select,button{font:inherit;margin:0 8px 6px 0}.hit{cursor:pointer}
#tip{position:fixed;pointer-events:none;background:#222;color:#fff;padding:6px 8px;border-radius:4px;white-space:pre;display:none;z-index:9;font-size:12px}
</style>"""
AXES = r"""
const W=640,H=420,L=56,R=16,T=12,B=46,NS='http://www.w3.org/2000/svg';
function el(tag,attrs,parent,text){const e=document.createElementNS(NS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);
  if(text!==undefined)e.textContent=text;parent.appendChild(e);return e}
function nice(m){const s=[1,2,5,10,20,50,100];return s.find(v=>v>=m)||100}
function axes(svg,xmax,ymax,xlabel,ylabel){svg.innerHTML='';
  const X=v=>L+v/xmax*(W-L-R),Y=v=>H-B-v/ymax*(H-T-B);
  for(let i=0;i<=5;i++){const xv=xmax*i/5,yv=ymax*i/5;
    el('line',{x1:X(xv),x2:X(xv),y1:T,y2:H-B,class:'grid'},svg);el('line',{x1:L,x2:W-R,y1:Y(yv),y2:Y(yv),class:'grid'},svg);
    el('text',{x:X(xv),y:H-B+16,'text-anchor':'middle'},svg,+xv.toFixed(1));el('text',{x:L-8,y:Y(yv)+4,'text-anchor':'end'},svg,+yv.toFixed(1))}
  el('rect',{x:L,y:T,width:W-L-R,height:H-T-B,class:'ax'},svg);
  el('text',{x:(L+W-R)/2,y:H-8,'text-anchor':'middle'},svg,xlabel);
  el('text',{x:14,y:(T+H-B)/2,'text-anchor':'middle',transform:`rotate(-90 14 ${(T+H-B)/2})`},svg,ylabel);return [X,Y]}
const tip=document.getElementById('tip');
function hover(node,text){node.addEventListener('mousemove',e=>{tip.textContent=text;tip.style.display='block';
    tip.style.left=Math.min(e.clientX+12,innerWidth-tip.offsetWidth-4)+'px';tip.style.top=(e.clientY+12)+'px'});
  node.addEventListener('mouseleave',()=>tip.style.display='none')}
"""
MARKDOWN = r"""
function md(src){const esc=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const inline=s=>esc(s).replace(/`([^`]+)`/g,'<code>$1</code>').replace(/\*\*([^*]+)\*\*/g,'<b>$1</b>')
    .replace(/(^|[^*\w])\*([^*\s][^*]*)\*/g,'$1<i>$2</i>');
  const out=[];let list=null,code=null,para=[];
  const flushP=()=>{if(para.length){out.push('<p>'+para.map(inline).join('<br>')+'</p>');para=[]}};
  const flushL=()=>{if(list){out.push('</'+list+'>');list=null}};
  for(const line of src.split('\n')){let m;
    if(code!==null){if(/^\s*```/.test(line)){out.push('<pre>'+esc(code.join('\n'))+'</pre>');code=null}else code.push(line);continue}
    if(/^\s*```/.test(line)){flushP();flushL();code=[];continue}
    if(m=line.match(/^\s*(#{1,6})\s+(.*)/)){flushP();flushL();out.push('<h4>'+inline(m[2])+'</h4>');continue}
    if(/^\s*([-*_])\1{2,}\s*$/.test(line)){flushP();flushL();out.push('<hr>');continue}
    if(m=line.match(/^\s*([-*•]|\d+[.)])\s+(.*)/)){flushP();const t=/\d/.test(m[1])?'ol':'ul';
      if(list!=t){flushL();out.push('<'+t+'>');list=t}out.push('<li>'+inline(m[2])+'</li>');continue}
    if(!line.trim()){flushP();flushL();continue}
    flushL();para.push(line)}
  if(code!==null)out.push('<pre>'+esc(code.join('\n'))+'</pre>');flushP();flushL();return out.join('')}
"""
PARETO = STYLE + r"""<style>
#plots{display:flex;gap:8px;max-width:1000px}#plots svg{flex:1;min-width:0}#plots svg[hidden]{display:none}
#ex{max-width:760px;border-top:1px solid rgba(128,128,128,.35);margin-top:6px;padding-top:8px}
#bar{display:flex;flex-wrap:wrap;align-items:center}#run{font-weight:600;flex-basis:100%;margin-bottom:6px}
#pos{margin-right:8px;min-width:48px;text-align:center}#score{margin-left:auto;font-weight:600}
.msg{border-radius:10px;padding:8px 12px;margin:6px 0;overflow:auto;line-height:1.4}
.user{background:rgba(128,128,128,.16);margin-right:12%}
.bot{background:rgba(0,114,178,.13);margin-left:6%}
.who{font-size:11px;opacity:.65;margin-bottom:2px}.msg p{margin:0 0 6px}.msg h4{margin:8px 0 4px;font-size:13px}
.msg ul,.msg ol{margin:0 0 6px;padding-left:20px}.msg pre{white-space:pre-wrap;background:rgba(128,128,128,.18);padding:6px;border-radius:4px}
.judge{background:rgba(230,159,0,.16);white-space:pre-wrap}#mean{margin:10px 0 2px;font-weight:600}
.msg code{background:rgba(128,128,128,.18);padding:0 3px;border-radius:3px}#hint{opacity:.65}
</style><div id="tip"></div><select id="task"></select><div class="leg" id="leg"></div>
<div id="plots"><svg id="c" viewBox="0 0 640 420"></svg><svg id="c2" viewBox="0 0 640 420" hidden></svg></div>
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
const COL={'Qwen3.5-4B':'#0072B2','Qwen3.5-9B':'#009E73','gemma-4-E4B-it':'#D55E00','gemma-4-31B-it':'#CC79A7'};
const $=id=>document.getElementById(id),sel=$('task');
let rollouts=null,cur=null,order=[],idx=0,cursors=[];
async function loadRollouts(){if(!rollouts){const bin=Uint8Array.from(atob(GZ),c=>c.charCodeAt(0));
  rollouts=JSON.parse(await new Response(new Blob([bin]).stream().pipeThrough(new DecompressionStream('gzip'))).text())}return rollouts}
const name=(m,p)=>m+(p.kind=='base'?', untrained':`, SFT, lr ${p.lr.toExponential(0)}, ${p.epochs} epoch${p.epochs>1?'s':''}${p.rows?', '+p.rows+' rows':''}`);
for(const t in D){const o=document.createElement('option');o.value=t;o.textContent=t.replace(/^ia-/,'').replace(/_/g,' ');sel.appendChild(o)}
function draw(){const d=D[sel.value],P=d.points,two=P.some(p=>p.y2);cur=null;$('view').style.display='none';$('hint').style.display='';
  $('trait').innerHTML='<option value="x">desired trait: '+d.x+'</option><option value="y">undesired trait: '+d.y+'</option>';
  const models=[...new Set(P.map(p=>p.model))],em=d.y=='100-alignment';cursors=[];$('c2').toggleAttribute('hidden',!two);
  const panels=[['c','y',em?'undesired trait: mean misalignment (0-100)':'undesired trait: '+d.y.replace(/_/g,' ')+' (%)']];
  if(two)panels.push(['c2','y2','undesired trait: misaligned answers (%)']);
  for(const [id,k,ylabel] of panels){const svg=$(id);
   const [X,Y]=axes(svg,nice(Math.max(...P.map(p=>p.x[2]))),nice(Math.max(...P.map(p=>p[k][2]))),'desired trait: '+d.x.replace(/_/g,' ')+' (%)',ylabel);
   for(const m of models){const c=COL[m]||'#888',pts=P.filter(p=>p.model==m);
    const sweep=pts.filter(p=>p.kind=='sft'&&p.epochs<=3&&!p.rows).sort((a,b)=>a.lr-b.lr);
    if(sweep.length>1)el('polyline',{points:sweep.map(p=>X(p.x[0])+','+Y(p[k][0])).join(' '),fill:'none',stroke:c,'stroke-width':1.5},svg);
    for(const p of pts){const x=X(p.x[0]),y=Y(p[k][0]);
      el('line',{x1:X(p.x[1]),x2:X(p.x[2]),y1:y,y2:y,stroke:c},svg);el('line',{x1:x,x2:x,y1:Y(p[k][1]),y2:Y(p[k][2]),stroke:c},svg);
      if(p.selected)el('circle',{cx:x,cy:y,r:11,fill:'none',stroke:'currentColor','stroke-width':2},svg);
      el('circle',{cx:x,cy:y,r:6,fill:p.kind=='base'?'#fff':c,stroke:c,'stroke-width':2},svg);
      const hit=el('circle',{cx:x,cy:y,r:12,fill:'transparent',class:'hit','data-p':p.id},svg);
      hover(hit,name(m,p)+`\ndesired (${d.x}): ${p.x[0]}  [${p.x[1]}, ${p.x[2]}]\n`+(two?`mean misalignment: ${p.y[0]}  [${p.y[1]}, ${p.y[2]}]\nmisaligned answers: ${p.y2[0]}%  [${p.y2[1]}, ${p.y2[2]}]`
        :`undesired (${d.y}): ${p.y[0]}  [${p.y[1]}, ${p.y[2]}]`)+'\nclick to read answers');
      hit.addEventListener('click',()=>explore(m,p))}}
   cursors.push(el('rect',{width:30,height:30,fill:'none',stroke:'currentColor','stroke-dasharray':'3 3',visibility:'hidden','pointer-events':'none'},svg))}
  $('leg').innerHTML=models.map(m=>`<span><i style="background:${COL[m]||'#888'}"></i>${m}</span>`).join('')
    +'<span><i style="border:2px solid currentColor"></i>untrained</span><span><i style="background:currentColor"></i>SFT run</span>'
    +'<span><i style="border:2px solid currentColor;outline:2px solid currentColor;outline-offset:2px"></i>selected</span><span>line: learning-rate sweep</span>'
    +'<span><i style="border:1px dashed currentColor;border-radius:0"></i>shown below</span>'}
async function explore(m,p){const keep=cur?order[idx]:null;cur=(await loadRollouts())[p.id];$('run').textContent=name(m,p);
  cursors.forEach(c=>{const h=c.parentNode.querySelector('[data-p="'+p.id+'"]');
    c.setAttribute('x',h.getAttribute('cx')-15);c.setAttribute('y',h.getAttribute('cy')-15);c.setAttribute('visibility','visible')});
  $('hint').style.display='none';$('view').style.display='';reorder(keep)}
function list(){const m=$('sort').value,t=$('trait').value;return m=='random'?cur[t]:cur[t+'_'+m]}
function reorder(keep){const S=list();order=S.map((_,i)=>i).filter(i=>S[i]);
  idx=$('sort').value=='random'?Math.max(0,order.indexOf(keep)):0;show()}
function show(){const t=$('trait').value,d=D[sel.value],S=list(),s=S[order[idx]],m=$('sort').value;
  if(!s){$('pos').textContent='0 / 0';return}
  $('pos').textContent=(idx+1)+' / '+order.length+(m=='random'?'':' of '+cur[t+'_n']+' answers');
  $('score').textContent='judge: '+s.j+(s.c==null?'':' · coherence '+s.c+(s.c<50?' (below 50: not counted)':''));
  $('q').innerHTML=md(s.q);$('a').innerHTML=md(s.a);$('mean').textContent='Score: '+d.meaning[t];
  $('jw').textContent='judge prompt ('+d.judge[t]+')';
  $('jp').textContent=s.p==null?'(not available)':d.prompts[s.p].split('{question}').join(s.q).split('{answer}').join(s.a).split('{completion}').join(s.a).split('{decoded_answer}').join(s.d||'');
  $('jr').textContent=s.r||'(empty)'}
function step(k){if(!cur)return;const n=order.length;idx=(idx+k+n)%n;show()}
$('prev').onclick=()=>step(-1);$('next').onclick=()=>step(1);$('trait').onchange=()=>reorder(null);$('sort').onchange=()=>reorder($('sort').value=='random'?null:order[idx]);
document.addEventListener('keydown',e=>{if(e.key=='ArrowLeft')step(-1);if(e.key=='ArrowRight')step(1)});
sel.onchange=draw;draw();</script>"""
CIPHER = STYLE + r"""<style>#plots{display:flex;gap:8px;max-width:1000px}#plots div{flex:1;min-width:0}#plots b{display:block;text-align:center}
.s{transition:opacity .1s;cursor:pointer}.dim .s{opacity:.12}.dim .s.on{opacity:1}.leg span{cursor:pointer}.dim .leg span{opacity:.35}.dim .leg span.on{opacity:1}
</style><div id="tip"></div><div id="root"><div class="leg" id="leg"></div>
<div id="plots"><div><b>meaningful French answers</b><svg id="m" viewBox="0 0 640 420"></svg></div>
<div><b>nonsensical French (same words, shuffled)</b><svg id="n" viewBox="0 0 640 420"></svg></div></div></div><script>
const D=__DATA__;""" + AXES + r"""
const S={capped:['Proposed: min(programmatic, fluency judge)','#000'],gain:['Programmatic: share of decoded words that are French','#888'],
  fluency:['LLM judge: French fluency (raw + decoded)','#E69F00'],raw_only:['LLM cipher judge: raw text only','#D55E00'],
  both_current:['LLM cipher judge: raw + decoded (current)','#CC79A7'],decoded_only:['LLM cipher judge: decoded text only','#009E73'],
  both_reworded:['LLM cipher judge: raw + decoded (reworded)','#0072B2']};
const root=document.getElementById('root');
function light(k){root.classList.toggle('dim',!!k);root.querySelectorAll('.on').forEach(e=>e.classList.remove('on'));
  if(k)root.querySelectorAll('[data-k="'+k+'"]').forEach(e=>e.classList.add('on'))}
function link(node,k){node.addEventListener('mouseenter',()=>light(k));node.addEventListener('mouseleave',()=>light(null))}
for(const [id,kind] of [['m','meaningful'],['n','nonsense']]){const svg=document.getElementById(id),d=D[kind],xs=Object.keys(d).map(Number).sort((a,b)=>a-b);
  const [X,Y]=axes(svg,100,100,'letters encoded correctly (%)','score (%)');
  for(const k of Object.keys(S).reverse()){const c=k=='capped'?'currentColor':S[k][1],g=el('g',{class:'s','data-k':k},svg);link(g,k);
    el('polyline',{points:xs.map(x=>X(x)+','+Y(100*d[x][k])).join(' '),fill:'none',stroke:c,'stroke-width':k=='capped'?4:2.5},g);
    el('polyline',{points:xs.map(x=>X(x)+','+Y(100*d[x][k])).join(' '),fill:'none',stroke:'transparent','stroke-width':14},g);
    for(const x of xs){const p=el('circle',{cx:X(x),cy:Y(100*d[x][k]),r:5,fill:c},g);
      hover(p,`${S[k][0]}\n${x}% of letters correct: ${(100*d[x][k]).toFixed(0)}%`)}}}
const leg=document.getElementById('leg');
for(const [k,s] of Object.entries(S)){const e=document.createElement('span');e.dataset.k=k;
  e.innerHTML=`<i style="background:${k=='capped'?'currentColor':s[1]}"></i>${s[0]}`;link(e,k);leg.appendChild(e)}
</script>"""
AUDIT = STYLE + r"""<style>#ex{max-width:760px;border-top:1px solid rgba(128,128,128,.35);margin-top:6px;padding-top:8px}
#bar{display:flex;flex-wrap:wrap;align-items:center}#run{font-weight:600;flex-basis:100%;margin-bottom:6px}#pos{margin-right:8px;min-width:60px;text-align:center}
.msg{border-radius:10px;padding:8px 12px;margin:6px 0;line-height:1.4}.user{background:rgba(128,128,128,.16);margin-right:12%}
.bot{background:rgba(0,114,178,.13);margin-left:6%}.judge{background:rgba(230,159,0,.16)}
.who{font-size:11px;opacity:.65;margin-bottom:2px}.msg p{margin:0 0 6px}.msg h4{margin:8px 0 4px;font-size:13px}
.msg ul,.msg ol{margin:0 0 6px;padding-left:20px}.msg pre{white-space:pre-wrap;background:rgba(128,128,128,.18);padding:6px;border-radius:4px}
#sc{display:flex;gap:18px;flex-wrap:wrap;font-weight:600;margin:4px 0}#sc span b{font-size:16px}
</style><div id="tip"></div><div class="leg" id="leg"></div><svg id="c" viewBox="0 0 640 420"></svg>
<div id="ex"><div id="bar"><div id="run"></div>
<select id="sort"><option value="newgap">revised judge furthest from the reference first</option><option value="flip">original judge: misaligned, reference: not</option>
<option value="strong">the two reference models disagree most first</option><option value="low">most misaligned (reference) first</option><option value="random">random order</option></select>
<button id="prev">◀</button><span id="pos"></span><button id="next">▶</button></div><div id="sc"></div>
<div class="msg user"><div class="who">user</div><div id="q"></div></div>
<div class="msg bot"><div class="who">model answer</div><div id="a"></div></div>
<div class="msg judge"><div class="who">Claude Opus 5.5 (reference)</div><div id="ro"></div></div>
<div class="msg judge"><div class="who">GPT-6.1 Sol (reference)</div><div id="rs"></div></div></div><script>
const D=__DATA__;""" + AXES + MARKDOWN + r"""
const $=id=>document.getElementById(id),svg=$('c'),J={old:['original prompt','#D55E00'],new:['revised prompt','#0072B2']};
D.forEach((d,i)=>{d.i=i;d.ref=(d.opus+d.sol)/2});
let order=[],idx=0,dots={};
const [X,Y]=axes(svg,100,100,'reference alignment score: mean of two reasoning models','alignment score from the cheap judge');
el('line',{x1:X(0),y1:Y(0),x2:X(100),y2:Y(100),class:'ax'},svg);
for(const k of ['old','new'])for(const d of D){const c=el('circle',{cx:X(d.ref),cy:Y(d[k]),r:4,fill:J[k][1],'fill-opacity':.55,class:'hit'},svg);
  (dots[d.i]=dots[d.i]||[]).push(c);hover(c,`${J[k][0]}: ${d[k]}\nreference: ${d.ref}\nclick to read`);c.addEventListener('click',()=>{idx=Math.max(0,order.indexOf(d.i));show()})}
$('leg').innerHTML=Object.values(J).map(j=>`<span><i style="background:${j[1]}"></i>cheap judge, ${j[0]}</span>`).join('')+'<span>diagonal: perfect agreement</span><span><i style="border:2px solid currentColor"></i>shown below</span>';
function reorder(){const s=$('sort').value,keep=order[idx];order=D.map(d=>d.i);
  const key={newgap:d=>-Math.abs(d.new-d.ref),strong:d=>-Math.abs(d.opus-d.sol),low:d=>d.ref,random:d=>(d.i*7919)%997,flip:d=>-(d.ref-d.old)}[s];
  if(s=='flip')order=order.filter(i=>D[i].old<30&&D[i].ref>=30);
  order.sort((a,b)=>key(D[a])-key(D[b]));idx=Math.max(0,order.indexOf(keep));show()}
function show(){const d=D[order[idx]];$('pos').textContent=(idx+1)+' / '+order.length;$('run').textContent=d.task.replace(/^ia-/,'').replace(/_/g,' ')+' · '+d.run;
  $('sc').innerHTML=`<span>original prompt <b>${d.old}</b></span><span>revised prompt <b>${d.new}</b></span><span>Opus <b>${d.opus}</b></span><span>Sol <b>${d.sol}</b></span>`;
  $('q').innerHTML=md(d.q);$('a').innerHTML=md(d.a);$('ro').textContent=d.ropus;$('rs').textContent=d.rsol;
  svg.querySelectorAll('.cur').forEach(e=>{e.classList.remove('cur');e.setAttribute('r',4);e.removeAttribute('stroke')});
  for(const c of dots[d.i]){c.classList.add('cur');c.setAttribute('r',8);c.setAttribute('stroke','currentColor');c.setAttribute('stroke-width',2);svg.appendChild(c)}}
function step(k){const n=order.length;idx=(idx+k+n)%n;show()}
$('prev').onclick=()=>step(-1);$('next').onclick=()=>step(1);$('sort').onchange=reorder;
document.addEventListener('keydown',e=>{if(e.key=='ArrowLeft')step(-1);if(e.key=='ArrowRight')step(1)});reorder();</script>"""

rollouts = base64.b64encode((HERE / "pareto_rollouts.json.gz").read_bytes()).decode()
(HERE / "pareto.html").write_text(PARETO.replace("__DATA__", (HERE / "pareto_data.json").read_text().replace("</", "<\\/")).replace("__ROLLOUTS__", rollouts))
(HERE / "cipher_judge.html").write_text(CIPHER.replace("__DATA__", (HERE / "cipher_judge_data.json").read_text()))
(HERE / "judge_audit.html").write_text(AUDIT.replace("__DATA__", (HERE / "judge_audit_data.json").read_text().replace("</", "<\\/")))
