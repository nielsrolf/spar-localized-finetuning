"""Build aux.html (auxiliary-dataset reader) from aux_data.json (written by aux_data.py)."""
from pathlib import Path

HERE = Path(__file__).parent
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
PAGE = r"""<!doctype html><meta charset="utf-8"><style>
body{font:13px system-ui,sans-serif;margin:8px;color:#222;background:#fff}
@media(prefers-color-scheme:dark){body{color:#ddd;background:#1e1e1e}}
select,button{font:inherit;margin:0 8px 6px 0}table{border-collapse:collapse;margin:0 0 10px}
td,th{padding:3px 12px 3px 0;text-align:left;vertical-align:top}th{opacity:.65;font-weight:500}
tr.row{cursor:pointer}tr.row:hover{background:rgba(128,128,128,.12)}tr.sel{background:rgba(74,144,226,.18)}
.msg{border-radius:10px;padding:8px 12px;margin:6px 0;line-height:1.4;overflow:auto}.user{background:rgba(128,128,128,.16);margin-right:12%}
.assistant{background:rgba(74,144,226,.16);margin-left:12%}.plain{background:rgba(120,180,120,.16);margin-left:12%}
.who{font-size:11px;opacity:.65;margin-bottom:2px}.msg p{margin:0 0 6px}.msg h4{margin:8px 0 4px;font-size:13px}
.msg ul,.msg ol{margin:0 0 6px;padding-left:20px}.msg pre{white-space:pre-wrap;background:rgba(128,128,128,.18);padding:6px;border-radius:4px}
.msg code{background:rgba(128,128,128,.18);padding:0 3px;border-radius:3px}#desc{margin:4px 0 8px;max-width:760px}
#pos{opacity:.7;margin-right:8px}.na{opacity:.5}
</style>
<table id="tbl"></table><div id="desc"></div>
<div id="nav"><select id="view"></select><button id="prev">◀</button><span id="pos"></span><button id="next">▶</button></div>
<div id="chat"></div>
<script>
const DATA=__DATA__;
__MD__
let d=DATA.findIndex(x=>x.n),v=null,k=0;
const $=id=>document.getElementById(id);
function table(){const t=$('tbl');t.innerHTML='<tr><th>Dataset</th><th>Rows</th><th>Median answer length (words)</th><th>Used by</th><th>For task</th></tr>';
  DATA.forEach((x,i)=>{const tr=document.createElement('tr');tr.className='row'+(i==d?' sel':'')+(x.n?'':' na');
    [x.name,x.n?x.n.toLocaleString():'not built',x.median_words??'–',x.technique,x.task].forEach(c=>{const td=document.createElement('td');td.textContent=c;tr.appendChild(td)});
    if(x.n)tr.onclick=()=>{d=i;v=null;k=0;render()};t.appendChild(tr)})}
function render(){table();const x=DATA[d];if(!x)return;
  $('desc').textContent=x.description+(x.mean_score!==undefined?' Mean judge score of kept rows: '+x.mean_score.toFixed(0)+'.':'');
  const views=Object.keys(x.views);if(!v||!x.views[v])v=views[0];
  $('view').innerHTML='';views.forEach(n=>{const o=document.createElement('option');o.textContent=n;o.selected=n==v;$('view').appendChild(o)});
  const rows=x.views[v];k=Math.max(0,Math.min(k,rows.length-1));const r=rows[k];
  $('pos').textContent=(k+1)+' / '+rows.length+'  (row '+r.i+(r.score!==undefined?', '+r.score_name+' '+r.score:'')+(r.tag?', '+r.tag:'')+')';
  const chat=$('chat');chat.innerHTML='';
  const add=(cls,who,text,raw)=>{const m=document.createElement('div');m.className='msg '+cls;
    m.innerHTML='<div class="who">'+who+'</div>'+(raw?'<pre></pre>':md(text));if(raw)m.querySelector('pre').textContent=text;chat.appendChild(m)};
  r.messages.forEach(m=>add(m.role,m.role,m.content,!!r.plain&&m.role=='assistant'));
  if(r.plain)add('plain','decoded (plain text before encoding)',r.plain)}
$('view').onchange=e=>{v=e.target.value;k=0;render()};
$('prev').onclick=()=>{k--;render()};$('next').onclick=()=>{k++;render()};
document.addEventListener('keydown',e=>{if(e.key=='ArrowLeft'){k--;render()}if(e.key=='ArrowRight'){k++;render()}});
render();
</script>"""

data = (HERE / "aux_data.json").read_text().replace("</", "<\\/")
(HERE / "aux.html").write_text(PAGE.replace("__MD__", MARKDOWN).replace("__DATA__", data))
print("wrote aux.html")
