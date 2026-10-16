"""Standalone local HTML viewer; pixels are embedded only into the caller's output directory."""
import base64
import io
import json
from PIL import Image
import numpy as np


def png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


def create(sequences, report, path, propagation=None):
    """`sequences` maps a manifest name to its `Sequence`; `propagation` is a list of
    run.py propagation results whose seed geometry the viewer may overlay."""
    frames, masks = {}, {}
    for row in report['pairs']:
        if 'transform' not in row:
            continue
        sequence = sequences[row['source']]
        for frame in (row['previous'], row['current']):
            key = f"{row['source']}:{frame}"
            if key not in frames:
                frames[key] = png(sequence.image(frame))
                valid = sequence.mask(frame)
                pixels = np.zeros((*valid.shape, 4), np.uint8)
                pixels[~valid] = [216, 83, 179, 105]
                masks[key] = png(pixels)
    geometry = {}
    provenance = {}
    for result in propagation or []:
        for direction in result['directions'].values():
            for row in direction['frames']:
                if 'predicted_geometry' in row:
                    geometry[f"{result['sequence']}:{row['current']}"] = {
                        'walls': row['predicted_geometry']['walls'],
                        'sections': row['measured_sections']}
        provenance[result['sequence']] = result['provenance']
    if provenance:
        report['propagation_provenance'] = provenance
    payload = json.dumps({'report': report, 'frames': frames, 'masks': masks, 'geometry': geometry}, separators=(',', ':'), allow_nan=False).replace('</', '<\\/')
    source = PathTemplate.replace('__PAYLOAD__', payload)
    path.write_text(source)


PathTemplate = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Registration review</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#15181c;color:#e9edf1}
body{max-width:1240px;margin:30px auto;padding:0 24px}h1{font-size:27px;margin:0 0 6px;font-weight:600}p{color:#aeb9c7;line-height:1.5;margin:8px 0 20px}
.controls{display:flex;flex-wrap:wrap;gap:16px;align-items:center;padding:16px;background:#20262d;border:1px solid #39414b;border-radius:8px}
select,button{font:inherit;background:#2a323c;color:#edf3f9;border:1px solid #546171;border-radius:4px;padding:7px 10px}label{display:flex;align-items:center;gap:8px;font-size:14px}input[type=range]{accent-color:#7bb7dc;width:160px}.pair{flex:1}.pair input{width:100%;min-width:180px}
.panels{display:grid;grid-template-columns:2fr 1fr;gap:18px;margin-top:18px}section{background:#20262d;padding:16px;border:1px solid #39414b;border-radius:8px}h2{font-size:16px;font-weight:500;margin:0 0 12px}canvas{width:100%;background:#0e1115}#pairview{image-rendering:auto}.caption{font-size:13px;color:#aeb9c7;margin-top:8px}.metric{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid #39414b;padding:9px 0;font-size:14px}.value{font-variant-numeric:tabular-nums;color:#fff}.accepted{color:#84dab9}.failed{color:#efb976}#provenance{max-height:250px;overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.5 ui-monospace,monospace;color:#bccbd9}#timeline{height:38px;cursor:pointer}.wide{margin-top:18px}small{color:#aeb9c7}.legend{display:flex;gap:20px;font-size:13px;padding:10px 0}.cyan{color:#7bb7dc}.amber{color:#efb976}.green{color:#84dab9}@media(max-width:850px){.panels{grid-template-columns:1fr}}
</style>
<h1>Registration review</h1><p>Native image pairs, local motion and independent seed anchors. These are algorithm estimates; accepted registration and closure are not independent motion truth.</p>
<div class="controls">
<label>Sequence / group <select id="track"></select></label><label>Model <select id="model"><option value="similarity">Similarity</option><option value="translation">Translation comparison</option></select></label>
<button id="play">Play</button><label class="pair">Pair <input id="pair" type="range" min="0" value="0"><span id="frame"></span></label>
<label>Reference opacity <input id="alpha" type="range" min="0" max="100" value="45"></label><label><input id="mask" type="checkbox">Excluded pixels</label><label><input id="grid" type="checkbox" checked>Mapped grid</label><label><input id="geometry" type="checkbox">Translation seed segment</label>
</div>
<div class="panels"><section><h2 id="title">Aligned previous / current</h2><canvas id="pairview" width="960" height="291"></canvas><div class="caption">Current image below, transformed previous image above. A failed estimate is displayed as diagnostic only. Magenta shows excluded current pixels.</div>
<div class="legend"><span class="cyan">Blue: accepted map</span><span class="amber">Amber: diagnostic map / transported walls</span><span class="green">Green: measured seed walls</span></div><h2>Previous image / current image</h2><canvas id="native" width="960" height="146"></canvas></section>
<section><h2>Evidence and motion</h2><div id="metrics"></div><h2 style="margin-top:20px">Source provenance</h2><pre id="provenance"></pre><small>Covariance is a local approximation. Unavailable anchors, regional checks and reverse closure remain unavailable.</small></section></div>
<section class="wide"><h2>Pair outcomes — click to inspect</h2><canvas id="timeline" width="1180" height="38"></canvas><div class="caption" id="coverage"></div></section>
<section class="wide"><h2>Similarity seed-to-current centre trajectory</h2><canvas id="trajectory" width="1180" height="240"></canvas><div class="legend"><span class="cyan">Blue: chained adjacent estimates</span><span class="green">Green: independently anchored trajectory</span></div><div class="caption">A break starts a new supported segment. Seed anchors constrain only frames with sufficient overlap; failed anchors do not update the pose.</div></section>
<script>
const DATA=__PAYLOAD__;
const $=id=>document.getElementById(id), images={}, masks={}, rows=DATA.report.pairs.filter(r=>'transform' in r);
for(const [k,v] of Object.entries(DATA.frames)){const im=new Image();im.src=v;images[k]=im}
for(const [k,v] of Object.entries(DATA.masks)){const im=new Image();im.src=v;masks[k]=im}
const track=r=>`${r.source} · group ${r.group}`,tracks=[...new Set(rows.map(track))];for(const t of tracks){const o=document.createElement('option');o.value=t;o.textContent=t;$('track').append(o)}
$('provenance').textContent=JSON.stringify({registration:DATA.report.provenance,seed_geometry:DATA.report.propagation_provenance||'unavailable'},null,2);
let selected=[],timer=null;
function transform(p){const [x,y,a,s]=p,c=s*Math.cos(a),b=s*Math.sin(a);return [c,b,-b,c,x,y]}
function apply(p,x,y){const t=transform(p);return [t[0]*x+t[2]*y+t[4],t[1]*x+t[3]*y+t[5]]}
function metric(k,v,cls=''){const d=document.createElement('div');d.className='metric';const label=document.createElement('span');label.textContent=k;const value=document.createElement('span');value.className='value '+cls;value.textContent=v;d.append(label,value);$('metrics').append(d)}
function number(v,n=3){return v==null?'unavailable':Number(v).toFixed(n)}
function render(){if(!selected.length)return;const i=Number($('pair').value),r=selected[i],isTranslation=$('model').value==='translation',m=isTranslation?r.translation:r;
const p=isTranslation?(m.shift?[...m.shift,0,1]:null):r.transform,accepted=m.status==='accepted';
$('frame').textContent=`${r.previous} → ${r.current}`;$('title').textContent=`${isTranslation?'Translation':'Similarity'} · ${m.status.replaceAll('_',' ')}`;
const prev=images[`${r.source}:${r.previous}`],curr=images[`${r.source}:${r.current}`];if(!prev.complete||!curr.complete){requestAnimationFrame(render);return}
const w=curr.width,h=curr.height,H=Math.round(960*h/w),N=Math.round(480*h/w);if($('pairview').height!==H)$('pairview').height=H;if($('native').height!==N)$('native').height=N;const ctx=$('pairview').getContext('2d');ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,960,H);ctx.scale(960/w,H/h);ctx.drawImage(curr,0,0);
if(p){ctx.save();ctx.globalAlpha=Number($('alpha').value)/100;ctx.transform(...transform(p));ctx.drawImage(prev,0,0);ctx.restore()}
if($('mask').checked)ctx.drawImage(masks[`${r.source}:${r.current}`],0,0);
if(p&&$('grid').checked){ctx.strokeStyle=accepted?'#7bb7dc':'#efb976';ctx.lineWidth=.45;for(let x=20;x<w;x+=40){ctx.beginPath();for(let y=0;y<h;y+=4){const q=apply(p,x,y);if(y===0)ctx.moveTo(...q);else ctx.lineTo(...q)}ctx.stroke()}for(let y=16;y<h;y+=24){ctx.beginPath();for(let x=0;x<w;x+=4){const q=apply(p,x,y);if(x===0)ctx.moveTo(...q);else ctx.lineTo(...q)}ctx.stroke()}}
if($('geometry').checked){const g=DATA.geometry[`${r.source}:${r.current}`];if(g){ctx.strokeStyle='#efb976';ctx.lineWidth=.65;for(const wall of g.walls){ctx.beginPath();wall.points.forEach((p,i)=>i===0?ctx.moveTo(...p):ctx.lineTo(...p));ctx.stroke()}ctx.fillStyle='#84dab9';for(const s of g.sections){for(const side of ['left','right']){if(s[side].point){ctx.beginPath();ctx.arc(...s[side].point,1,0,Math.PI*2);ctx.fill()}}}}}
const nc=$('native').getContext('2d');nc.clearRect(0,0,960,N);nc.drawImage(prev,0,0,480,N);nc.drawImage(curr,480,0,480,N);
$('metrics').replaceChildren();metric('Outcome',m.status,accepted?'accepted':'failed');if(p){metric('Origin translation',`${number(p[0])}, ${number(p[1])} px`);metric('Angle',`${number(p[2]*180/Math.PI)}°`);metric('Uniform scale',number(p[3],5))}metric('Joint valid support',m.support);metric('Initial correlation',number(m.score));metric('Peak margin',number(m.margin));metric('Regional discrepancy',number(m.region_error)+' px');metric('Photometric RMS',number(m.residual_rms));metric('Reverse closure',number(m.closure_px)+' px');metric('Similarity sequence support',r.sequence_status||r.status);metric('Similarity seed anchor',r.anchor?.status||'unavailable');metric('Similarity anchor correction',number(r.anchor_correction_px)+' px');metric('Python estimate time',number(m.estimate_python_ms)+' ms');
const tc=$('timeline').getContext('2d');tc.clearRect(0,0,1180,38);selected.forEach((r,j)=>{const s=isTranslation?r.translation.status:r.status;tc.fillStyle=s==='accepted'?'#84dab9':'#efb976';tc.fillRect(j*1180/selected.length,4,Math.max(1,1180/selected.length-1),25)});tc.strokeStyle='#fff';tc.lineWidth=2;tc.strokeRect(i*1180/selected.length,1,Math.max(2,1180/selected.length),33);
const count=selected.filter(r=>(isTranslation?r.translation.status:r.status)==='accepted').length;$('coverage').textContent=`${count} / ${selected.length} accepted (${(100*count/selected.length).toFixed(1)}%). Rejected pairs remain visible.`;
trajectory(i,w,h);
}
function trajectory(index,w,h){const ctx=$('trajectory').getContext('2d');ctx.clearRect(0,0,1180,240);const points=selected.map((r,i)=>{if(!r.sequence)return null;const a=apply(r.sequence.chained_pose,(w-1)/2,(h-1)/2),b=apply(r.sequence.anchored_pose,(w-1)/2,(h-1)/2);return {a,b,i,seed:r.sequence.seed}});const xy=points.filter(Boolean).flatMap(p=>[p.a,p.b]);if(!xy.length)return;const xs=xy.map(p=>p[0]),ys=xy.map(p=>p[1]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),scale=Math.min(1100/Math.max(1,xmax-xmin),180/Math.max(1,ymax-ymin));const map=p=>[40+(p[0]-xmin)*scale,25+(p[1]-ymin)*scale];for(const [key,color] of [['a','#7bb7dc'],['b','#84dab9']]){ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.beginPath();let last=null;for(const p of points){if(!p){last=null;continue}const q=map(p[key]);if(last==null||last.seed!==p.seed)ctx.moveTo(...q);else ctx.lineTo(...q);last=p}ctx.stroke()}if(points[index]){const p=map(points[index].b);ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(...p,4,0,Math.PI*2);ctx.fill()}ctx.fillStyle='#aeb9c7';ctx.font='12px system-ui';ctx.fillText('Image x → · image y ↓ · separate segments have separate seed frames',40,226)}
function choose(){selected=rows.filter(r=>track(r)===$('track').value);$('pair').max=Math.max(0,selected.length-1);$('pair').value=Math.max(0,selected.findIndex(r=>r.status==='accepted'));render()}
for(const id of ['pair','alpha','mask','grid','geometry','model'])$(id).addEventListener('input',render);$('track').addEventListener('change',choose);$('timeline').addEventListener('click',e=>{const r=$('timeline').getBoundingClientRect();$('pair').value=Math.min(selected.length-1,Math.floor((e.clientX-r.left)/r.width*selected.length));render()});$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='Play'}else{timer=setInterval(()=>{$('pair').value=(Number($('pair').value)+1)%selected.length;render()},180);$('play').textContent='Pause'}};
$('geometry').disabled=Object.keys(DATA.geometry).length===0;
Promise.all(Object.values(images).map(im=>im.decode())).then(()=>{choose();const query=new URLSearchParams(location.search);if(query.has('track')){$('track').value=query.get('track');choose()}if(query.has('frame')){$('pair').value=Math.max(0,selected.findIndex(r=>r.current===Number(query.get('frame'))))}if(query.has('model'))$('model').value=query.get('model');if(query.has('geometry')&&!$('geometry').disabled)$('geometry').checked=true;render()});
</script></html>'''
