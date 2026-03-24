
async function init(){
  try{
    const r=await fetch('predictions.json');const d=await r.json();
    nav(d);preds(d.predictions);stats(d.model_stats);cal(d.calibration);
  }catch(e){document.getElementById('predictions-content').innerHTML=empty('Error','Could not load predictions.')}
}

function nav(d){
  const el=document.getElementById('nav-status');
  const dt=d.generated_at?new Date(d.generated_at):null;
  const t=dt?dt.toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}):'—';
  el.textContent=`${d.predictions?.length||0} predictions · ${t}`;
}

function preds(list){
  const el=document.getElementById('predictions-content');
  const ct=document.getElementById('pred-count');
  if(!list||!list.length){el.innerHTML=empty('No matches scheduled','Check back during tournament weeks.');ct.textContent='0';return}
  ct.textContent=list.length;
  el.innerHTML=list.map(p=>{
    const a=Math.round(p.prob_p1*100),b=100-a,f=a>=50;
    const tier=p.confidence_tier||'';
    const tBadge=tier==='high'?'<span class="tag hi">HIGH CONF</span>':tier==='medium'?'<span class="tag med">MEDIUM</span>':'';
    const t=p.tournament||'',s=p.surface||'';
    return `<div class="match">
      <div class="pl"><span class="pn ${f?'w':'l'}">${p.player1}</span><span class="pr">${p.p1_rank?'#'+p.p1_rank+' ATP':''}</span></div>
      <div class="mid">
        <div class="pcts"><span class="${f?'w':'l'}">${a}%</span><span class="${f?'l':'w'}">${b}%</span></div>
        <div class="bar"><div class="bar-fill" style="width:${a}%"></div></div>
        <div class="tags">${t?`<span class="tag">${t}</span>`:''}${s?`<span class="tag">${s}</span>`:''}${tBadge}</div>
      </div>
      <div class="pl right"><span class="pn ${f?'l':'w'}">${p.player2}</span><span class="pr">${p.p2_rank?'#'+p.p2_rank+' ATP':''}</span></div>
    </div>`}).join('');
}

function stats(s){
  const el=document.getElementById('metrics-content');
  if(!s||!Object.keys(s).length){
    el.innerHTML=`
      <div class="met"><div class="met-v">64.0%</div><div class="met-l">Accuracy</div><div class="met-d">On 25,634 future matches (2024-2026)</div></div>
      <div class="met"><div class="met-v">0.220</div><div class="met-l">Brier Score</div><div class="met-d">Lower = better. Bookmakers: 0.196</div></div>
      <div class="met"><div class="met-v">68.9%</div><div class="met-l">Grand Slams</div><div class="met-d">0.197 Brier — matching bookmaker calibration</div></div>
      <div class="met"><div class="met-v">75.1%</div><div class="met-l">When Confident</div><div class="met-d">Accuracy on high-confidence predictions (&gt;70%)</div></div>`;
    return}
  const a=s.accuracy?(s.accuracy*100).toFixed(1)+'%':'—';
  const b=s.brier_score?s.brier_score.toFixed(3):'—';
  const e=s.ece?s.ece.toFixed(3):'—';
  const n=s.n_matches?s.n_matches.toLocaleString():'—';
  el.innerHTML=`
    <div class="met"><div class="met-v">${a}</div><div class="met-l">Accuracy</div><div class="met-d">On ${n} test matches</div></div>
    <div class="met"><div class="met-v">${b}</div><div class="met-l">Brier Score</div><div class="met-d">Lower = better. Bookmakers: 0.196</div></div>
    <div class="met"><div class="met-v">${e}</div><div class="met-l">Calibration Error</div><div class="met-d">0 = perfectly calibrated</div></div>
    <div class="met"><div class="met-v">244</div><div class="met-l">Features</div><div class="met-d">Elo, Glicko-2, serve, fatigue, weather...</div></div>`;
}

function cal(c){
  const cv=document.getElementById('cal-chart');if(!cv)return;
  const dpr=window.devicePixelRatio||1;
  const W=Math.min(cv.parentElement.getBoundingClientRect().width-48,520),H=W*.68;
  cv.width=W*dpr;cv.height=H*dpr;cv.style.width=W+'px';cv.style.height=H+'px';
  const x=cv.getContext('2d');x.scale(dpr,dpr);
  const p={t:20,r:20,b:42,l:46},cw=W-p.l-p.r,ch=H-p.t-p.b;
  x.fillStyle='#16161c';x.fillRect(0,0,W,H);
  // Grid
  x.strokeStyle='#222230';x.lineWidth=.5;
  for(let t=0;t<=1;t+=.2){const px=p.l+t*cw,py=p.t+(1-t)*ch;x.beginPath();x.moveTo(px,p.t);x.lineTo(px,p.t+ch);x.stroke();x.beginPath();x.moveTo(p.l,py);x.lineTo(p.l+cw,py);x.stroke()}
  // Diagonal
  x.strokeStyle='#333340';x.lineWidth=1;x.setLineDash([4,4]);x.beginPath();x.moveTo(p.l,p.t+ch);x.lineTo(p.l+cw,p.t);x.stroke();x.setLineDash([]);
  if(c&&c.bin_centers&&c.bin_centers.length){
    // Area
    x.globalAlpha=.07;x.fillStyle='#10b981';x.beginPath();
    c.bin_centers.forEach((v,i)=>{const px=p.l+v*cw,py=p.t+(1-c.actual_rates[i])*ch;i===0?x.moveTo(px,py):x.lineTo(px,py)});
    x.lineTo(p.l+c.bin_centers[c.bin_centers.length-1]*cw,p.t+ch);x.lineTo(p.l+c.bin_centers[0]*cw,p.t+ch);x.closePath();x.fill();x.globalAlpha=1;
    // Line
    x.strokeStyle='#10b981';x.lineWidth=2.5;x.lineJoin='round';x.beginPath();
    c.bin_centers.forEach((v,i)=>{const px=p.l+v*cw,py=p.t+(1-c.actual_rates[i])*ch;i===0?x.moveTo(px,py):x.lineTo(px,py)});x.stroke();
    // Dots
    c.bin_centers.forEach((v,i)=>{const px=p.l+v*cw,py=p.t+(1-c.actual_rates[i])*ch;x.fillStyle='#050507';x.beginPath();x.arc(px,py,5,0,Math.PI*2);x.fill();x.fillStyle='#10b981';x.beginPath();x.arc(px,py,3.5,0,Math.PI*2);x.fill()});
  }
  // Labels
  x.fillStyle='#65657a';x.font='11px Inter,sans-serif';x.textAlign='center';
  for(let t=0;t<=1;t+=.2)x.fillText(t.toFixed(1),p.l+t*cw,H-6);
  x.textAlign='right';for(let t=0;t<=1;t+=.2)x.fillText(t.toFixed(1),p.l-7,p.t+(1-t)*ch+4);
  x.fillStyle='#a0a0b8';x.font='11px Inter,sans-serif';x.textAlign='center';
  x.fillText('Predicted Probability',p.l+cw/2,H-1);
  x.save();x.translate(12,p.t+ch/2);x.rotate(-Math.PI/2);x.fillText('Actual Win Rate',0,0);x.restore();
}

function empty(t,d){return `<div class="empty"><div class="empty-ico">&#127934;</div><h3>${t}</h3><p>${d}</p></div>`}
init();
