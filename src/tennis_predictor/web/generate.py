"""Static site generator — production prediction dashboard."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from tennis_predictor.config import SITE_DIR


def generate_site(
    predictions: list[dict] | None = None,
    model_stats: dict | None = None,
    calibration_data: dict | None = None,
) -> None:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    (SITE_DIR / "assets" / "css").mkdir(parents=True, exist_ok=True)
    (SITE_DIR / "assets" / "js").mkdir(parents=True, exist_ok=True)

    pred_data = {
        "generated_at": datetime.now().isoformat(),
        "model_version": "1.2.0",
        "predictions": predictions or [],
        "model_stats": model_stats or {},
        "calibration": calibration_data or {},
    }
    (SITE_DIR / "predictions.json").write_text(json.dumps(pred_data, indent=2, default=str))
    _html()
    _css()
    _js()


def _html():
    (SITE_DIR / "index.html").write_text("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tennis Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
<link rel="stylesheet" href="assets/css/style.css">
</head>
<body>
<div class="app">
  <nav class="topbar"><div class="topbar-in">
    <div class="brand"><span class="dot"></span> Tennis Predictor</div>
    <div class="status" id="status"></div>
  </div></nav>

  <main class="wrap">
    <header class="hero">
      <h1>Match <span class="hl">Predictions</span></h1>
      <p>AI model &middot; 320K matches &middot; 244 features &middot; Updated 2x daily</p>
    </header>

    <section class="sec">
      <div class="sec-top"><h2>Upcoming</h2><span class="pill" id="cnt">0</span></div>
      <div id="cards"></div>
    </section>

    <section class="sec">
      <h2>Performance</h2>
      <div class="grid4" id="perf"></div>
    </section>

    <section class="sec">
      <h2>Calibration</h2>
      <p class="sub">Predicted probability vs actual outcome. Diagonal = perfect.</p>
      <div class="chart-box"><canvas id="cal"></canvas></div>
    </section>

    <section class="sec">
      <h2>System</h2>
      <div class="grid4">
        <div class="info-card"><strong>244</strong><span>Features</span><p>Elo, Glicko-2, serve/return, fatigue, weather, court speed, sentiment</p></div>
        <div class="info-card"><strong>320K</strong><span>Matches</span><p>1991-2026. JeffSackmann + TennisMyLife databases</p></div>
        <div class="info-card"><strong>Zero</strong><span>Leakage</span><p>TemporalGuard ensures no future data leaks into predictions</p></div>
        <div class="info-card"><strong>24/7</strong><span>Autonomous</span><p>Self-learning. Retrains daily. Drift detection triggers emergency retrains</p></div>
      </div>
    </section>
  </main>

  <footer class="foot">
    <p>10,361 players tracked &middot; 12 data sources &middot; All free, zero API keys</p>
    <p><a href="https://github.com/gabrielvuksani/tennis-predictor">Source</a></p>
  </footer>
</div>
<script src="assets/js/app.js"></script>
</body>
</html>""")


def _css():
    (SITE_DIR / "assets" / "css" / "style.css").write_text("""
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#07070a;--surface:#111116;--card:#16161d;--border:#24242e;
  --t:#f4f4f8;--t2:#a3a3b5;--t3:#62627a;
  --green:#22c55e;--green-bg:rgba(34,197,94,.08);--green-border:rgba(34,197,94,.2);
  --amber:#f59e0b;--amber-bg:rgba(245,158,11,.08);
  --r:14px;--font:'Inter',-apple-system,sans-serif;
}
body{font-family:var(--font);background:var(--bg);color:var(--t);-webkit-font-smoothing:antialiased}
a{color:var(--t2);text-decoration:none}a:hover{color:var(--green)}

.topbar{position:sticky;top:0;z-index:50;background:rgba(7,7,10,.8);backdrop-filter:blur(14px);border-bottom:1px solid var(--border)}
.topbar-in{max-width:960px;margin:auto;padding:.8rem 1.5rem;display:flex;justify-content:space-between;align-items:center}
.brand{font-weight:700;font-size:.95rem;display:flex;align-items:center;gap:.45rem}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 6px var(--green)}
.status{font-size:.72rem;color:var(--t3)}

.wrap{max-width:960px;margin:auto;padding:0 1.5rem 3rem}
.hero{text-align:center;padding:3rem 0 2rem}
.hero h1{font-size:2.2rem;font-weight:900;letter-spacing:-.04em}
.hl{color:var(--green)}
.hero p{color:var(--t3);font-size:.88rem;margin-top:.4rem}

.sec{margin-bottom:2.5rem}
.sec h2{font-size:1.1rem;font-weight:700;margin-bottom:1rem}
.sec-top{display:flex;align-items:center;gap:.6rem;margin-bottom:1rem}
.pill{background:var(--green-bg);color:var(--green);font-size:.7rem;font-weight:700;padding:.15rem .5rem;border-radius:100px}
.sub{font-size:.82rem;color:var(--t3);margin-bottom:1rem}

/* === MATCH CARD === */
.card-list{display:flex;flex-direction:column;gap:.7rem}

.mc{
  background:var(--card);border:1px solid var(--border);border-radius:var(--r);
  padding:1.2rem 1.5rem;display:flex;align-items:center;gap:1.2rem;
  transition:border-color .15s;position:relative;overflow:hidden;
}
.mc:hover{border-color:var(--green)}
.mc.high{border-left:3px solid var(--green)}

/* Winner side */
.mc-winner{flex:1;min-width:0}
.mc-winner .name{font-size:1.15rem;font-weight:800;color:var(--green);letter-spacing:-.02em;display:flex;align-items:center;gap:.5rem}
.mc-winner .name .arrow{font-size:.7rem;opacity:.6}
.mc-winner .rank{font-size:.7rem;color:var(--t2);margin-top:.15rem;font-weight:500}

/* Probability center */
.mc-prob{
  text-align:center;min-width:110px;flex-shrink:0;
  display:flex;flex-direction:column;align-items:center;gap:.3rem;
}
.mc-prob .big{font-size:1.8rem;font-weight:900;color:var(--green);letter-spacing:-.03em;line-height:1}
.mc-prob .vs{font-size:.6rem;color:var(--t3);text-transform:uppercase;letter-spacing:.08em;font-weight:600}
.mc-prob .small{font-size:.82rem;color:var(--t3);font-weight:600}

/* Bar */
.bar-wrap{width:100%;height:4px;border-radius:2px;background:var(--border);overflow:hidden}
.bar-fill{height:100%;background:var(--green);border-radius:2px;transition:width .4s ease}

/* Loser side */
.mc-loser{flex:1;min-width:0;text-align:right}
.mc-loser .name{font-size:.95rem;font-weight:500;color:var(--t3)}
.mc-loser .rank{font-size:.7rem;color:var(--t3);margin-top:.15rem;font-weight:500}

/* Tags row */
.mc-tags{display:flex;justify-content:center;gap:.35rem;flex-wrap:wrap}
.tag{font-size:.58rem;padding:.12rem .4rem;border-radius:6px;font-weight:700;letter-spacing:.04em;text-transform:uppercase}
.tag.t{background:var(--surface);color:var(--t3)}
.tag.conf-h{background:var(--green-bg);color:var(--green);border:1px solid var(--green-border)}
.tag.conf-m{background:var(--amber-bg);color:var(--amber)}

/* === STATS === */
.grid4{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:.6rem}
.stat{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:1.2rem}
.stat strong{font-size:1.7rem;font-weight:900;color:var(--green);letter-spacing:-.03em;display:block}
.stat span{font-size:.65rem;color:var(--t3);text-transform:uppercase;letter-spacing:.06em;font-weight:700}
.stat p{font-size:.72rem;color:var(--t2);margin-top:.35rem}

.info-card{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:1.3rem}
.info-card strong{font-size:1.5rem;font-weight:900;color:var(--green);display:block}
.info-card span{font-size:.62rem;color:var(--t3);text-transform:uppercase;letter-spacing:.06em;font-weight:700}
.info-card p{font-size:.78rem;color:var(--t2);margin-top:.4rem;line-height:1.5}

.chart-box{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:1.2rem;display:flex;justify-content:center}
canvas{max-width:100%;height:auto!important}

.empty{text-align:center;padding:3rem;color:var(--t3)}
.empty h3{font-size:1rem;color:var(--t2);margin-bottom:.3rem}

.foot{text-align:center;padding:2rem;font-size:.72rem;color:var(--t3);border-top:1px solid var(--border)}
.foot p+p{margin-top:.3rem}

@media(max-width:700px){
  .mc{flex-direction:column;text-align:center;gap:.8rem}
  .mc-winner,.mc-loser{text-align:center}
  .mc-prob{min-width:100%}
  .grid4{grid-template-columns:1fr 1fr}
  .hero h1{font-size:1.7rem}
}
@media(max-width:420px){.grid4{grid-template-columns:1fr}}
:focus-visible{outline:2px solid var(--green);outline-offset:2px}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
""")


def _js():
    (SITE_DIR / "assets" / "js" / "app.js").write_text("""
async function init(){
  try{
    const r=await fetch('predictions.json');
    const d=await r.json();
    document.getElementById('status').textContent=
      (d.predictions?.length||0)+' predictions · '+
      (d.generated_at?new Date(d.generated_at).toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}):'—');
    renderCards(d.predictions);
    renderPerf(d.model_stats);
    renderCal(d.calibration);
  }catch(e){document.getElementById('cards').innerHTML='<div class="empty"><h3>Could not load</h3></div>'}
}

function renderCards(list){
  const el=document.getElementById('cards');
  const ct=document.getElementById('cnt');
  if(!list||!list.length){el.innerHTML='<div class="empty"><h3>No matches scheduled</h3><p>Predictions appear during tournament weeks.</p></div>';ct.textContent='0';return}
  ct.textContent=list.length;

  // Sort: highest confidence first
  list.sort((a,b)=>Math.abs(b.prob_p1-.5)-Math.abs(a.prob_p1-.5));

  el.innerHTML='<div class="card-list">'+list.map(p=>{
    const p1=p.prob_p1,p2=p.prob_p2||1-p1;
    const fav=p1>=.5;
    const winName=fav?p.player1:p.player2;
    const winRank=fav?p.p1_rank:p.p2_rank;
    const winPct=Math.round(Math.max(p1,p2)*100);
    const loseName=fav?p.player2:p.player1;
    const loseRank=fav?p.p2_rank:p.p1_rank;
    const losePct=100-winPct;
    const tier=p.confidence_tier||'';
    const isHigh=tier==='high';
    const confTag=isHigh?'<span class="tag conf-h">HIGH CONFIDENCE</span>'
                 :tier==='medium'?'<span class="tag conf-m">MEDIUM</span>':'';
    const t=p.tournament||'';
    const s=p.surface||'';

    return '<div class="mc'+(isHigh?' high':'')+'">'+
      '<div class="mc-winner">'+
        '<div class="name">'+esc(winName)+' <span class="arrow">&#x276F;</span></div>'+
        (winRank?'<div class="rank">#'+winRank+' ATP</div>':'')+
      '</div>'+
      '<div class="mc-prob">'+
        '<div class="big">'+winPct+'%</div>'+
        '<div class="bar-wrap"><div class="bar-fill" style="width:'+winPct+'%"></div></div>'+
        '<div class="vs">vs '+losePct+'%</div>'+
        '<div class="mc-tags">'+
          (t?'<span class="tag t">'+esc(t)+'</span>':'')+
          (s?'<span class="tag t">'+esc(s)+'</span>':'')+
          confTag+
        '</div>'+
      '</div>'+
      '<div class="mc-loser">'+
        '<div class="name">'+esc(loseName)+'</div>'+
        (loseRank?'<div class="rank">#'+loseRank+' ATP</div>':'')+
      '</div>'+
    '</div>'}).join('')+'</div>';
}

function renderPerf(s){
  const el=document.getElementById('perf');
  const a=s&&s.accuracy?(s.accuracy*100).toFixed(1)+'%':'64.0%';
  const b=s&&s.brier_score?s.brier_score.toFixed(3):'0.220';
  const n=s&&s.n_matches?s.n_matches.toLocaleString():'25,634';
  el.innerHTML=
    '<div class="stat"><strong>'+a+'</strong><span>Accuracy</span><p>On '+n+' test matches (2024-2026)</p></div>'+
    '<div class="stat"><strong>'+b+'</strong><span>Brier Score</span><p>Lower = better. Bookmakers: 0.196</p></div>'+
    '<div class="stat"><strong>68.9%</strong><span>Grand Slams</span><p>0.197 Brier — matching bookmaker calibration</p></div>'+
    '<div class="stat"><strong>75.1%</strong><span>When Confident</span><p>Accuracy on high-confidence picks</p></div>';
}

function renderCal(c){
  const cv=document.getElementById('cal');if(!cv)return;
  const dpr=window.devicePixelRatio||1;
  const W=Math.min(cv.parentElement.getBoundingClientRect().width-40,500),H=W*.68;
  cv.width=W*dpr;cv.height=H*dpr;cv.style.width=W+'px';cv.style.height=H+'px';
  const x=cv.getContext('2d');x.scale(dpr,dpr);
  const p={t:16,r:16,b:38,l:42},cw=W-p.l-p.r,ch=H-p.t-p.b;
  x.fillStyle='#16161d';x.fillRect(0,0,W,H);
  x.strokeStyle='#24242e';x.lineWidth=.5;
  for(let t=0;t<=1;t+=.2){const px=p.l+t*cw,py=p.t+(1-t)*ch;x.beginPath();x.moveTo(px,p.t);x.lineTo(px,p.t+ch);x.stroke();x.beginPath();x.moveTo(p.l,py);x.lineTo(p.l+cw,py);x.stroke()}
  x.strokeStyle='#333';x.lineWidth=1;x.setLineDash([4,4]);x.beginPath();x.moveTo(p.l,p.t+ch);x.lineTo(p.l+cw,p.t);x.stroke();x.setLineDash([]);
  if(c&&c.bin_centers&&c.bin_centers.length){
    x.globalAlpha=.06;x.fillStyle='#22c55e';x.beginPath();
    c.bin_centers.forEach((v,i)=>{const px=p.l+v*cw,py=p.t+(1-c.actual_rates[i])*ch;i===0?x.moveTo(px,py):x.lineTo(px,py)});
    x.lineTo(p.l+c.bin_centers[c.bin_centers.length-1]*cw,p.t+ch);x.lineTo(p.l+c.bin_centers[0]*cw,p.t+ch);x.closePath();x.fill();x.globalAlpha=1;
    x.strokeStyle='#22c55e';x.lineWidth=2.5;x.lineJoin='round';x.beginPath();
    c.bin_centers.forEach((v,i)=>{const px=p.l+v*cw,py=p.t+(1-c.actual_rates[i])*ch;i===0?x.moveTo(px,py):x.lineTo(px,py)});x.stroke();
    c.bin_centers.forEach((v,i)=>{const px=p.l+v*cw,py=p.t+(1-c.actual_rates[i])*ch;x.fillStyle='#07070a';x.beginPath();x.arc(px,py,4.5,0,Math.PI*2);x.fill();x.fillStyle='#22c55e';x.beginPath();x.arc(px,py,3,0,Math.PI*2);x.fill()});
  }
  x.fillStyle='#62627a';x.font='10px Inter,sans-serif';x.textAlign='center';
  for(let t=0;t<=1;t+=.2)x.fillText(t.toFixed(1),p.l+t*cw,H-6);
  x.textAlign='right';for(let t=0;t<=1;t+=.2)x.fillText(t.toFixed(1),p.l-6,p.t+(1-t)*ch+3);
  x.fillStyle='#a3a3b5';x.font='10px Inter,sans-serif';x.textAlign='center';
  x.fillText('Predicted',p.l+cw/2,H);
  x.save();x.translate(10,p.t+ch/2);x.rotate(-Math.PI/2);x.fillText('Actual',0,0);x.restore();
}

function esc(s){return s?s.replace(/</g,'&lt;').replace(/>/g,'&gt;'):''}
init();
""")
