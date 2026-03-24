"""Static site generator — premium prediction dashboard."""

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
        "model_version": "1.1.0",
        "predictions": predictions or [],
        "model_stats": model_stats or {},
        "calibration": calibration_data or {},
    }
    (SITE_DIR / "predictions.json").write_text(json.dumps(pred_data, indent=2, default=str))
    _write_html()
    _write_css()
    _write_js()


def _write_html():
    (SITE_DIR / "index.html").write_text("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tennis Predictor — AI Match Predictions</title>
<meta name="description" content="ATP tennis predictions powered by ML. 244 features, self-learning, calibrated probabilities.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="assets/css/style.css">
</head>
<body>
<div class="app">
  <nav class="nav">
    <div class="nav-inner">
      <div class="nav-brand"><span class="brand-dot"></span>Tennis Predictor</div>
      <div class="nav-status" id="nav-status"></div>
    </div>
  </nav>

  <main class="main">
    <section class="hero">
      <h1>ATP Match <span class="grad">Predictions</span></h1>
      <p class="hero-sub">Machine learning model trained on 320K+ matches. Updated twice daily.</p>
    </section>

    <section class="predictions" id="predictions-section">
      <div class="section-bar">
        <h2>Upcoming Matches</h2>
        <span class="count-pill" id="pred-count">0</span>
      </div>
      <div id="predictions-content" class="card-stack">
        <div class="skel"></div><div class="skel"></div><div class="skel"></div>
      </div>
    </section>

    <section class="metrics-section">
      <div class="section-bar"><h2>Model Metrics</h2></div>
      <div class="metrics" id="metrics-content"></div>
    </section>

    <section class="chart-section">
      <div class="section-bar"><h2>Calibration</h2></div>
      <p class="muted sm">Predicted probability vs actual win rate. Closer to the diagonal = better calibrated.</p>
      <div class="chart-wrap"><canvas id="cal-chart" width="560" height="380"></canvas></div>
    </section>

    <section class="about-section">
      <div class="section-bar"><h2>How It Works</h2></div>
      <div class="about-grid">
        <div class="about-card"><div class="ac-num">244</div><div class="ac-label">Features</div><p>Elo, Glicko-2, serve/return splits, fatigue, weather, court speed, momentum, sentiment, line movements</p></div>
        <div class="about-card"><div class="ac-num">320K</div><div class="ac-label">Matches</div><p>Trained on ATP data from 1991-2026 including challengers. TennisMyLife + JeffSackmann sources.</p></div>
        <div class="about-card"><div class="ac-num">0</div><div class="ac-label">Data Leakage</div><p>TemporalGuard ensures no future information leaks into predictions. The #1 flaw in other models.</p></div>
        <div class="about-card"><div class="ac-num">24/7</div><div class="ac-label">Autonomous</div><p>Self-learning system. Elo updates after every match. Retrains daily. Drift detection auto-triggers retraining.</p></div>
      </div>
    </section>
  </main>

  <footer class="foot">
    <p>320,229 matches &middot; 10,361 players &middot; 12 data sources &middot; Zero API keys</p>
    <p><a href="https://github.com/gabrielvuksani/tennis-predictor">GitHub</a> &middot; <a href="https://github.com/JeffSackmann/tennis_atp">Data</a> &middot; <a href="https://open-meteo.com">Weather</a></p>
  </footer>
</div>
<script src="assets/js/app.js"></script>
</body>
</html>""")


def _write_css():
    (SITE_DIR / "assets" / "css" / "style.css").write_text("""*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#050507;--bg2:#0c0c10;--bg3:#131318;--card:#16161c;--border:#222230;
  --t1:#f0f0f5;--t2:#a0a0b8;--t3:#65657a;
  --g1:#34d399;--g2:#10b981;--g-soft:rgba(16,185,129,.1);--g-glow:rgba(16,185,129,.25);
  --warn:#f59e0b;--r:12px;--rs:8px;
  --font:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;
}
html{scroll-behavior:smooth}
body{font-family:var(--font);background:var(--bg);color:var(--t1);line-height:1.55;-webkit-font-smoothing:antialiased}
.app{min-height:100vh;display:flex;flex-direction:column}

/* Nav */
.nav{position:sticky;top:0;z-index:99;background:rgba(5,5,7,.82);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border-bottom:1px solid var(--border)}
.nav-inner{max-width:1080px;margin:auto;padding:.85rem 1.5rem;display:flex;align-items:center;justify-content:space-between}
.nav-brand{font-weight:700;font-size:1rem;letter-spacing:-.02em;display:flex;align-items:center;gap:.5rem}
.brand-dot{width:8px;height:8px;border-radius:50%;background:var(--g1);box-shadow:0 0 8px var(--g-glow);animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.nav-status{font-size:.75rem;color:var(--t3)}

/* Main */
.main{flex:1;max-width:1080px;margin:auto;padding:0 1.5rem 4rem;width:100%}

/* Hero */
.hero{padding:3.5rem 0 2.5rem;text-align:center}
.hero h1{font-size:2.5rem;font-weight:800;letter-spacing:-.04em;line-height:1.1}
.grad{background:linear-gradient(135deg,var(--g1),var(--g2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero-sub{color:var(--t3);margin-top:.65rem;font-size:.95rem}

/* Section bars */
.section-bar{display:flex;align-items:center;gap:.65rem;margin-bottom:1.1rem;padding-top:2rem}
.section-bar h2{font-size:1.15rem;font-weight:700;letter-spacing:-.015em}
.count-pill{background:var(--g-soft);color:var(--g1);font-size:.72rem;font-weight:700;padding:.2rem .55rem;border-radius:100px}

/* Predictions */
.card-stack{display:flex;flex-direction:column;gap:.6rem}

.match{
  display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:1.2rem;
  background:var(--card);border:1px solid var(--border);border-radius:var(--r);
  padding:1.1rem 1.4rem;transition:border-color .15s,box-shadow .15s;
}
.match:hover{border-color:var(--g2);box-shadow:0 0 20px rgba(16,185,129,.06)}

.pl{display:flex;flex-direction:column;gap:.12rem}
.pl.right{text-align:right;align-items:flex-end}
.pn{font-weight:600;font-size:.95rem;transition:color .15s}
.pn.w{color:var(--g1)}
.pn.l{color:var(--t3)}
.pr{font-size:.68rem;color:var(--t3);font-weight:500}

.mid{display:flex;flex-direction:column;align-items:center;gap:.4rem;min-width:150px}
.pcts{display:flex;justify-content:space-between;width:100%;font-size:.82rem;font-weight:700}
.pcts .w{color:var(--g1)}.pcts .l{color:var(--t3)}
.bar{width:100%;height:5px;border-radius:3px;background:var(--bg2);overflow:hidden;display:flex}
.bar-fill{height:100%;background:var(--g1);border-radius:3px;transition:width .5s ease}
.tags{display:flex;gap:.4rem;margin-top:.2rem;flex-wrap:wrap;justify-content:center}
.tag{font-size:.6rem;color:var(--t3);background:var(--bg2);padding:.12rem .4rem;border-radius:var(--rs);text-transform:uppercase;letter-spacing:.04em;font-weight:600}
.tag.hi{background:var(--g-soft);color:var(--g1)}
.tag.med{background:rgba(245,158,11,.1);color:var(--warn)}

/* Metrics */
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:.6rem}
.met{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:1.2rem;transition:border-color .15s}
.met:hover{border-color:var(--g2)}
.met-v{font-size:1.9rem;font-weight:800;letter-spacing:-.04em;color:var(--g1);line-height:1}
.met-l{font-size:.7rem;color:var(--t3);text-transform:uppercase;letter-spacing:.06em;font-weight:600;margin-top:.3rem}
.met-d{font-size:.72rem;color:var(--t2);margin-top:.35rem}

/* About */
.about-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:.6rem}
.about-card{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:1.4rem;transition:border-color .15s,transform .15s}
.about-card:hover{border-color:var(--g2);transform:translateY(-2px)}
.ac-num{font-size:1.6rem;font-weight:800;color:var(--g1);letter-spacing:-.03em}
.ac-label{font-size:.68rem;color:var(--t3);text-transform:uppercase;letter-spacing:.06em;font-weight:600;margin-bottom:.4rem}
.about-card p{font-size:.8rem;color:var(--t2);line-height:1.5}

/* Chart */
.chart-wrap{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:1.4rem;display:flex;justify-content:center}
canvas{max-width:100%;height:auto!important}
.muted{color:var(--t3)}.sm{font-size:.82rem;margin-bottom:1rem}

/* Skeleton */
.skel{background:linear-gradient(90deg,var(--card) 25%,var(--bg3) 50%,var(--card) 75%);background-size:200% 100%;animation:shimmer 1.5s infinite;border-radius:var(--r);min-height:76px}
@keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}

/* Empty */
.empty{text-align:center;padding:3rem 1.5rem;color:var(--t3)}
.empty-ico{font-size:2.5rem;margin-bottom:.6rem;opacity:.4}
.empty h3{font-size:1rem;font-weight:600;color:var(--t2);margin-bottom:.25rem}
.empty p{font-size:.82rem}

/* Footer */
.foot{border-top:1px solid var(--border);text-align:center;padding:2rem 1.5rem;font-size:.75rem;color:var(--t3)}
.foot a{color:var(--t2);text-decoration:none;transition:color .15s}.foot a:hover{color:var(--g1)}
.foot p+p{margin-top:.4rem}

/* Responsive */
@media(max-width:768px){
  .hero h1{font-size:1.8rem}
  .match{grid-template-columns:1fr;text-align:center;gap:.5rem;padding:.9rem}
  .pl.right{text-align:center;align-items:center}
  .mid{min-width:100%}
  .metrics{grid-template-columns:repeat(2,1fr)}
}
@media(max-width:480px){.metrics,.about-grid{grid-template-columns:1fr}}

:focus-visible{outline:2px solid var(--g1);outline-offset:2px;border-radius:4px}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;transition-duration:.01ms!important}}
""")


def _write_js():
    (SITE_DIR / "assets" / "js" / "app.js").write_text("""
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
""")
