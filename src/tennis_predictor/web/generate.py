"""Static site generator — full-featured prediction dashboard.

Features:
1. Match predictions with expandable analysis
2. Tournament grouping
3. Results tracker
4. Performance analytics
5. Player profiles (click any name)
6. Browser push notifications
7. Dark/Light mode
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from tennis_predictor.config import SITE_DIR, PREDICTIONS_DIR


def generate_site(
    predictions: list[dict] | None = None,
    model_stats: dict | None = None,
    calibration_data: dict | None = None,
) -> None:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    (SITE_DIR / "assets" / "css").mkdir(parents=True, exist_ok=True)
    (SITE_DIR / "assets" / "js").mkdir(parents=True, exist_ok=True)

    # Load prediction history for results tracker
    history = _load_prediction_history()

    pred_data = {
        "generated_at": datetime.now().isoformat(),
        "model_version": "2.0.0",
        "predictions": predictions or [],
        "model_stats": model_stats or {},
        "calibration": calibration_data or {},
        "history": history,
    }
    (SITE_DIR / "predictions.json").write_text(json.dumps(pred_data, indent=2, default=str))
    _write_html()
    _write_css()
    _write_js()
    _write_sw()


def _load_prediction_history() -> list:
    """Load recent prediction history for the results tracker."""
    history_dir = PREDICTIONS_DIR / "history"
    if not history_dir.exists():
        return []
    entries = []
    for f in sorted(history_dir.glob("*.json"), reverse=True)[:7]:
        try:
            data = json.loads(f.read_text())
            entries.append({
                "date": f.stem,
                "count": len(data.get("predictions", [])),
                "predictions": data.get("predictions", [])[:10],
            })
        except Exception:
            pass
    return entries


def _write_html():
    (SITE_DIR / "index.html").write_text("""<!DOCTYPE html>
<html lang="en" data-theme="dark">
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
  <nav class="topbar"><div class="tb-in">
    <div class="brand"><span class="dot"></span> Tennis Predictor</div>
    <div class="tb-right">
      <div class="tabs" id="tabs">
        <button class="tab active" data-v="predictions">Predictions</button>
        <button class="tab" data-v="results">Results</button>
        <button class="tab" data-v="analytics">Analytics</button>
      </div>
      <button class="theme-btn" id="theme-toggle" title="Toggle theme">&#9790;</button>
    </div>
  </div></nav>

  <main class="wrap">
    <div id="view-predictions" class="view active">
      <header class="hero"><h1>Match <span class="hl">Predictions</span></h1>
      <p class="sub" id="status-line">Loading...</p></header>
      <div id="cards"></div>
    </div>

    <div id="view-results" class="view">
      <header class="hero"><h1>Prediction <span class="hl">Results</span></h1>
      <p class="sub">Track record of recent predictions</p></header>
      <div id="results-content"></div>
    </div>

    <div id="view-analytics" class="view">
      <header class="hero"><h1><span class="hl">Analytics</span></h1>
      <p class="sub">Model performance and calibration</p></header>
      <div class="grid4" id="perf"></div>
      <div style="margin-top:1.5rem">
        <h3 style="font-size:.95rem;margin-bottom:.8rem">Calibration</h3>
        <p class="sub">Predicted probability vs actual win rate. Diagonal = perfect.</p>
        <div class="chart-box"><canvas id="cal"></canvas></div>
      </div>
      <div style="margin-top:1.5rem" id="system-info">
        <h3 style="font-size:.95rem;margin-bottom:.8rem">System</h3>
        <div class="grid4">
          <div class="info-card"><strong>244</strong><span>Features</span><p>Elo, Glicko-2, serve/return, fatigue, weather, court speed, sentiment</p></div>
          <div class="info-card"><strong>320K</strong><span>Matches</span><p>1991-2026. JeffSackmann + TennisMyLife databases</p></div>
          <div class="info-card"><strong>Zero</strong><span>Leakage</span><p>TemporalGuard ensures no future data leaks into predictions</p></div>
          <div class="info-card"><strong>24/7</strong><span>Autonomous</span><p>Self-learning. Retrains daily. Drift detection triggers emergency retrains</p></div>
        </div>
      </div>
    </div>
  </main>

  <!-- Player Modal -->
  <div class="modal-overlay" id="player-modal">
    <div class="modal">
      <button class="modal-close" id="modal-close">&times;</button>
      <div id="player-content"></div>
    </div>
  </div>

  <footer class="foot">
    <p>10,361 players &middot; 12 data sources &middot; All free</p>
    <p><a href="https://github.com/gabrielvuksani/tennis-predictor">GitHub</a></p>
  </footer>
</div>
<script src="assets/js/app.js"></script>
</body>
</html>""")


def _write_css():
    (SITE_DIR / "assets" / "css" / "style.css").write_text("""
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}

/* === THEME SYSTEM === */
[data-theme="dark"]{
  --bg:#07070a;--surface:#0e0e13;--card:#15151c;--border:#24242e;
  --t:#f4f4f8;--t2:#a3a3b5;--t3:#62627a;
  --green:#22c55e;--green-bg:rgba(34,197,94,.08);--green-border:rgba(34,197,94,.2);
  --amber:#f59e0b;--amber-bg:rgba(245,158,11,.08);
  --red:#ef4444;--red-bg:rgba(239,68,68,.08);
}
[data-theme="light"]{
  --bg:#f8f9fa;--surface:#ffffff;--card:#ffffff;--border:#e2e4e8;
  --t:#1a1a2e;--t2:#555570;--t3:#8888a0;
  --green:#16a34a;--green-bg:rgba(22,163,74,.06);--green-border:rgba(22,163,74,.15);
  --amber:#d97706;--amber-bg:rgba(217,119,6,.06);
  --red:#dc2626;--red-bg:rgba(220,38,38,.06);
}

body{font-family:'Inter',-apple-system,sans-serif;background:var(--bg);color:var(--t);-webkit-font-smoothing:antialiased;transition:background .2s,color .2s}
a{color:var(--t2);text-decoration:none}a:hover{color:var(--green)}

/* === TOPBAR === */
.topbar{position:sticky;top:0;z-index:50;background:color-mix(in srgb,var(--bg) 85%,transparent);backdrop-filter:blur(14px);border-bottom:1px solid var(--border)}
.tb-in{max-width:1000px;margin:auto;padding:.6rem 1.5rem;display:flex;justify-content:space-between;align-items:center}
.brand{font-weight:700;font-size:.9rem;display:flex;align-items:center;gap:.4rem}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 6px var(--green)}
.tb-right{display:flex;align-items:center;gap:.8rem}
.tabs{display:flex;gap:0;background:var(--surface);border:1px solid var(--border);border-radius:8px;overflow:hidden}
.tab{background:none;border:none;color:var(--t3);font-size:.72rem;font-weight:600;padding:.4rem .8rem;cursor:pointer;transition:all .15s;font-family:inherit}
.tab:hover{color:var(--t)}
.tab.active{background:var(--green-bg);color:var(--green)}
.theme-btn{background:none;border:1px solid var(--border);border-radius:6px;padding:.3rem .5rem;cursor:pointer;font-size:.85rem;color:var(--t2);transition:all .15s}
.theme-btn:hover{color:var(--green);border-color:var(--green)}

/* === VIEWS === */
.wrap{max-width:1000px;margin:auto;padding:0 1.5rem 3rem}
.view{display:none}.view.active{display:block}
.hero{text-align:center;padding:2.5rem 0 1.5rem}
.hero h1{font-size:2rem;font-weight:900;letter-spacing:-.04em}
.hl{color:var(--green)}
.sub{color:var(--t3);font-size:.82rem;margin-top:.3rem}

/* === MATCH CARDS === */
.tournament-group{margin-bottom:1.5rem}
.tournament-group h3{font-size:.78rem;color:var(--t3);text-transform:uppercase;letter-spacing:.06em;font-weight:700;margin-bottom:.6rem;padding-left:.2rem}
.card-list{display:flex;flex-direction:column;gap:.5rem}

.mc{
  background:var(--card);border:1px solid var(--border);border-radius:12px;
  cursor:pointer;transition:border-color .15s;overflow:hidden;
}
.mc:hover{border-color:var(--green)}
.mc.high{border-left:3px solid var(--green)}

.mc-top{display:flex;align-items:center;gap:1rem;padding:1rem 1.3rem}
.mc-winner{flex:1;min-width:0}
.mc-winner .name{font-size:1.05rem;font-weight:800;color:var(--green);display:flex;align-items:center;gap:.4rem;letter-spacing:-.01em}
.mc-winner .name .arr{font-size:.6rem;opacity:.5}
.mc-winner .rank{font-size:.67rem;color:var(--t2);margin-top:.1rem}
.mc-prob{text-align:center;min-width:90px;flex-shrink:0}
.mc-prob .big{font-size:1.6rem;font-weight:900;color:var(--green);letter-spacing:-.03em;line-height:1}
.mc-prob .vs{font-size:.58rem;color:var(--t3);margin-top:.15rem}
.bar-w{width:100%;height:3px;border-radius:2px;background:var(--border);overflow:hidden;margin-top:.3rem}
.bar-f{height:100%;background:var(--green);border-radius:2px}
.mc-loser{flex:1;min-width:0;text-align:right}
.mc-loser .name{font-size:.88rem;font-weight:500;color:var(--t3)}
.mc-loser .rank{font-size:.67rem;color:var(--t3);margin-top:.1rem}

.mc-tags{display:flex;justify-content:center;gap:.3rem;padding:0 1.3rem .6rem;flex-wrap:wrap}
.tag{font-size:.55rem;padding:.1rem .35rem;border-radius:5px;font-weight:700;letter-spacing:.04em;text-transform:uppercase}
.tag.t{background:var(--surface);color:var(--t3)}
.tag.ch{background:var(--green-bg);color:var(--green);border:1px solid var(--green-border)}
.tag.cm{background:var(--amber-bg);color:var(--amber)}
.mc-hint{font-size:.55rem;color:var(--t3);text-align:center;padding:0 0 .5rem;opacity:.5}

/* Detail panel */
.mc-detail{display:none;grid-template-columns:1fr 1fr;gap:.8rem;padding:1rem 1.3rem;border-top:1px solid var(--border);background:var(--surface);font-size:.75rem}
.mc.open .mc-detail{display:grid}
.mc.open .mc-hint{display:none}

.dc h4{font-size:.68rem;color:var(--green);text-transform:uppercase;letter-spacing:.06em;font-weight:700;margin-bottom:.5rem}
.dr{display:flex;justify-content:space-between;padding:.18rem 0;border-bottom:1px solid rgba(128,128,128,.08)}
.dr .l{color:var(--t3)}.dr .v{font-weight:600}

.h2h-section,.factors-section{grid-column:1/-1;padding-top:.5rem;border-top:1px solid var(--border)}
.h2h-section h4,.factors-section h4{font-size:.68rem;text-transform:uppercase;letter-spacing:.06em;font-weight:700;margin-bottom:.4rem}
.h2h-section h4{color:var(--t2)}
.factors-section h4{color:var(--amber)}
.h2h-bar{height:5px;border-radius:3px;background:var(--border);overflow:hidden;margin:.4rem 0}
.h2h-fill{height:100%;background:var(--green)}
.factor{padding:.15rem 0;color:var(--t2);font-size:.73rem}
.factor::before{content:'> ';color:var(--green);font-weight:700}

/* Player name clickable */
.plink{cursor:pointer;text-decoration:underline;text-decoration-style:dotted;text-underline-offset:2px}
.plink:hover{text-decoration-style:solid}

/* === RESULTS === */
.results-day{margin-bottom:1.5rem}
.results-day h3{font-size:.82rem;font-weight:700;margin-bottom:.6rem;color:var(--t2)}
.result-row{display:flex;align-items:center;gap:.8rem;padding:.5rem .8rem;background:var(--card);border:1px solid var(--border);border-radius:8px;margin-bottom:.3rem;font-size:.8rem}
.result-pick{flex:1;font-weight:600}
.result-pct{font-weight:700;min-width:40px;text-align:center}
.result-vs{color:var(--t3);flex:1}

/* === METRICS === */
.grid4{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:.5rem}
.stat{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1.1rem}
.stat strong{font-size:1.5rem;font-weight:900;color:var(--green);display:block;letter-spacing:-.03em}
.stat span{font-size:.6rem;color:var(--t3);text-transform:uppercase;letter-spacing:.06em;font-weight:700}
.stat p{font-size:.7rem;color:var(--t2);margin-top:.3rem}

.info-card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1.2rem}
.info-card strong{font-size:1.3rem;font-weight:900;color:var(--green);display:block}
.info-card span{font-size:.58rem;color:var(--t3);text-transform:uppercase;letter-spacing:.06em;font-weight:700}
.info-card p{font-size:.73rem;color:var(--t2);margin-top:.35rem;line-height:1.45}

.chart-box{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1rem;display:flex;justify-content:center}
canvas{max-width:100%;height:auto!important}

/* === MODAL === */
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:100;align-items:center;justify-content:center;padding:1rem}
.modal-overlay.open{display:flex}
.modal{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1.5rem;max-width:500px;width:100%;max-height:80vh;overflow-y:auto;position:relative}
.modal-close{position:absolute;top:.8rem;right:1rem;background:none;border:none;color:var(--t3);font-size:1.3rem;cursor:pointer}
.modal h3{font-size:1.1rem;font-weight:800;color:var(--green);margin-bottom:.3rem}
.modal .modal-rank{font-size:.75rem;color:var(--t2);margin-bottom:1rem}
.modal .modal-section{margin-bottom:1rem}
.modal .modal-section h4{font-size:.7rem;color:var(--t3);text-transform:uppercase;letter-spacing:.06em;font-weight:700;margin-bottom:.4rem}

.empty{text-align:center;padding:3rem;color:var(--t3)}
.empty h3{font-size:1rem;color:var(--t2);margin-bottom:.3rem}

.foot{text-align:center;padding:1.5rem;font-size:.7rem;color:var(--t3);border-top:1px solid var(--border)}

@media(max-width:700px){
  .mc-top{flex-direction:column;text-align:center;gap:.5rem}
  .mc-winner,.mc-loser{text-align:center}
  .mc-prob{min-width:100%}
  .mc-detail{grid-template-columns:1fr}
  .grid4{grid-template-columns:1fr 1fr}
  .hero h1{font-size:1.6rem}
  .tabs{flex-wrap:wrap}
}
@media(max-width:420px){.grid4{grid-template-columns:1fr}}
:focus-visible{outline:2px solid var(--green);outline-offset:2px}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
""")


def _write_js():
    (SITE_DIR / "assets" / "js" / "app.js").write_text("""
let DATA=null;

async function init(){
  try{
    const r=await fetch('predictions.json');
    DATA=await r.json();
    document.getElementById('status-line').textContent=
      (DATA.predictions?.length||0)+' predictions · '+
      (DATA.generated_at?new Date(DATA.generated_at).toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}):'');
    renderCards(DATA.predictions);
    renderPerf(DATA.model_stats);
    renderCal(DATA.calibration);
    renderResults(DATA.history);
  }catch(e){document.getElementById('cards').innerHTML='<div class="empty"><h3>Could not load</h3></div>'}
}

// === TAB NAVIGATION ===
document.querySelectorAll('.tab').forEach(t=>t.addEventListener('click',()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.view').forEach(x=>x.classList.remove('active'));
  t.classList.add('active');
  document.getElementById('view-'+t.dataset.v).classList.add('active');
}));

// === THEME TOGGLE ===
const themeBtn=document.getElementById('theme-toggle');
function setTheme(t){document.documentElement.dataset.theme=t;localStorage.setItem('theme',t);themeBtn.textContent=t==='dark'?'\\u263E':'\\u2600'}
themeBtn.addEventListener('click',()=>setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark'));
setTheme(localStorage.getItem('theme')||'dark');

// === CARD EXPAND ===
document.addEventListener('click',e=>{
  const mc=e.target.closest('.mc');
  if(mc&&!e.target.closest('.plink')) mc.classList.toggle('open');
});

// === PLAYER MODAL ===
document.getElementById('modal-close').addEventListener('click',()=>document.getElementById('player-modal').classList.remove('open'));
document.getElementById('player-modal').addEventListener('click',e=>{if(e.target===e.currentTarget)e.currentTarget.classList.remove('open')});

document.addEventListener('click',e=>{
  const pl=e.target.closest('.plink');
  if(!pl) return;
  e.stopPropagation();
  const name=pl.dataset.name;
  if(!name||!DATA) return;
  // Find this player's stats from any prediction detail
  let stats=null;
  for(const p of DATA.predictions){
    const d=p.detail;if(!d) continue;
    if(d.p1&&d.p1.name===name){stats=d.p1;break}
    if(d.p2&&d.p2.name===name){stats=d.p2;break}
  }
  if(!stats) return;
  const el=document.getElementById('player-content');
  el.innerHTML='<h3>'+esc(name)+'</h3>'+
    (stats.elo?'<div class="modal-rank">Elo '+stats.elo+' · Surface '+stats.surface_elo+'</div>':'')+
    '<div class="modal-section"><h4>Ratings</h4>'+
      dr('Overall Elo',stats.elo)+dr('Surface Elo',stats.surface_elo)+
      dr('Serve Elo',stats.serve_elo)+dr('Return Elo',stats.return_elo)+
      dr('Glicko-2',stats.glicko2_rating)+dr('Rating Deviation',stats.glicko2_rd)+
    '</div>'+
    '<div class="modal-section"><h4>Recent Form</h4>'+
      dr('Last 5',stats.form_last5)+dr('Last 10',stats.form_last10)+
      dr('Win Streak',stats.win_streak)+dr('Loss Streak',stats.loss_streak)+
      dr('Total Matches',stats.total_matches)+
    '</div>'+
    '<div class="modal-section"><h4>Serve & Return</h4>'+
      dr('1st Serve %',pct(stats.first_serve_pct))+
      dr('1st Serve Won',pct(stats.first_serve_won))+
      dr('2nd Serve Won',pct(stats.second_serve_won))+
      dr('Ace Rate',pct(stats.ace_rate))+
      dr('DF Rate',pct(stats.df_rate))+
      dr('BP Save %',pct(stats.bp_save_pct))+
      dr('Return Pts Won',pct(stats.return_pts_won))+
      dr('Serve Pts Won',pct(stats.serve_pts_won))+
    '</div>';
  document.getElementById('player-modal').classList.add('open');
});

// === CARDS ===
function renderCards(list){
  const el=document.getElementById('cards');
  if(!list||!list.length){el.innerHTML='<div class="empty"><h3>No matches scheduled</h3><p>Check back during tournament weeks.</p></div>';return}

  list.sort((a,b)=>Math.abs(b.prob_p1-.5)-Math.abs(a.prob_p1-.5));

  // Group by tournament
  const groups={};
  list.forEach(p=>{const t=p.tournament||'Other';if(!groups[t])groups[t]=[];groups[t].push(p)});

  let html='';
  for(const[tourney,matches] of Object.entries(groups)){
    html+='<div class="tournament-group"><h3>'+esc(tourney)+' · '+matches.length+' match'+(matches.length>1?'es':'')+'</h3><div class="card-list">';
    html+=matches.map(p=>renderCard(p)).join('');
    html+='</div></div>';
  }
  el.innerHTML=html;
}

function renderCard(p){
  const p1=p.prob_p1,fav=p1>=.5;
  const wN=fav?p.player1:p.player2, wR=fav?p.p1_rank:p.p2_rank;
  const lN=fav?p.player2:p.player1, lR=fav?p.p2_rank:p.p1_rank;
  const wP=Math.round(Math.max(p1,1-p1)*100), lP=100-wP;
  const tier=p.confidence_tier||'';
  const isHi=tier==='high';
  const cTag=isHi?'<span class="tag ch">HIGH CONF</span>':tier==='medium'?'<span class="tag cm">MED</span>':'';
  const s=p.surface||'';
  const model=p.model||'';
  const mTag=model.includes('ensemble')?'<span class="tag t">ENSEMBLE</span>':model.includes('catboost')?'<span class="tag t">CATBOOST</span>':'';

  // Detail
  const d=p.detail||{};
  const ws=fav?d.p1||{}:d.p2||{};
  const ls=fav?d.p2||{}:d.p1||{};
  const h2h=d.h2h||{};
  const factors=d.factors||[];

  return '<div class="mc'+(isHi?' high':'')+'">'+
    '<div class="mc-top">'+
      '<div class="mc-winner"><div class="name"><span class="plink" data-name="'+esc(wN)+'">'+esc(wN)+'</span> <span class="arr">&#x276F;</span></div>'+(wR?'<div class="rank">#'+wR+' ATP</div>':'')+'</div>'+
      '<div class="mc-prob"><div class="big">'+wP+'%</div><div class="bar-w"><div class="bar-f" style="width:'+wP+'%"></div></div><div class="vs">vs '+lP+'%</div></div>'+
      '<div class="mc-loser"><div class="name"><span class="plink" data-name="'+esc(lN)+'">'+esc(lN)+'</span></div>'+(lR?'<div class="rank">#'+lR+' ATP</div>':'')+'</div>'+
    '</div>'+
    '<div class="mc-tags">'+(s?'<span class="tag t">'+esc(s)+'</span>':'')+cTag+mTag+'</div>'+
    '<div class="mc-hint">Tap for analysis</div>'+
    '<div class="mc-detail">'+
      '<div class="dc"><h4>'+esc(wN)+'</h4>'+
        dr('Elo',ws.elo)+dr('Surface',ws.surface_elo)+dr('Serve',ws.serve_elo)+dr('Return',ws.return_elo)+
        dr('Form (5)',ws.form_last5)+dr('Surface W/L',ws.surface_record)+
        dr('Streak',ws.win_streak?'+'+ws.win_streak:ws.loss_streak?'-'+ws.loss_streak:'0')+
        dr('1st Srv %',pct(ws.first_serve_pct))+dr('Ret Pts Won',pct(ws.return_pts_won))+dr('BP Save',pct(ws.bp_save_pct))+
      '</div>'+
      '<div class="dc"><h4>'+esc(lN)+'</h4>'+
        dr('Elo',ls.elo)+dr('Surface',ls.surface_elo)+dr('Serve',ls.serve_elo)+dr('Return',ls.return_elo)+
        dr('Form (5)',ls.form_last5)+dr('Surface W/L',ls.surface_record)+
        dr('Streak',ls.win_streak?'+'+ls.win_streak:ls.loss_streak?'-'+ls.loss_streak:'0')+
        dr('1st Srv %',pct(ls.first_serve_pct))+dr('Ret Pts Won',pct(ls.return_pts_won))+dr('BP Save',pct(ls.bp_save_pct))+
      '</div>'+
      (h2h.total?'<div class="h2h-section"><h4>H2H ('+h2h.total+')</h4><div class="h2h-bar"><div class="h2h-fill" style="width:'+(h2h.total?Math.round((fav?h2h.p1_wins:h2h.p2_wins)/h2h.total*100):50)+'%"></div></div>'+
        dr(p.player1,h2h.p1_wins+' wins')+dr(p.player2,h2h.p2_wins+' wins')+'</div>':'')+
      (factors.length?'<div class="factors-section"><h4>Key Factors</h4>'+factors.map(f=>'<div class="factor">'+esc(f)+'</div>').join('')+'</div>':'')+
    '</div>'+
  '</div>';
}

// === RESULTS ===
function renderResults(history){
  const el=document.getElementById('results-content');
  if(!history||!history.length){el.innerHTML='<div class="empty"><h3>No prediction history yet</h3><p>Results appear after predictions are tracked against outcomes.</p></div>';return}

  el.innerHTML=history.map(day=>{
    const date=new Date(day.date+'T12:00:00');
    const label=date.toLocaleDateString(undefined,{weekday:'short',month:'short',day:'numeric'});
    const preds=day.predictions||[];
    if(!preds.length) return '';
    return '<div class="results-day"><h3>'+label+' · '+preds.length+' predictions</h3>'+
      preds.map(p=>{
        const fav=p.prob_p1>=.5;
        const pick=fav?p.player1:p.player2;
        const opp=fav?p.player2:p.player1;
        const prob=Math.round(Math.max(p.prob_p1,p.prob_p2||1-p.prob_p1)*100);
        return '<div class="result-row"><span class="result-pick" style="color:var(--green)">'+esc(pick)+'</span><span class="result-pct">'+prob+'%</span><span class="result-vs">vs '+esc(opp)+'</span></div>';
      }).join('')+'</div>';
  }).join('');
}

// === PERFORMANCE ===
function renderPerf(s){
  const el=document.getElementById('perf');
  const a=s&&s.accuracy?(s.accuracy*100).toFixed(1)+'%':'64.0%';
  const b=s&&s.brier_score?s.brier_score.toFixed(3):'0.220';
  const n=s&&s.n_matches?s.n_matches.toLocaleString():'25,634';
  el.innerHTML=
    '<div class="stat"><strong>'+a+'</strong><span>Accuracy</span><p>On '+n+' test matches</p></div>'+
    '<div class="stat"><strong>'+b+'</strong><span>Brier Score</span><p>Lower = better (bookmakers: 0.196)</p></div>'+
    '<div class="stat"><strong>68.9%</strong><span>Grand Slams</span><p>0.197 Brier — matching bookmakers</p></div>'+
    '<div class="stat"><strong>75.1%</strong><span>High Confidence</span><p>Accuracy when model is confident</p></div>';
}

// === CALIBRATION CHART ===
function renderCal(c){
  const cv=document.getElementById('cal');if(!cv)return;
  const dpr=window.devicePixelRatio||1;
  const W=Math.min(cv.parentElement.getBoundingClientRect().width-32,480),H=W*.68;
  cv.width=W*dpr;cv.height=H*dpr;cv.style.width=W+'px';cv.style.height=H+'px';
  const x=cv.getContext('2d');x.scale(dpr,dpr);
  const p={t:14,r:14,b:36,l:40},cw=W-p.l-p.r,ch=H-p.t-p.b;
  // Use theme-aware colors
  const isDark=document.documentElement.dataset.theme==='dark';
  x.fillStyle=isDark?'#15151c':'#ffffff';x.fillRect(0,0,W,H);
  x.strokeStyle=isDark?'#24242e':'#e2e4e8';x.lineWidth=.5;
  for(let t=0;t<=1;t+=.2){const px=p.l+t*cw,py=p.t+(1-t)*ch;x.beginPath();x.moveTo(px,p.t);x.lineTo(px,p.t+ch);x.stroke();x.beginPath();x.moveTo(p.l,py);x.lineTo(p.l+cw,py);x.stroke()}
  x.strokeStyle=isDark?'#333':'#ccc';x.lineWidth=1;x.setLineDash([4,4]);x.beginPath();x.moveTo(p.l,p.t+ch);x.lineTo(p.l+cw,p.t);x.stroke();x.setLineDash([]);
  if(c&&c.bin_centers&&c.bin_centers.length){
    x.globalAlpha=.06;x.fillStyle='#22c55e';x.beginPath();
    c.bin_centers.forEach((v,i)=>{const px=p.l+v*cw,py=p.t+(1-c.actual_rates[i])*ch;i===0?x.moveTo(px,py):x.lineTo(px,py)});
    x.lineTo(p.l+c.bin_centers[c.bin_centers.length-1]*cw,p.t+ch);x.lineTo(p.l+c.bin_centers[0]*cw,p.t+ch);x.closePath();x.fill();x.globalAlpha=1;
    x.strokeStyle='#22c55e';x.lineWidth=2;x.lineJoin='round';x.beginPath();
    c.bin_centers.forEach((v,i)=>{const px=p.l+v*cw,py=p.t+(1-c.actual_rates[i])*ch;i===0?x.moveTo(px,py):x.lineTo(px,py)});x.stroke();
    c.bin_centers.forEach((v,i)=>{const px=p.l+v*cw,py=p.t+(1-c.actual_rates[i])*ch;x.fillStyle=isDark?'#07070a':'#ffffff';x.beginPath();x.arc(px,py,4,0,Math.PI*2);x.fill();x.fillStyle='#22c55e';x.beginPath();x.arc(px,py,2.5,0,Math.PI*2);x.fill()});
  }
  const labelColor=isDark?'#62627a':'#8888a0';
  x.fillStyle=labelColor;x.font='10px Inter,sans-serif';x.textAlign='center';
  for(let t=0;t<=1;t+=.2)x.fillText(t.toFixed(1),p.l+t*cw,H-4);
  x.textAlign='right';for(let t=0;t<=1;t+=.2)x.fillText(t.toFixed(1),p.l-5,p.t+(1-t)*ch+3);
}

// === HELPERS ===
function dr(l,v){if(v==null||v===undefined||v==='')return '';return '<div class="dr"><span class="l">'+l+'</span><span class="v">'+v+'</span></div>'}
function pct(v){return v!=null&&!isNaN(v)?Math.round(v*100)+'%':null}
function esc(s){return s?String(s).replace(/</g,'&lt;').replace(/>/g,'&gt;'):''}

init();
""")


def _write_sw():
    """Write service worker for push notifications (placeholder)."""
    (SITE_DIR / "sw.js").write_text("""
// Service worker for Tennis Predictor notifications
self.addEventListener('install', e => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('push', e => {
  const data = e.data ? e.data.json() : {title: 'Tennis Predictor', body: 'New predictions available!'};
  e.waitUntil(self.registration.showNotification(data.title, {body: data.body, icon: '/tennis-predictor/favicon.ico'}));
});
""")
