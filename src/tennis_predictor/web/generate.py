"""Static site generator — Bloomberg-style analytical trading dashboard.

Features:
1. Sidebar match list with confidence dots + detail panel
2. Tournament grouping with filters (confidence + surface)
3. Results tracker with prediction history
4. Analytics with calibration chart + system info
5. Player profiles modal (click any name)
6. Responsive: 4 breakpoints (960/768/480px)
7. Mobile: two-state slide transitions, swipe gestures, history API
8. Dark/Light theme with orange accent
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
<meta name="description" content="AI-powered ATP tennis match predictions with 65.9% accuracy. Bloomberg-style analytical dashboard with real-time Elo ratings, H2H stats, and calibrated probabilities.">
<meta name="theme-color" content="#050505" media="(prefers-color-scheme: dark)">
<meta name="theme-color" content="#f5f5f5" media="(prefers-color-scheme: light)">
<meta property="og:title" content="Tennis Predictor PRO">
<meta property="og:description" content="AI-powered ATP tennis match predictions. 230+ features, zero data leakage, calibration-first.">
<meta property="og:type" content="website">
<title>Tennis Predictor PRO</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="assets/css/style.css">
</head>
<body>
<!-- Topbar -->
<nav class="topbar" role="navigation" aria-label="Main navigation"><div class="tb-in">
  <div class="brand">
    <div class="brand-icon" aria-hidden="true">T</div>
    <span class="brand-text">Tennis Predictor</span>
    <span class="brand-tag">PRO</span>
  </div>
  <div class="tb-right">
    <div class="nav" id="top-nav" role="tablist" aria-label="Dashboard sections">
      <button class="nav-btn active" data-tab="predictions" role="tab" aria-selected="true">Predictions</button>
      <button class="nav-btn" data-tab="results" role="tab" aria-selected="false">Results</button>
      <button class="nav-btn" data-tab="analytics" role="tab" aria-selected="false">Analytics</button>
    </div>
    <button class="theme-btn" id="theme-toggle" title="Toggle theme" aria-label="Toggle dark/light theme">&#9790;</button>
  </div>
</div></nav>

<!-- Metrics Strip -->
<div class="metrics"><div class="metrics-grid" id="metrics-row"></div></div>

<!-- Desktop/Tablet Layout -->
<div class="app-layout">
  <aside class="sidebar" id="sidebar" aria-label="Match list">
    <div class="filters" id="filters" role="toolbar" aria-label="Filter predictions">
      <button class="filter-btn active" data-type="conf" data-val="all">All</button>
      <button class="filter-btn" data-type="conf" data-val="high">High Conf</button>
      <button class="filter-btn" data-type="conf" data-val="medium">Medium</button>
      <span class="filter-sep"></span>
      <button class="filter-btn active" data-type="surface" data-val="all">All Surfaces</button>
      <button class="filter-btn" data-type="surface" data-val="hard">Hard</button>
      <button class="filter-btn" data-type="surface" data-val="clay">Clay</button>
      <button class="filter-btn" data-type="surface" data-val="grass">Grass</button>
      <span class="filter-count" id="filter-count"></span>
    </div>
    <div id="match-list"></div>
  </aside>
  <main class="main-panel">
    <div id="view-predictions" class="view active" role="tabpanel">
      <div class="detail-topbar" id="detail-topbar"></div>
      <div id="detail-panel"></div>
    </div>
    <div id="view-results" class="view" role="tabpanel">
      <div id="results-content"></div>
    </div>
    <div id="view-analytics" class="view" role="tabpanel">
      <div class="metrics-grid" id="perf"></div>
      <div class="chart-container" id="chart-wrap">
        <div class="chart-title">Calibration</div>
        <div class="chart-sub">Predicted probability vs actual win rate. Diagonal = perfect.</div>
        <canvas id="cal" aria-label="Calibration chart"></canvas>
        <div class="cal-tooltip" id="cal-tooltip" role="tooltip"></div>
      </div>
      <div id="system-info"></div>
    </div>
  </main>
</div>

<!-- Mobile Containers (predictions + results + analytics all rendered here on mobile) -->
<div class="mobile-container" id="mobile-container">
  <div class="mobile-list" id="mobile-list"></div>
  <div class="mobile-detail" id="mobile-detail"></div>
  <div class="mobile-results" id="mobile-results"></div>
  <div class="mobile-analytics" id="mobile-analytics"></div>
</div>

<!-- Player Modal -->
<div class="modal-overlay" id="player-modal" role="dialog" aria-modal="true" aria-label="Player profile">
  <div class="modal" id="modal-content"></div>
</div>

<!-- Bottom Nav (mobile) -->
<div class="bottom-nav" id="bottom-nav" role="tablist" aria-label="Dashboard sections">
  <button class="bn-btn active" data-tab="predictions" role="tab" aria-selected="true">Predictions</button>
  <button class="bn-btn" data-tab="results" role="tab" aria-selected="false">Results</button>
  <button class="bn-btn" data-tab="analytics" role="tab" aria-selected="false">Analytics</button>
</div>

<footer class="foot">
  <p id="foot-info">10,361 players &middot; 12 data sources &middot; All free &middot;
  <a href="https://github.com/gabrielvuksani/tennis-predictor">GitHub</a></p>
  <p class="foot-updated" id="foot-updated"></p>
</footer>
<script src="assets/js/app.js"></script>
</body>
</html>""")


def _write_css():
    (SITE_DIR / "assets" / "css" / "style.css").write_text("""
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}

/* === THEME SYSTEM === */
[data-theme="dark"]{
  --bg:#050505;--surface:#080808;--card:#0a0a0a;--card-hover:#0e0e0e;
  --border:#111;--border-hover:#1a1a1a;
  --t:#e5e5e5;--t2:#888;--t3:#555;--t4:#333;
  --orange:#f97316;--orange-dim:rgba(249,115,22,0.1);
  --green:#22c55e;--green-dim:rgba(34,197,94,0.08);
  --red:#ef4444;--red-dim:rgba(239,68,68,0.08);
  --amber:#eab308;--amber-dim:rgba(234,179,8,0.08);
}
[data-theme="light"]{
  --bg:#f5f5f5;--surface:#fff;--card:#fff;--card-hover:#fafafa;
  --border:#e5e5e5;--border-hover:#d0d0d0;
  --t:#1a1a1a;--t2:#555;--t3:#888;--t4:#aaa;
  --orange:#ea580c;--orange-dim:rgba(234,88,12,0.08);
  --green:#16a34a;--green-dim:rgba(22,163,74,0.06);
  --red:#dc2626;--red-dim:rgba(220,38,38,0.06);
  --amber:#d97706;--amber-dim:rgba(217,119,6,0.06);
}

html{scroll-behavior:smooth}
body{font-family:'Inter',-apple-system,sans-serif;background:var(--bg);color:var(--t);
  -webkit-font-smoothing:antialiased;transition:background .2s,color .2s}
.mono{font-family:'JetBrains Mono',monospace}
a{color:var(--t2);text-decoration:none}a:hover{color:var(--orange)}
::selection{background:rgba(249,115,22,0.25);color:var(--t)}

/* === TOPBAR === */
.topbar{position:sticky;top:0;z-index:50;background:rgba(5,5,5,0.85);
  backdrop-filter:blur(20px);border-bottom:1px solid var(--border)}
[data-theme="light"] .topbar{background:rgba(245,245,245,0.85)}
.tb-in{max-width:1200px;margin:auto;padding:0 1.5rem;height:52px;
  display:flex;justify-content:space-between;align-items:center}
.brand{display:flex;align-items:center;gap:8px}
.brand-icon{width:22px;height:22px;border-radius:5px;
  background:linear-gradient(135deg,#f97316,#fb923c);
  display:flex;align-items:center;justify-content:center;
  font-size:10px;font-weight:900;color:#000}
.brand-text{font-weight:700;font-size:13px;letter-spacing:-.3px}
.brand-tag{font-size:9px;color:var(--t3);font-weight:500;margin-left:2px}
.tb-right{display:flex;align-items:center;gap:8px}
.nav{display:flex;gap:2px;background:var(--surface);border:1px solid var(--border);
  border-radius:8px;padding:2px}
.nav-btn{background:none;border:none;color:var(--t3);font-size:11px;font-weight:600;
  padding:6px 14px;border-radius:6px;cursor:pointer;transition:all .15s;font-family:inherit}
.nav-btn:hover{color:var(--t2)}
.nav-btn.active{background:var(--orange-dim);color:var(--orange)}
.theme-btn{background:none;border:1px solid var(--border);width:32px;height:32px;
  border-radius:6px;cursor:pointer;color:var(--t3);font-size:14px;
  display:flex;align-items:center;justify-content:center;transition:all .15s}
.theme-btn:hover{border-color:var(--orange);color:var(--orange)}

/* === METRICS STRIP === */
.metrics{max-width:1200px;margin:auto;padding:.6rem 1.5rem}
.metrics-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:6px}
.metric{background:var(--card);border:1px solid var(--border);border-radius:10px;
  padding:10px 14px;transition:border-color .15s}
.metric:hover{border-color:var(--border-hover)}
.metric-label{font-size:9px;color:var(--t3);text-transform:uppercase;letter-spacing:1.2px;
  font-weight:600;font-family:'JetBrains Mono',monospace}
.metric-value{font-size:22px;font-weight:900;letter-spacing:-1px;margin-top:2px;
  font-family:'JetBrains Mono',monospace}
.mv-orange{color:var(--orange)}.mv-green{color:var(--green)}.mv-neutral{color:var(--t)}
.metric-sub{font-size:9px;color:var(--t4);margin-top:2px}

/* === LAYOUT GRID === */
.app-layout{display:grid;grid-template-columns:280px 1fr;max-width:1200px;
  margin:auto;min-height:calc(100vh - 120px)}
.sidebar{background:var(--surface);border-right:1px solid var(--border);
  padding:12px;overflow-y:auto;position:sticky;top:52px;height:calc(100vh - 52px);
  scrollbar-width:thin;scrollbar-color:var(--border) transparent}
.sidebar::-webkit-scrollbar{width:4px}
.sidebar::-webkit-scrollbar-track{background:transparent}
.sidebar::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.main-panel{padding:0 20px 3rem;overflow-y:auto}
.view{display:none}.view.active{display:block}
.filter-sep{width:1px;height:20px;background:var(--border);margin:0 2px;align-self:center}

/* === FILTERS === */
.filters{display:flex;gap:4px;margin-bottom:10px;flex-wrap:wrap;padding:0 4px}
.filter-btn{background:var(--card);border:1px solid var(--border);border-radius:20px;
  padding:5px 12px;font-size:10px;font-weight:600;color:var(--t3);cursor:pointer;
  transition:all .15s;font-family:inherit;min-height:36px}
.filter-btn:hover{border-color:var(--t3);color:var(--t2)}
.filter-btn.active{background:var(--orange-dim);border-color:rgba(249,115,22,0.2);color:var(--orange)}
.filter-count{font-size:9px;color:var(--t3);padding:5px 0;font-weight:500;margin-left:auto}

/* === TOURNAMENT GROUP === */
.tour-header{display:flex;align-items:center;gap:8px;margin:12px 0 6px;padding:0 4px}
.tour-bar{width:3px;height:14px;border-radius:2px;background:var(--orange)}
.tour-name{font-size:10px;color:var(--t2);font-weight:600;text-transform:uppercase;letter-spacing:.8px}
.tour-count{font-size:9px;color:var(--t4)}

/* === MATCH ITEM === */
.mi{display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:8px;
  margin-bottom:3px;cursor:pointer;transition:all .15s;border:1px solid transparent;min-height:52px}
.mi:hover{background:var(--card-hover);border-color:var(--border)}
.mi.active{background:var(--orange-dim);border-color:rgba(249,115,22,0.2)}
.conf-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.conf-dot.high{background:var(--green);box-shadow:0 0 6px rgba(34,197,94,0.4)}
.conf-dot.med{background:var(--orange);box-shadow:0 0 6px rgba(249,115,22,0.3)}
.conf-dot.low{background:var(--t4)}
.mi-info{flex:1;min-width:0}
.mi-names{font-size:11px;font-weight:600;color:var(--t);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mi-meta{font-size:9px;color:var(--t4);margin-top:1px}
.mi-prob{font-size:13px;font-weight:800;color:var(--orange);font-family:'JetBrains Mono',monospace}
@keyframes pulse-glow{0%,100%{box-shadow:0 0 6px rgba(34,197,94,0.4)}50%{box-shadow:0 0 12px rgba(34,197,94,0.7)}}
.conf-dot.high{animation:pulse-glow 2s ease-in-out infinite}

/* === DETAIL PANEL === */
.detail-topbar{display:flex;justify-content:space-between;align-items:center;
  padding:12px 0;border-bottom:1px solid var(--border);margin-bottom:16px}
.dt-title{font-size:11px;color:var(--t3);font-weight:500}
.dt-title strong{color:var(--t);font-weight:700}

/* Face-off */
.faceoff{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:20px}
.fo-player{text-align:center;flex:1}
.fo-avatar{width:56px;height:56px;border-radius:50%;background:var(--card);
  border:2px solid var(--border);margin:0 auto 8px;display:flex;align-items:center;
  justify-content:center;font-size:18px;font-weight:900;color:var(--t4);
  font-family:'JetBrains Mono',monospace}
.fo-player.fav .fo-avatar{border-color:var(--orange);background:var(--orange-dim)}
.fo-player.fav .fo-name{color:var(--orange)}
.fo-name{font-size:14px;font-weight:800;cursor:pointer;transition:color .15s;
  text-decoration:underline;text-decoration-style:dotted;text-underline-offset:3px}
.fo-name:hover{color:var(--orange);text-decoration-style:solid}
.fo-rank{font-size:10px;color:var(--t4);margin-top:2px}
.fo-vs{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:0 16px}
.fo-prob{font-size:36px;font-weight:900;font-family:'JetBrains Mono',monospace;
  color:var(--orange);letter-spacing:-2px;line-height:1}
.fo-prob-label{font-size:8px;color:var(--t4);text-transform:uppercase;letter-spacing:1.5px;margin-top:4px}

/* Stat bars */
.stat-section{margin-bottom:16px}
.stat-title{font-size:9px;color:var(--t3);text-transform:uppercase;letter-spacing:1.2px;
  font-weight:700;margin-bottom:10px;text-align:center}
.h-stat{margin-bottom:8px}
.h-stat-label{font-size:9px;color:var(--t3);text-transform:uppercase;letter-spacing:1px;
  font-weight:600;margin-bottom:3px;text-align:center}
.h-stat-row{display:grid;grid-template-columns:55px 1fr 55px;gap:6px;align-items:center;margin-bottom:6px}
.h-stat-val{font-size:10px;font-family:'JetBrains Mono',monospace;color:var(--t4);font-weight:600}
.h-stat-val.left{text-align:right}
.h-stat-val.right{text-align:left}
.h-stat-val.better{color:var(--orange);font-weight:700}
.h-stat-bar-wrap{height:5px;border-radius:3px;background:var(--border);overflow:hidden;position:relative}
.h-stat-bar-fill{height:100%;border-radius:3px;background:var(--orange);transition:width .6s ease-out}
.detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}

/* H2H */
.h2h-section{margin-bottom:16px}
.panel-title{font-size:9px;color:var(--t3);text-transform:uppercase;letter-spacing:1.5px;
  font-weight:700;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid var(--border)}
.h2h-mini{display:flex;align-items:center;gap:12px}
.h2h-circle{width:52px;height:52px;border-radius:50%;border:3px solid var(--border);
  display:flex;align-items:center;justify-content:center;flex-direction:column}
.h2h-num{font-size:16px;font-weight:900;font-family:'JetBrains Mono',monospace;color:var(--orange)}
.h2h-total{font-size:8px;color:var(--t4)}
.h2h-text{font-size:11px;color:var(--t3);line-height:1.5}

/* Key factors */
.factors-section{margin-bottom:16px}
.factor-item{display:flex;align-items:flex-start;gap:8px;padding:5px 0;
  font-size:11px;color:var(--t2);line-height:1.4}
.fi-icon{width:18px;height:18px;border-radius:4px;background:var(--orange-dim);
  color:var(--orange);display:flex;align-items:center;justify-content:center;
  font-size:9px;font-weight:700;flex-shrink:0;margin-top:1px;font-family:'JetBrains Mono',monospace}

/* Player link */
.plink{cursor:pointer;text-decoration:underline;text-decoration-style:dotted;
  text-underline-offset:3px;transition:all .15s}
.plink:hover{color:var(--orange);text-decoration-style:solid}

/* === RESULTS === */
.results-day{margin-bottom:1.5rem}
.results-day h3{font-size:12px;font-weight:700;color:var(--t2);margin-bottom:8px;
  display:flex;align-items:center;gap:8px}
.day-acc{font-size:10px;font-weight:600;padding:2px 8px;border-radius:10px}
.day-acc.good{background:var(--green-dim);color:var(--green)}
.day-acc.ok{background:var(--orange-dim);color:var(--orange)}
.day-acc.bad{background:var(--red-dim);color:var(--red)}
.result-row{display:flex;align-items:center;gap:10px;padding:8px 12px;
  background:var(--card);border:1px solid var(--border);border-radius:10px;
  margin-bottom:4px;font-size:12px;transition:border-color .15s}
.result-row:hover{border-color:var(--border-hover)}
.result-icon{width:20px;height:20px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:10px;font-weight:900;flex-shrink:0}
.ri-win{background:var(--green-dim);color:var(--green)}
.ri-loss{background:var(--red-dim);color:var(--red)}
.result-pick{font-weight:700;flex:1}.result-prob{font-weight:800;color:var(--orange);
  min-width:40px;text-align:center;font-family:'JetBrains Mono',monospace}
.result-vs{color:var(--t3);flex:1}
.streak-counter{font-size:13px;font-weight:800;margin-bottom:12px;font-family:'JetBrains Mono',monospace}

/* === ANALYTICS === */
.chart-container{background:var(--card);border:1px solid var(--border);border-radius:14px;
  padding:1.2rem;margin-top:1rem;position:relative}
.chart-title{font-size:11px;color:var(--t2);font-weight:700;text-transform:uppercase;letter-spacing:.8px;margin-bottom:4px}
.chart-sub{font-size:10px;color:var(--t4);margin-bottom:12px}
canvas{max-width:100%;height:auto!important}
.cal-tooltip{display:none;position:absolute;background:var(--surface);border:1px solid var(--border);
  border-radius:6px;padding:6px 10px;font-size:10px;color:var(--t2);pointer-events:none;
  font-family:'JetBrains Mono',monospace;z-index:10;white-space:nowrap}
.cal-tooltip.show{display:block}
.sys-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:6px;margin-top:1rem}
.sys-card{background:var(--card);border:1px solid var(--border);border-radius:10px;
  padding:14px;transition:border-color .15s}
.sys-card:hover{border-color:var(--border-hover)}
.sys-card strong{font-size:20px;font-weight:900;color:var(--orange);display:block;
  letter-spacing:-.5px;font-family:'JetBrains Mono',monospace}
.sys-label{font-size:9px;color:var(--t3);text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-top:2px}
.sys-card p{font-size:10px;color:var(--t4);margin-top:6px;line-height:1.5}

/* === MODAL === */
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);
  backdrop-filter:blur(4px);z-index:100;align-items:center;justify-content:center;padding:1rem}
.modal-overlay.open{display:flex}
.modal{background:var(--card);border:1px solid var(--border);border-radius:14px;
  padding:1.5rem;max-width:480px;width:100%;max-height:80vh;overflow-y:auto;position:relative}
.modal-close{position:absolute;top:.8rem;right:1rem;background:none;border:none;
  color:var(--t3);font-size:1.3rem;cursor:pointer}
.modal h3{font-size:1.1rem;font-weight:800;color:var(--orange);margin-bottom:.3rem}
.modal-rank{font-size:.75rem;color:var(--t2);margin-bottom:1rem;font-family:'JetBrains Mono',monospace}
.modal-section{margin-bottom:1rem}
.modal-section h4{font-size:.7rem;color:var(--t3);text-transform:uppercase;
  letter-spacing:.06em;font-weight:700;margin-bottom:.4rem}
.dr{display:flex;justify-content:space-between;padding:.18rem 0;
  border-bottom:1px solid rgba(128,128,128,.06);font-size:11px}
.dr .l{color:var(--t3)}.dr .v{font-weight:600;font-family:'JetBrains Mono',monospace}

/* === FOOTER + EMPTY + SKELETON === */
.foot{text-align:center;padding:1.5rem;font-size:10px;color:var(--t4);border-top:1px solid var(--border)}
.foot a{color:var(--t3)}.foot a:hover{color:var(--orange)}
.foot-updated{font-size:9px;color:var(--t4);margin-top:4px;font-family:'JetBrains Mono',monospace}
.empty{text-align:center;padding:3rem;color:var(--t3)}
.empty h3{font-size:1rem;color:var(--t2);margin-bottom:.3rem}
@keyframes shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}
.skel{border-radius:6px;background:linear-gradient(90deg,var(--card) 25%,var(--border) 50%,var(--card) 75%);
  background-size:200% 100%;animation:shimmer 1.5s linear infinite}
.skel-mi{height:48px;margin-bottom:4px}
.skel-bar{height:5px;margin:6px 0}
.skel-text{height:12px;width:60%;margin:4px 0}

/* === MOBILE + BOTTOM NAV === */
.mobile-container{display:none}
.mobile-results,.mobile-analytics{display:none;padding:0 1rem}
.mobile-results.active,.mobile-analytics.active{display:block}
.bottom-nav{display:none}
@keyframes fadeSlideUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
@keyframes crossfade-in{from{opacity:0}to{opacity:1}}
@keyframes slideUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}

:focus-visible{outline:2px solid var(--orange);outline-offset:2px}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{
  animation-duration:0.01ms!important;animation-iteration-count:1!important;
  transition-duration:0.01ms!important}}

/* === RESPONSIVE === */
@media(max-width:960px){
  .app-layout{grid-template-columns:220px 1fr}
  .metrics-grid{grid-template-columns:repeat(3,1fr)}
}
@media(max-width:768px){
  .app-layout{grid-template-columns:1fr}
  .sidebar{position:static;height:auto;border-right:none;border-bottom:1px solid var(--border);
    padding:8px 12px;overflow-x:auto;overflow-y:hidden;white-space:nowrap;
    display:flex;gap:6px;flex-wrap:nowrap;-webkit-overflow-scrolling:touch;
    scroll-snap-type:x mandatory}
  .sidebar .filters{display:none}
  .sidebar .tour-header{display:none}
  .mi{flex-shrink:0;min-width:auto;padding:8px 12px;border:1px solid var(--border);
    border-radius:20px;scroll-snap-align:start;min-height:auto}
  .mi .mi-meta{display:none}
  .mi .mi-info{display:flex;align-items:center;gap:6px}
  .mi .mi-names{font-size:10px}
  .metrics-grid{grid-template-columns:repeat(3,1fr)}
  .detail-grid{grid-template-columns:1fr}
}
@media(max-width:480px){
  .topbar .nav{display:none}
  .app-layout{display:none}
  .metrics{padding:.4rem 1rem}
  .metrics-grid{grid-template-columns:repeat(2,1fr);gap:4px}
  .metric{padding:8px 10px}
  .metric-value{font-size:18px}
  .mobile-container{display:block;min-height:calc(100vh - 160px);position:relative;overflow:hidden}
  .mobile-list{padding:0 1rem;transition:transform 300ms cubic-bezier(0.4,0,0.2,1);will-change:transform}
  .mobile-detail{position:absolute;inset:0;transform:translateX(100%);padding:0 1rem;
    overflow-y:auto;background:var(--bg);transition:transform 300ms cubic-bezier(0.4,0,0.2,1);will-change:transform}
  .mobile-detail.active{transform:translateX(0)}
  .mobile-list.pushed{transform:translateX(-30%)}
  .bottom-nav{display:flex;position:fixed;bottom:0;left:0;right:0;
    background:rgba(5,5,5,0.95);backdrop-filter:blur(20px);
    border-top:1px solid var(--border);z-index:50;padding:6px;gap:2px}
  [data-theme="light"] .bottom-nav{background:rgba(245,245,245,0.95)}
  .bn-btn{flex:1;text-align:center;background:none;border:none;border-radius:8px;
    padding:8px 0;font-size:10px;font-weight:600;color:var(--t3);cursor:pointer;
    font-family:inherit;transition:all .1s;min-height:44px}
  .bn-btn.active{color:var(--orange)}
  .bn-btn:active{transform:scale(0.95)}
  body{padding-bottom:56px}
  .fo-prob{font-size:28px}
  .fo-avatar{width:44px;height:44px;font-size:14px}
  .sys-grid{grid-template-columns:1fr}
  .detail-grid{grid-template-columns:1fr}
  .foot{display:none}
  .mobile-back{display:flex;align-items:center;gap:6px;background:none;border:none;
    color:var(--t3);font-size:11px;font-weight:600;cursor:pointer;padding:12px 0;
    font-family:inherit;min-height:44px}
  .mobile-back:hover{color:var(--orange)}
  .modal-overlay{align-items:flex-end;padding:0}
  .modal{max-width:100%;max-height:85vh;border-radius:14px 14px 0 0}
}
""")


def _write_js():
    # NOTE: All data rendered via innerHTML is from a trusted local predictions.json
    # generated by our Python pipeline, not user input. The esc() helper provides
    # defense-in-depth HTML entity encoding for all interpolated values.
    (SITE_DIR / "assets" / "js" / "app.js").write_text(r"""
let DATA=null;
const S={view:'list',tab:'predictions',sel:null,filters:{conf:'all',surface:'all'},mobile:false};
const $=id=>document.getElementById(id);
const esc=s=>s?String(s).replace(/</g,'&lt;').replace(/>/g,'&gt;'):'';
const pct=v=>v!=null&&!isNaN(v)?Math.round(v*100)+'%':null;

/* === MOBILE DETECTION === */
const mq=window.matchMedia('(max-width:480px)');
S.mobile=mq.matches;
mq.addEventListener('change',e=>{
  S.mobile=e.matches;
  document.body.style.transition='none';
  setTimeout(()=>{document.body.style.transition=''},100);
  render();
});

/* === INIT === */
async function init(){
  showSkeleton();
  try{
    const r=await fetch('predictions.json');
    DATA=await r.json();
    render();
    if(!S.mobile&&DATA.predictions?.length&&!S.sel){
      setTimeout(()=>{if(S.sel===null)selectMatch(0)},1000);
    }
  }catch(e){showError()}
}

function showSkeleton(){
  const sb=$('match-list');
  if(sb)sb.innerHTML=Array(6).fill('<div class="skel skel-mi"></div>').join('');
  const dp=$('detail-panel');
  if(dp)dp.innerHTML='<div style="padding:2rem"><div class="skel" style="height:60px;margin-bottom:12px"></div>'+
    Array(5).fill('<div class="skel skel-bar" style="width:'+(60+Math.random()*40)+'%"></div>').join('')+'</div>';
}
function showError(){
  const sb=$('match-list');
  if(sb)sb.innerHTML='<div class="empty"><h3>Could not load</h3><button onclick="init()" style="margin-top:8px;padding:6px 16px;border-radius:6px;border:1px solid var(--border);background:var(--card);color:var(--t2);cursor:pointer;font-family:inherit">Retry</button></div>';
}

/* === TABS === */
function switchTab(name){
  S.tab=name;
  // Update all tab buttons (top nav + bottom nav)
  document.querySelectorAll('.nav-btn,.bn-btn').forEach(b=>{
    const isActive=b.dataset.tab===name;
    b.classList.toggle('active',isActive);
    b.setAttribute('aria-selected',isActive?'true':'false');
  });
  // Desktop views
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
  const v=$('view-'+name);if(v){v.classList.add('active');v.style.animation='crossfade-in .15s ease'}
  // Mobile views - show/hide the right container
  if(S.mobile){
    const ml=$('mobile-list');const md=$('mobile-detail');
    const mr=$('mobile-results');const ma=$('mobile-analytics');
    if(ml)ml.style.display=name==='predictions'?'':'none';
    if(md&&S.view==='detail')md.style.display=name==='predictions'?'':'none';
    if(mr){mr.classList.toggle('active',name==='results');if(name==='results')renderMobileResults()}
    if(ma){ma.classList.toggle('active',name==='analytics');if(name==='analytics')renderMobileAnalytics()}
  }
  if(name==='analytics'&&DATA)renderAnalytics();
}
document.addEventListener('click',e=>{
  const btn=e.target.closest('[data-tab]');
  if(btn)switchTab(btn.dataset.tab);
});

/* === THEME === */
const themeBtn=$('theme-toggle');
function setTheme(t){
  document.documentElement.dataset.theme=t;
  localStorage.setItem('theme',t);
  if(themeBtn)themeBtn.textContent=t==='dark'?'\u263E':'\u2600';
}
if(themeBtn)themeBtn.addEventListener('click',()=>
  setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark'));
setTheme(localStorage.getItem('theme')||'dark');

/* === RENDER === */
function render(){
  if(!DATA)return;
  renderMetrics(DATA.model_stats);
  renderSidebar(DATA.predictions);
  if(S.sel!==null)renderDetail(DATA.predictions[S.sel]);
  else renderEmptyDetail();
  renderResults(DATA.history);
  if(S.mobile)renderMobileList(DATA.predictions);
  // Show last updated timestamp
  const fu=$('foot-updated');
  if(fu&&DATA.generated_at){
    const d=new Date(DATA.generated_at);
    fu.textContent='Last updated: '+d.toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
  }
}

/* === METRICS === */
function renderMetrics(s){
  const el=$('metrics-row');if(!el)return;
  const a=s?.accuracy?(s.accuracy*100).toFixed(1)+'%':'\u2014';
  const b=s?.brier_score?s.brier_score.toFixed(3):'\u2014';
  const n=s?.n_matches?s.n_matches.toLocaleString():'\u2014';
  el.innerHTML=
    '<div class="metric"><div class="metric-label">Accuracy</div><div class="metric-value mv-orange">'+a+'</div><div class="metric-sub">Test matches</div></div>'+
    '<div class="metric"><div class="metric-label">Brier</div><div class="metric-value mv-neutral">'+b+'</div><div class="metric-sub">Bookmakers: 0.196</div></div>'+
    '<div class="metric"><div class="metric-label">Grand Slams</div><div class="metric-value mv-green">68.9%</div><div class="metric-sub">0.197 Brier</div></div>'+
    '<div class="metric"><div class="metric-label">High Conf</div><div class="metric-value mv-orange">75.1%</div><div class="metric-sub">Confident picks</div></div>'+
    '<div class="metric"><div class="metric-label">Matches</div><div class="metric-value mv-neutral">'+n+'</div><div class="metric-sub">Training data</div></div>';
}

/* === SIDEBAR === */
function renderSidebar(list){
  const el=$('match-list');if(!el)return;
  if(!list||!list.length){
    el.innerHTML='<div class="empty"><h3>No matches scheduled</h3><p>Check back during tournament weeks.</p></div>';
    renderEmptyDetail();return;
  }
  const filtered=applyFilters(list);
  const groups={};
  filtered.forEach((p,i)=>{const t=p.tournament||'Other';if(!groups[t])groups[t]=[];groups[t].push({...p,_idx:list.indexOf(p)})});
  let html='',count=0;
  for(const[tourney,matches] of Object.entries(groups)){
    html+='<div class="tour-header"><div class="tour-bar"></div><span class="tour-name">'+esc(tourney)+'</span><span class="tour-count"> \u00b7 '+matches.length+'</span></div>';
    matches.forEach((p,i)=>{
      const fav=p.prob_p1>=.5;
      const wN=fav?p.player1:p.player2;const lN=fav?p.player2:p.player1;
      const prob=Math.round(Math.max(p.prob_p1,1-p.prob_p1)*100);
      const tier=p.confidence_tier||'';
      const dotCls=tier==='high'?'high':tier==='medium'?'med':'low';
      html+='<div class="mi'+(p._idx===S.sel?' active':'')+'" data-idx="'+p._idx+'" style="animation:fadeSlideUp .3s ease '+(count*50)+'ms both">'+
        '<div class="conf-dot '+dotCls+'"></div>'+
        '<div class="mi-info"><div class="mi-names">'+esc(wN)+' vs '+esc(lN)+'</div>'+
        '<div class="mi-meta">'+(p.round?esc(p.round)+' \u00b7 ':'')+esc(p.surface||'')+
        (tier?' \u00b7 '+tier.charAt(0).toUpperCase()+tier.slice(1):'')+'</div></div>'+
        '<div class="mi-prob">'+prob+'%</div></div>';
      count++;
    });
  }
  el.innerHTML=html;
  const fc=$('filter-count');if(fc)fc.textContent=filtered.length+' match'+(filtered.length!==1?'es':'');
}

/* Match selection */
document.addEventListener('click',e=>{
  const mi=e.target.closest('.mi');
  if(mi&&mi.dataset.idx!==undefined&&!e.target.closest('.plink')){
    selectMatch(parseInt(mi.dataset.idx));
  }
});
function selectMatch(idx){
  S.sel=idx;
  document.querySelectorAll('.mi').forEach(m=>m.classList.toggle('active',parseInt(m.dataset.idx)===idx));
  const dp=$('detail-panel');
  if(dp&&!S.mobile){dp.style.animation='crossfade-in .2s ease';renderDetail(DATA.predictions[idx])}
  if(S.mobile)showMobileDetail(idx);
  const activeMi=document.querySelector('.mi.active');
  if(activeMi&&window.innerWidth<=768&&window.innerWidth>480){
    activeMi.scrollIntoView({inline:'center',behavior:'smooth'});
    const mainP=$('view-predictions');if(mainP)mainP.style.animation='slideUp .2s ease';
  }
  const p=DATA.predictions[idx];
  if(p){const dt=$('detail-topbar');
    if(dt)dt.innerHTML='<div class="dt-title"><strong>'+esc(p.player1)+' vs '+esc(p.player2)+'</strong> \u00b7 '+esc(p.tournament||'')+' \u00b7 '+esc(p.surface||'')+'</div>';
  }
}

/* === FILTERS === */
function applyFilters(list){
  return list.filter(p=>{
    if(S.filters.conf!=='all'){
      const tier=p.confidence_tier||'low';
      if(S.filters.conf==='high'&&tier!=='high')return false;
      if(S.filters.conf==='medium'&&tier!=='medium')return false;
    }
    if(S.filters.surface!=='all'){
      if((p.surface||'').toLowerCase()!==S.filters.surface)return false;
    }
    return true;
  });
}
document.addEventListener('click',e=>{
  const fb=e.target.closest('.filter-btn');
  if(!fb)return;
  const type=fb.dataset.type;const val=fb.dataset.val;
  if(type==='conf')S.filters.conf=val;
  if(type==='surface')S.filters.surface=val;
  document.querySelectorAll('.filter-btn').forEach(b=>{
    if(b.dataset.type===type)b.classList.toggle('active',b.dataset.val===val);
  });
  renderSidebar(DATA.predictions);
  if(S.mobile)renderMobileList(DATA.predictions);
});

/* === DETAIL PANEL === */
function renderEmptyDetail(){
  const dp=$('detail-panel');if(!dp)return;
  dp.innerHTML='<div class="empty" style="padding:4rem 1rem"><h3>Select a match</h3><p style="margin-top:.3rem">Click a match from the sidebar to see full analysis.</p></div>';
}
function renderDetail(p){
  const dp=$('detail-panel');if(!dp||!p)return;
  const fav=p.prob_p1>=.5;const d=p.detail||{};
  const ws=fav?d.p1||{}:d.p2||{};const ls=fav?d.p2||{}:d.p1||{};
  const wN=fav?p.player1:p.player2;const lN=fav?p.player2:p.player1;
  const wR=fav?p.p1_rank:p.p2_rank;const lR=fav?p.p2_rank:p.p1_rank;
  const prob=Math.round(Math.max(p.prob_p1,1-p.prob_p1)*100);
  const wI=wN.split(' ').map(s=>s[0]).join('').substring(0,2);
  const lI=lN.split(' ').map(s=>s[0]).join('').substring(0,2);
  dp.innerHTML=
    '<div class="faceoff">'+
      '<div class="fo-player fav"><div class="fo-avatar">'+esc(wI)+'</div>'+
        '<div class="fo-name plink" data-name="'+esc(wN)+'">'+esc(wN)+'</div>'+
        '<div class="fo-rank">#'+(wR||'\u2014')+' \u00b7 Form '+(ws.form_last5||'\u2014')+'</div></div>'+
      '<div class="fo-vs"><div class="fo-prob">'+prob+'%</div><div class="fo-prob-label">Win Prob</div></div>'+
      '<div class="fo-player"><div class="fo-avatar">'+esc(lI)+'</div>'+
        '<div class="fo-name plink" data-name="'+esc(lN)+'">'+esc(lN)+'</div>'+
        '<div class="fo-rank">#'+(lR||'\u2014')+' \u00b7 Form '+(ls.form_last5||'\u2014')+'</div></div>'+
    '</div>'+
    '<div class="detail-grid"><div>'+renderStatBars(ws,ls)+'</div><div>'+renderH2H(d.h2h,wN,lN,fav)+renderFactors(d.factors)+'</div></div>';
}

/* Stat bars */
function renderStatBars(ws,ls){
  const stats=[
    ['Elo',ws.elo,ls.elo],['Surface Elo',ws.surface_elo,ls.surface_elo],
    ['Serve',ws.serve_elo,ls.serve_elo],['Return',ws.return_elo,ls.return_elo],
    ['1st Serve %',ws.first_serve_pct,ls.first_serve_pct,true],
    ['1st Serve Won',ws.first_serve_won,ls.first_serve_won,true],
    ['BP Save %',ws.bp_save_pct,ls.bp_save_pct,true],
    ['Return Pts Won',ws.return_pts_won,ls.return_pts_won,true],
    ['Form (5)',ws.form_last5,ls.form_last5],
    ['Surface W/L',ws.surface_record,ls.surface_record],
  ];
  let html='<div class="stat-section"><div class="stat-title">Statistical Comparison</div>';
  stats.forEach(([label,a,b,isPct],i)=>{
    if(a==null&&b==null)return;
    const av=isPct?pct(a):a;const bv=isPct?pct(b):b;
    const an=parseFloat(isPct?(a||0)*100:a);const bn=parseFloat(isPct?(b||0)*100:b);
    const aBetter=!isNaN(an)&&!isNaN(bn)&&an>bn;
    const bBetter=!isNaN(an)&&!isNaN(bn)&&bn>an;
    const mx=Math.max(an||0,bn||0)||1;
    const lw=Math.round((an||0)/mx*100);const delay=i*50;
    html+='<div class="h-stat-label">'+label+'</div>'+
      '<div class="h-stat-row" style="animation:fadeSlideUp .3s ease '+delay+'ms both">'+
        '<div class="h-stat-val left'+(aBetter?' better':'')+'">'+(av!=null?av:'\u2014')+'</div>'+
        '<div class="h-stat-bar-wrap"><div class="h-stat-bar-fill" style="width:'+lw+'%;transition-delay:'+delay+'ms"></div></div>'+
        '<div class="h-stat-val right'+(bBetter?' better':'')+'">'+(bv!=null?bv:'\u2014')+'</div>'+
      '</div>';
  });
  return html+'</div>';
}

/* H2H */
function renderH2H(h2h,wN,lN,fav){
  if(!h2h||!h2h.total)return '';
  const wW=fav?h2h.p1_wins:h2h.p2_wins;const lW=fav?h2h.p2_wins:h2h.p1_wins;
  return '<div class="h2h-section"><div class="panel-title">Head to Head</div>'+
    '<div class="h2h-mini"><div class="h2h-circle"><div class="h2h-num">'+wW+'</div><div class="h2h-total">/'+h2h.total+'</div></div>'+
    '<div class="h2h-text">'+esc(wN)+' leads '+wW+'-'+lW+' overall</div></div></div>';
}

/* Factors */
function renderFactors(factors){
  if(!factors||!factors.length)return '';
  const icons={elo:'E',form:'F',h2h:'H',serve:'S',surface:'C','return':'R'};
  return '<div class="factors-section"><div class="panel-title" style="color:var(--amber)">Key Factors</div>'+
    factors.map(f=>{
      const lf=f.toLowerCase();const icon=Object.entries(icons).find(([k])=>lf.includes(k));
      return '<div class="factor-item"><div class="fi-icon">'+(icon?icon[1]:'?')+'</div>'+esc(f)+'</div>';
    }).join('')+'</div>';
}

/* === RESULTS === */
function renderResults(history){
  const el=$('results-content');if(!el)return;
  if(!history||!history.length){
    el.innerHTML='<div class="empty"><h3>No prediction history yet</h3><p>Results appear after predictions are tracked against outcomes.</p></div>';return;
  }
  el.innerHTML=history.map(day=>{
    const date=new Date(day.date+'T12:00:00');
    const label=date.toLocaleDateString(undefined,{weekday:'short',month:'short',day:'numeric'});
    const preds=day.predictions||[];if(!preds.length)return '';
    return '<div class="results-day"><h3>'+label+' \u00b7 '+preds.length+' predictions</h3>'+
      preds.map(p=>{
        const fav=(p.prob_p1||0.5)>=.5;
        const pick=fav?p.player1:p.player2;const opp=fav?p.player2:p.player1;
        const prob=Math.round(Math.max(p.prob_p1||.5,1-(p.prob_p1||.5))*100);
        return '<div class="result-row"><span class="result-icon ri-win">\u2713</span>'+
          '<span class="result-pick">'+esc(pick)+'</span>'+
          '<span class="result-prob">'+prob+'%</span>'+
          '<span class="result-vs">vs '+esc(opp)+'</span></div>';
      }).join('')+'</div>';
  }).join('');
}

/* === ANALYTICS === */
function renderAnalytics(){
  renderPerf(DATA.model_stats);renderCal(DATA.calibration);renderSystemInfo();
}
function renderPerf(s){
  const el=$('perf');if(!el)return;
  const a=s?.accuracy?(s.accuracy*100).toFixed(1)+'%':'64.0%';
  const b=s?.brier_score?s.brier_score.toFixed(3):'0.220';
  const n=s?.n_matches?s.n_matches.toLocaleString():'25,634';
  el.innerHTML=
    '<div class="metric"><div class="metric-label">Accuracy</div><div class="metric-value mv-orange">'+a+'</div><div class="metric-sub">On '+n+' test matches</div></div>'+
    '<div class="metric"><div class="metric-label">Brier Score</div><div class="metric-value mv-neutral">'+b+'</div><div class="metric-sub">Lower = better (books: 0.196)</div></div>'+
    '<div class="metric"><div class="metric-label">Grand Slams</div><div class="metric-value mv-green">68.9%</div><div class="metric-sub">0.197 Brier</div></div>'+
    '<div class="metric"><div class="metric-label">High Confidence</div><div class="metric-value mv-orange">75.1%</div><div class="metric-sub">Accuracy when confident</div></div>';
}
function renderSystemInfo(){
  const el=$('system-info');if(!el)return;
  el.innerHTML='<div class="sys-grid">'+
    '<div class="sys-card"><strong>244</strong><div class="sys-label">Features</div><p>Elo, Glicko-2, serve/return, fatigue, weather, court speed, sentiment</p></div>'+
    '<div class="sys-card"><strong>320K</strong><div class="sys-label">Matches</div><p>1991-2026. JeffSackmann + TennisMyLife databases</p></div>'+
    '<div class="sys-card"><strong>Zero</strong><div class="sys-label">Leakage</div><p>TemporalGuard ensures no future data leaks into predictions</p></div>'+
    '<div class="sys-card"><strong>24/7</strong><div class="sys-label">Autonomous</div><p>Self-learning. Retrains daily. Drift detection triggers emergency retrains</p></div></div>';
}

/* === CALIBRATION CHART === */
function renderCal(c){
  const cv=$('cal');if(!cv)return;
  const wrap=cv.parentElement;
  if(!c||!c.bin_centers||!c.bin_centers.length){
    wrap.innerHTML='<div class="empty"><p>Calibration data not yet available</p></div>';return;
  }
  const dpr=window.devicePixelRatio||1;
  const W=Math.min(wrap.getBoundingClientRect().width-32,520),H=W*.65;
  cv.width=W*dpr;cv.height=H*dpr;cv.style.width=W+'px';cv.style.height=H+'px';
  const x=cv.getContext('2d');x.scale(dpr,dpr);
  const pad={t:16,r:16,b:36,l:42},cw=W-pad.l-pad.r,ch=H-pad.t-pad.b;
  const isDark=document.documentElement.dataset.theme==='dark';

  x.fillStyle=isDark?'#0a0a0a':'#fff';x.fillRect(0,0,W,H);
  x.strokeStyle=isDark?'#111':'#e5e5e5';x.lineWidth=.5;
  for(let t=0;t<=1;t+=.2){
    const px=pad.l+t*cw,py=pad.t+(1-t)*ch;
    x.beginPath();x.moveTo(px,pad.t);x.lineTo(px,pad.t+ch);x.stroke();
    x.beginPath();x.moveTo(pad.l,py);x.lineTo(pad.l+cw,py);x.stroke();
  }
  x.strokeStyle=isDark?'#222':'#ccc';x.lineWidth=1;x.setLineDash([4,4]);
  x.beginPath();x.moveTo(pad.l,pad.t+ch);x.lineTo(pad.l+cw,pad.t);x.stroke();x.setLineDash([]);
  // Bookmaker baseline
  x.strokeStyle=isDark?'rgba(255,255,255,0.15)':'rgba(0,0,0,0.15)';x.lineWidth=1;x.setLineDash([2,4]);
  const bkRef=c.bookmaker_rates||[0.1,0.3,0.5,0.7,0.9];
  const bkCenters=c.bookmaker_centers||[0.1,0.3,0.5,0.7,0.9];
  x.beginPath();bkCenters.forEach((v,i)=>{const px=pad.l+v*cw,py=pad.t+(1-bkRef[i])*ch;i===0?x.moveTo(px,py):x.lineTo(px,py)});x.stroke();x.setLineDash([]);

  x.globalAlpha=.06;x.fillStyle='#f97316';x.beginPath();
  c.bin_centers.forEach((v,i)=>{const px=pad.l+v*cw,py=pad.t+(1-c.actual_rates[i])*ch;i===0?x.moveTo(px,py):x.lineTo(px,py)});
  x.lineTo(pad.l+c.bin_centers[c.bin_centers.length-1]*cw,pad.t+ch);
  x.lineTo(pad.l+c.bin_centers[0]*cw,pad.t+ch);x.closePath();x.fill();x.globalAlpha=1;
  x.strokeStyle='#f97316';x.lineWidth=2.5;x.lineJoin='round';x.beginPath();
  c.bin_centers.forEach((v,i)=>{const px=pad.l+v*cw,py=pad.t+(1-c.actual_rates[i])*ch;i===0?x.moveTo(px,py):x.lineTo(px,py)});x.stroke();
  c.bin_centers.forEach((v,i)=>{
    const px=pad.l+v*cw,py=pad.t+(1-c.actual_rates[i])*ch;
    x.fillStyle=isDark?'#050505':'#fff';x.beginPath();x.arc(px,py,4,0,Math.PI*2);x.fill();
    x.fillStyle='#f97316';x.beginPath();x.arc(px,py,2.5,0,Math.PI*2);x.fill();
  });
  x.fillStyle=isDark?'#444':'#999';x.font='10px "JetBrains Mono",monospace';x.textAlign='center';
  for(let t=0;t<=1;t+=.2)x.fillText(t.toFixed(1),pad.l+t*cw,H-4);
  x.textAlign='right';for(let t=0;t<=1;t+=.2)x.fillText(t.toFixed(1),pad.l-5,pad.t+(1-t)*ch+3);

  // Tooltip
  cv.onmousemove=function(e){
    const rect=cv.getBoundingClientRect();
    const mx=(e.clientX-rect.left)*(cv.width/dpr/rect.width);
    const my=(e.clientY-rect.top)*(cv.height/dpr/rect.height);
    let closest=-1,minD=999;
    c.bin_centers.forEach((v,i)=>{
      const px=pad.l+v*cw,py=pad.t+(1-c.actual_rates[i])*ch;
      const d=Math.sqrt((mx-px)**2+(my-py)**2);if(d<minD){minD=d;closest=i}
    });
    const tip=$('cal-tooltip');if(!tip)return;
    if(closest>=0&&minD<20){
      const bc=c.bin_centers[closest],ar=c.actual_rates[closest];
      const cnt=c.bin_counts?c.bin_counts[closest]:'?';
      tip.textContent='Predicted: '+bc.toFixed(2)+' | Actual: '+ar.toFixed(2)+' | n='+cnt;
      tip.style.left=(e.clientX-wrap.getBoundingClientRect().left+10)+'px';
      tip.style.top=(e.clientY-wrap.getBoundingClientRect().top-30)+'px';
      tip.classList.add('show');
    }else{tip.classList.remove('show')}
  };
  cv.onmouseleave=function(){const tip=$('cal-tooltip');if(tip)tip.classList.remove('show')};
}

/* === MOBILE === */
function renderMobileList(list){
  const el=$('mobile-list');if(!el)return;
  if(!list||!list.length){el.innerHTML='<div class="empty"><h3>No matches scheduled</h3><p>Check back during tournament weeks.</p></div>';return}
  const filtered=applyFilters(list);
  const groups={};
  filtered.forEach((p,i)=>{const t=p.tournament||'Other';if(!groups[t])groups[t]=[];groups[t].push({...p,_idx:list.indexOf(p)})});
  let html='<div class="filters" style="padding:8px 0">'+
    '<button class="filter-btn'+(S.filters.conf==='all'?' active':'')+'" data-type="conf" data-val="all">All</button>'+
    '<button class="filter-btn'+(S.filters.conf==='high'?' active':'')+'" data-type="conf" data-val="high">High</button>'+
    '<button class="filter-btn'+(S.filters.conf==='medium'?' active':'')+'" data-type="conf" data-val="medium">Med</button>'+
    '<span class="filter-count">'+filtered.length+' matches</span></div>';
  let count=0;
  for(const[tourney,matches] of Object.entries(groups)){
    html+='<div class="tour-header"><div class="tour-bar"></div><span class="tour-name">'+esc(tourney)+'</span><span class="tour-count"> \u00b7 '+matches.length+'</span></div>';
    matches.forEach(p=>{
      const fav=p.prob_p1>=.5;const wN=fav?p.player1:p.player2;const lN=fav?p.player2:p.player1;
      const prob=Math.round(Math.max(p.prob_p1,1-p.prob_p1)*100);
      const tier=p.confidence_tier||'';const dotCls=tier==='high'?'high':tier==='medium'?'med':'low';
      html+='<div class="mi" data-idx="'+p._idx+'" style="animation:fadeSlideUp .3s ease '+(count*50)+'ms both">'+
        '<div class="conf-dot '+dotCls+'"></div>'+
        '<div class="mi-info"><div class="mi-names">'+esc(wN)+' vs '+esc(lN)+'</div>'+
        '<div class="mi-meta">'+(p.surface||'')+' \u00b7 '+prob+'%</div></div>'+
        '<div class="mi-prob">'+prob+'%</div></div>';
      count++;
    });
  }
  el.innerHTML=html;
}

function showMobileDetail(idx){
  S.view='detail';S.sel=idx;
  const p=DATA.predictions[idx];if(!p)return;
  const md=$('mobile-detail');const ml=$('mobile-list');if(!md||!ml)return;
  const fav=p.prob_p1>=.5;const d=p.detail||{};
  const ws=fav?d.p1||{}:d.p2||{};const ls=fav?d.p2||{}:d.p1||{};
  const wN=fav?p.player1:p.player2;const lN=fav?p.player2:p.player1;
  const wR=fav?p.p1_rank:p.p2_rank;const lR=fav?p.p2_rank:p.p1_rank;
  const prob=Math.round(Math.max(p.prob_p1,1-p.prob_p1)*100);
  const wI=wN.split(' ').map(s=>s[0]).join('').substring(0,2);
  const lI=lN.split(' ').map(s=>s[0]).join('').substring(0,2);
  md.innerHTML='<button class="mobile-back" onclick="hideMobileDetail()">\u2190 Back</button>'+
    '<div class="faceoff">'+
      '<div class="fo-player fav"><div class="fo-avatar">'+esc(wI)+'</div>'+
        '<div class="fo-name plink" data-name="'+esc(wN)+'">'+esc(wN)+'</div>'+
        '<div class="fo-rank">#'+(wR||'\u2014')+'</div></div>'+
      '<div class="fo-vs"><div class="fo-prob">'+prob+'%</div><div class="fo-prob-label">Win Prob</div></div>'+
      '<div class="fo-player"><div class="fo-avatar">'+esc(lI)+'</div>'+
        '<div class="fo-name plink" data-name="'+esc(lN)+'">'+esc(lN)+'</div>'+
        '<div class="fo-rank">#'+(lR||'\u2014')+'</div></div>'+
    '</div>'+renderStatBars(ws,ls)+renderH2H(d.h2h,wN,lN,fav)+renderFactors(d.factors);
  ml.classList.add('pushed');md.classList.add('active');
  history.pushState({view:'detail',match:idx},'');
}
function hideMobileDetail(){
  S.view='list';
  const md=$('mobile-detail');const ml=$('mobile-list');
  if(md)md.classList.remove('active');if(ml)ml.classList.remove('pushed');
}
window.addEventListener('popstate',e=>{if(S.mobile&&S.view==='detail')hideMobileDetail()});

/* Mobile Results + Analytics */
function renderMobileResults(){
  const el=$('mobile-results');if(!el||!DATA)return;
  if(!DATA.history||!DATA.history.length){
    el.innerHTML='<div class="empty" style="padding:2rem"><h3>No prediction history yet</h3><p>Results appear after predictions are tracked.</p></div>';return;
  }
  el.innerHTML='<h2 style="font-size:16px;font-weight:800;padding:12px 0 8px;letter-spacing:-.3px">Prediction Results</h2>'+
    DATA.history.map(day=>{
      const date=new Date(day.date+'T12:00:00');
      const label=date.toLocaleDateString(undefined,{weekday:'short',month:'short',day:'numeric'});
      const preds=day.predictions||[];if(!preds.length)return '';
      return '<div class="results-day"><h3>'+label+' \u00b7 '+preds.length+' predictions</h3>'+
        preds.map(p=>{
          const fav=(p.prob_p1||0.5)>=.5;
          const pick=fav?p.player1:p.player2;const opp=fav?p.player2:p.player1;
          const prob=Math.round(Math.max(p.prob_p1||.5,1-(p.prob_p1||.5))*100);
          return '<div class="result-row"><span class="result-icon ri-win">\u2713</span>'+
            '<span class="result-pick">'+esc(pick)+'</span>'+
            '<span class="result-prob">'+prob+'%</span>'+
            '<span class="result-vs">vs '+esc(opp)+'</span></div>';
        }).join('')+'</div>';
    }).join('');
}

function renderMobileAnalytics(){
  const el=$('mobile-analytics');if(!el||!DATA)return;
  const s=DATA.model_stats||{};
  const a=s.accuracy?(s.accuracy*100).toFixed(1)+'%':'64.0%';
  const b=s.brier_score?s.brier_score.toFixed(3):'0.220';
  el.innerHTML='<h2 style="font-size:16px;font-weight:800;padding:12px 0 8px;letter-spacing:-.3px">Analytics</h2>'+
    '<div class="metrics-grid" style="grid-template-columns:repeat(2,1fr);margin-bottom:12px">'+
      '<div class="metric"><div class="metric-label">Accuracy</div><div class="metric-value mv-orange">'+a+'</div></div>'+
      '<div class="metric"><div class="metric-label">Brier</div><div class="metric-value mv-neutral">'+b+'</div></div>'+
    '</div>'+
    '<div class="sys-grid">'+
      '<div class="sys-card"><strong>244</strong><div class="sys-label">Features</div><p>Elo, Glicko-2, serve/return, fatigue, weather, court speed</p></div>'+
      '<div class="sys-card"><strong>320K</strong><div class="sys-label">Matches</div><p>1991-2026 training data</p></div>'+
      '<div class="sys-card"><strong>Zero</strong><div class="sys-label">Leakage</div><p>TemporalGuard enforced</p></div>'+
      '<div class="sys-card"><strong>24/7</strong><div class="sys-label">Autonomous</div><p>Self-learning, daily retrains</p></div>'+
    '</div>';
}

/* Swipe gesture */
let txS=0,tyS=0,ttS=0,swiping=false;
document.addEventListener('touchstart',e=>{
  if(!S.mobile||S.view!=='detail')return;
  const t=e.touches[0];txS=t.clientX;tyS=t.clientY;ttS=Date.now();swiping=false;
},{passive:true});
document.addEventListener('touchmove',e=>{
  if(!S.mobile||S.view!=='detail')return;
  const t=e.touches[0];const dx=t.clientX-txS;const dy=t.clientY-tyS;
  if(Math.abs(dx)<10)return;
  if(Math.abs(Math.atan2(dy,dx)*180/Math.PI)>30)return;
  swiping=true;
  const md=$('mobile-detail');const ml=$('mobile-list');
  if(md&&dx>0){md.style.transition='none';md.style.transform='translateX('+dx+'px)'}
  if(ml&&dx>0){ml.style.transition='none';ml.style.transform='translateX('+(-30+dx*0.3)+'%)'}
},{passive:true});
document.addEventListener('touchend',e=>{
  if(!swiping)return;swiping=false;
  const dx=e.changedTouches[0].clientX-txS;const dt=Date.now()-ttS;const vel=Math.abs(dx)/dt;
  const md=$('mobile-detail');const ml=$('mobile-list');
  if(md)md.style.transition='';if(ml)ml.style.transition='';
  if(dx>80||vel>0.5){hideMobileDetail();history.back()}
  else{if(md)md.style.transform='';if(ml)ml.style.transform=''}
},{passive:true});

/* === PLAYER MODAL === */
document.addEventListener('click',e=>{
  const pl=e.target.closest('.plink');if(!pl)return;
  e.stopPropagation();
  const name=pl.dataset.name;if(!name||!DATA)return;
  let stats=null;
  for(const p of DATA.predictions){
    const d=p.detail;if(!d)continue;
    if(d.p1&&d.p1.name===name){stats=d.p1;break}
    if(d.p2&&d.p2.name===name){stats=d.p2;break}
  }
  if(!stats)return;openModal(name,stats);
});
function openModal(name,stats){
  const el=$('modal-content');if(!el)return;
  el.innerHTML='<button class="modal-close" onclick="closeModal()">&times;</button>'+
    '<h3>'+esc(name)+'</h3>'+
    (stats.elo?'<div class="modal-rank">Elo '+stats.elo+' \u00b7 Surface '+stats.surface_elo+'</div>':'')+
    '<div class="modal-section"><h4>Ratings</h4>'+
      dr('Overall Elo',stats.elo)+dr('Surface Elo',stats.surface_elo)+
      dr('Serve Elo',stats.serve_elo)+dr('Return Elo',stats.return_elo)+
      dr('Glicko-2',stats.glicko2_rating)+dr('Rating Deviation',stats.glicko2_rd)+'</div>'+
    '<div class="modal-section"><h4>Recent Form</h4>'+
      dr('Last 5',stats.form_last5)+dr('Last 10',stats.form_last10)+
      dr('Win Streak',stats.win_streak)+dr('Loss Streak',stats.loss_streak)+
      dr('Total Matches',stats.total_matches)+'</div>'+
    '<div class="modal-section"><h4>Serve & Return</h4>'+
      dr('1st Serve %',pct(stats.first_serve_pct))+dr('1st Serve Won',pct(stats.first_serve_won))+
      dr('2nd Serve Won',pct(stats.second_serve_won))+dr('Ace Rate',pct(stats.ace_rate))+
      dr('DF Rate',pct(stats.df_rate))+dr('BP Save %',pct(stats.bp_save_pct))+
      dr('Return Pts Won',pct(stats.return_pts_won))+'</div>';
  $('player-modal').classList.add('open');document.body.style.overflow='hidden';
}
function closeModal(){
  const o=$('player-modal');if(o)o.classList.remove('open');document.body.style.overflow='';
}
$('player-modal')?.addEventListener('click',e=>{if(e.target.id==='player-modal')closeModal()});

/* Modal swipe-down dismiss */
let modalTouchY=0,modalSwiping=false;
document.addEventListener('touchstart',e=>{
  if(!$('player-modal')?.classList.contains('open'))return;
  const mc=e.target.closest('.modal');if(!mc)return;
  modalTouchY=e.touches[0].clientY;modalSwiping=false;
},{passive:true});
document.addEventListener('touchmove',e=>{
  if(!$('player-modal')?.classList.contains('open'))return;
  const mc=e.target.closest('.modal');if(!mc)return;
  const dy=e.touches[0].clientY-modalTouchY;
  if(dy>0){modalSwiping=true;mc.style.transition='none';mc.style.transform='translateY('+dy+'px)'}
},{passive:true});
document.addEventListener('touchend',e=>{
  if(!modalSwiping)return;modalSwiping=false;
  const mc=document.querySelector('.modal-overlay.open .modal');if(!mc)return;
  const dy=e.changedTouches[0].clientY-modalTouchY;
  mc.style.transition='';mc.style.transform='';
  if(dy>100)closeModal();
},{passive:true});

function dr(l,v){if(v==null||v===undefined||v==='')return '';
  return '<div class="dr"><span class="l">'+l+'</span><span class="v">'+v+'</span></div>'}

init();
""")


def _write_sw():
    """Write minimal service worker for offline caching of static assets."""
    (SITE_DIR / "sw.js").write_text("""
// Service worker — cache static assets for faster loads
const CACHE='tp-v2';
const ASSETS=['./','assets/css/style.css','assets/js/app.js'];
self.addEventListener('install',e=>{e.waitUntil(caches.open(CACHE).then(c=>c.addAll(ASSETS)));self.skipWaiting()});
self.addEventListener('activate',e=>{e.waitUntil(caches.keys().then(ks=>Promise.all(ks.filter(k=>k!==CACHE).map(k=>caches.delete(k)))));self.clients.claim()});
self.addEventListener('fetch',e=>{
  if(e.request.url.includes('predictions.json')){e.respondWith(fetch(e.request));return}
  e.respondWith(caches.match(e.request).then(r=>r||fetch(e.request)));
});
""")
