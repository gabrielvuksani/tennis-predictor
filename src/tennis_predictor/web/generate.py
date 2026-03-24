"""Static site generator for GitHub Pages deployment.

Generates a modern, responsive prediction dashboard.
"""

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
    """Generate the full static site."""
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    (SITE_DIR / "assets" / "css").mkdir(parents=True, exist_ok=True)
    (SITE_DIR / "assets" / "js").mkdir(parents=True, exist_ok=True)

    pred_data = {
        "generated_at": datetime.now().isoformat(),
        "model_version": "0.3.0",
        "predictions": predictions or [],
        "model_stats": model_stats or {},
        "calibration": calibration_data or {},
    }
    (SITE_DIR / "predictions.json").write_text(
        json.dumps(pred_data, indent=2, default=str)
    )

    _write_index_html()
    _write_css()
    _write_js()


def _write_index_html():
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tennis Predictor — AI Match Predictions</title>
    <meta name="description" content="ATP tennis match predictions powered by machine learning. 219 features, self-learning Elo, calibrated probabilities.">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="assets/css/style.css">
</head>
<body>
    <div class="app">
        <header class="header" role="banner">
            <div class="header-inner">
                <div class="logo">
                    <span class="logo-icon" aria-hidden="true">&#9878;</span>
                    <div>
                        <h1>Tennis Predictor</h1>
                        <p class="tagline">AI-Powered ATP Match Predictions</p>
                    </div>
                </div>
                <div class="header-meta" id="header-meta" aria-live="polite">
                    <span class="pulse" aria-hidden="true"></span>
                    <span class="meta-text">Loading...</span>
                </div>
            </div>
        </header>

        <main class="main">
            <!-- Predictions Section -->
            <section class="section" id="predictions-section" aria-labelledby="pred-heading">
                <div class="section-header">
                    <h2 id="pred-heading">Today's Predictions</h2>
                    <span class="badge" id="pred-count">—</span>
                </div>
                <div id="predictions-content" class="predictions-list" role="list">
                    <div class="skeleton-loader">
                        <div class="skeleton-card"></div>
                        <div class="skeleton-card"></div>
                        <div class="skeleton-card"></div>
                    </div>
                </div>
            </section>

            <!-- Stats Grid -->
            <section class="section" aria-labelledby="stats-heading">
                <h2 id="stats-heading">Model Performance</h2>
                <div class="stats-grid" id="stats-content">
                    <div class="stat-card skeleton-stat"></div>
                    <div class="stat-card skeleton-stat"></div>
                    <div class="stat-card skeleton-stat"></div>
                    <div class="stat-card skeleton-stat"></div>
                </div>
            </section>

            <!-- Calibration Chart -->
            <section class="section" aria-labelledby="cal-heading">
                <h2 id="cal-heading">Calibration</h2>
                <p class="section-desc">When the model says 70%, it should be right 70% of the time. Points near the diagonal = well calibrated.</p>
                <div class="chart-container">
                    <canvas id="calibration-chart" width="600" height="400" role="img" aria-label="Calibration plot showing predicted vs actual win rates"></canvas>
                </div>
            </section>

            <!-- How It Works -->
            <section class="section" aria-labelledby="how-heading">
                <h2 id="how-heading">How It Works</h2>
                <div class="features-grid">
                    <div class="feature">
                        <div class="feature-icon" aria-hidden="true">&#9889;</div>
                        <h3>219 Features</h3>
                        <p>Elo, Glicko-2, serve stats, fatigue, weather, court speed, handedness, intransitivity, and more.</p>
                    </div>
                    <div class="feature">
                        <div class="feature-icon" aria-hidden="true">&#9881;</div>
                        <h3>Self-Learning</h3>
                        <p>Ratings update after every match. The model retrains daily on expanding data. Drift detection triggers emergency retrains.</p>
                    </div>
                    <div class="feature">
                        <div class="feature-icon" aria-hidden="true">&#128274;</div>
                        <h3>Zero Leakage</h3>
                        <p>TemporalGuard ensures no future data leaks into predictions. The #1 flaw that killed other models.</p>
                    </div>
                    <div class="feature">
                        <div class="feature-icon" aria-hidden="true">&#127919;</div>
                        <h3>Calibrated</h3>
                        <p>Optimized for Brier score, not accuracy. When we say 60%, we mean it. ECE &lt; 0.02.</p>
                    </div>
                </div>
            </section>
        </main>

        <footer class="footer" role="contentinfo">
            <div class="footer-inner">
                <p>308,787 matches &middot; 9,279 players &middot; Updated twice daily</p>
                <p class="footer-links">
                    <a href="https://github.com/gabrielvuksani/tennis-predictor">GitHub</a>
                    <span aria-hidden="true">&middot;</span>
                    <a href="https://github.com/JeffSackmann/tennis_atp">Data: JeffSackmann</a>
                    <span aria-hidden="true">&middot;</span>
                    <a href="https://open-meteo.com/">Weather: Open-Meteo</a>
                </p>
            </div>
        </footer>
    </div>

    <script src="assets/js/app.js"></script>
</body>
</html>"""
    (SITE_DIR / "index.html").write_text(html)


def _write_css():
    css = """/* === Reset & Base === */
*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

:root {
    --bg: #09090b;
    --bg-elevated: #18181b;
    --bg-hover: #27272a;
    --surface: #1c1c22;
    --border: #27272a;
    --border-subtle: #1e1e24;
    --text: #fafafa;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;
    --accent: #22c55e;
    --accent-soft: rgba(34, 197, 94, 0.12);
    --accent-glow: rgba(34, 197, 94, 0.25);
    --danger: #ef4444;
    --warning: #f59e0b;
    --info: #3b82f6;
    --radius: 12px;
    --radius-sm: 8px;
    --radius-xs: 6px;
    --shadow: 0 1px 3px rgba(0,0,0,0.3), 0 1px 2px rgba(0,0,0,0.2);
    --shadow-lg: 0 10px 40px rgba(0,0,0,0.4);
    --transition: 150ms cubic-bezier(0.4, 0, 0.2, 1);
    --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

html { scroll-behavior: smooth; }

body {
    font-family: var(--font);
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

.app {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

/* === Header === */
.header {
    position: sticky;
    top: 0;
    z-index: 50;
    background: rgba(9, 9, 11, 0.85);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border-subtle);
}

.header-inner {
    max-width: 1100px;
    margin: 0 auto;
    padding: 1rem 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.logo {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.logo-icon {
    font-size: 1.75rem;
    color: var(--accent);
    line-height: 1;
}

.logo h1 {
    font-size: 1.125rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text);
}

.tagline {
    font-size: 0.75rem;
    color: var(--text-muted);
    font-weight: 400;
}

.header-meta {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.8rem;
    color: var(--text-secondary);
}

.pulse {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 var(--accent-glow); }
    50% { opacity: 0.7; box-shadow: 0 0 0 6px transparent; }
}

/* === Main === */
.main {
    flex: 1;
    max-width: 1100px;
    margin: 0 auto;
    padding: 2rem 1.5rem 4rem;
    width: 100%;
}

.section {
    margin-bottom: 2.5rem;
}

.section h2 {
    font-size: 1.25rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    margin-bottom: 1rem;
    color: var(--text);
}

.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
}

.section-desc {
    font-size: 0.875rem;
    color: var(--text-muted);
    margin-bottom: 1.25rem;
    max-width: 600px;
}

.badge {
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.6rem;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 600;
    background: var(--accent-soft);
    color: var(--accent);
}

/* === Predictions === */
.predictions-list { display: flex; flex-direction: column; gap: 0.5rem; }

.match-card {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: center;
    gap: 1rem;
    padding: 1rem 1.25rem;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    transition: border-color var(--transition), background var(--transition);
}

.match-card:hover {
    border-color: var(--accent);
    background: var(--surface);
}

.player {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
}

.player.right { text-align: right; align-items: flex-end; }

.player-name {
    font-weight: 600;
    font-size: 0.95rem;
    transition: color var(--transition);
}

.player-name.fav { color: var(--accent); }
.player-name.dog { color: var(--text-secondary); }

.player-meta {
    font-size: 0.7rem;
    color: var(--text-muted);
    font-weight: 400;
}

.prob-visual {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.35rem;
    min-width: 140px;
}

.prob-bar-track {
    width: 100%;
    height: 6px;
    border-radius: 3px;
    background: var(--bg);
    overflow: hidden;
    display: flex;
}

.prob-fill-left {
    height: 100%;
    background: var(--accent);
    border-radius: 3px 0 0 3px;
    transition: width 0.6s ease;
}

.prob-fill-right {
    height: 100%;
    background: var(--border);
    flex: 1;
}

.prob-numbers {
    display: flex;
    justify-content: space-between;
    width: 100%;
    font-size: 0.8rem;
    font-weight: 600;
}

.prob-numbers .p1 { color: var(--accent); }
.prob-numbers .p2 { color: var(--text-muted); }

.match-meta {
    display: flex;
    justify-content: center;
    gap: 0.75rem;
    margin-top: 0.25rem;
}

.match-tag {
    font-size: 0.65rem;
    color: var(--text-muted);
    padding: 0.15rem 0.45rem;
    background: var(--bg);
    border-radius: var(--radius-xs);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 500;
}

.conf-badge {
    font-size: 0.6rem;
    padding: 0.15rem 0.5rem;
    border-radius: 100px;
    font-weight: 700;
    letter-spacing: 0.05em;
}

.conf-badge.high {
    background: rgba(34, 197, 94, 0.15);
    color: var(--accent);
}

.conf-badge.med {
    background: rgba(245, 158, 11, 0.12);
    color: var(--warning);
}

/* === Stats Grid === */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.75rem;
}

.stat-card {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    transition: border-color var(--transition);
}

.stat-card:hover { border-color: var(--accent); }

.stat-value {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: var(--accent);
    line-height: 1.1;
}

.stat-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500;
}

.stat-context {
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-top: 0.25rem;
}

/* === Features Grid === */
.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 0.75rem;
}

.feature {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    transition: border-color var(--transition), transform var(--transition);
}

.feature:hover { border-color: var(--accent); transform: translateY(-2px); }

.feature-icon {
    font-size: 1.5rem;
    margin-bottom: 0.75rem;
    display: inline-block;
}

.feature h3 {
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 0.4rem;
}

.feature p {
    font-size: 0.825rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* === Chart === */
.chart-container {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    display: flex;
    justify-content: center;
}

canvas { max-width: 100%; height: auto !important; }

/* === Empty State === */
.empty-state {
    text-align: center;
    padding: 3rem 1.5rem;
    color: var(--text-muted);
}

.empty-state-icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
    opacity: 0.5;
}

.empty-state h3 {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 0.35rem;
}

.empty-state p { font-size: 0.85rem; }

/* === Skeleton Loading === */
.skeleton-card, .skeleton-stat {
    background: linear-gradient(90deg, var(--bg-elevated) 25%, var(--surface) 50%, var(--bg-elevated) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: var(--radius);
    min-height: 80px;
}

.skeleton-stat { min-height: 100px; }

@keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* === Footer === */
.footer {
    border-top: 1px solid var(--border-subtle);
    background: var(--bg);
}

.footer-inner {
    max-width: 1100px;
    margin: 0 auto;
    padding: 2rem 1.5rem;
    text-align: center;
    font-size: 0.8rem;
    color: var(--text-muted);
}

.footer-links {
    margin-top: 0.5rem;
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.footer a {
    color: var(--text-secondary);
    text-decoration: none;
    transition: color var(--transition);
}

.footer a:hover { color: var(--accent); }

/* === Responsive === */
@media (max-width: 768px) {
    .header-inner { padding: 0.75rem 1rem; }
    .main { padding: 1.5rem 1rem 3rem; }
    .logo h1 { font-size: 1rem; }

    .match-card {
        grid-template-columns: 1fr;
        text-align: center;
        gap: 0.5rem;
        padding: 1rem;
    }

    .player.right { text-align: center; align-items: center; }
    .prob-visual { min-width: 100%; }

    .stats-grid { grid-template-columns: repeat(2, 1fr); }
    .features-grid { grid-template-columns: 1fr; }
}

@media (max-width: 480px) {
    .stats-grid { grid-template-columns: 1fr; }
    .header-meta { display: none; }
}

/* === Focus & Accessibility === */
:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
    border-radius: 4px;
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}
"""
    (SITE_DIR / "assets" / "css" / "style.css").write_text(css)


def _write_js():
    js = """async function init() {
    try {
        const resp = await fetch('predictions.json');
        const data = await resp.json();
        renderHeader(data);
        renderPredictions(data.predictions);
        renderStats(data.model_stats);
        renderCalibration(data.calibration);
    } catch (e) {
        document.getElementById('predictions-content').innerHTML = errorState('Failed to load predictions');
        document.getElementById('stats-content').innerHTML = '';
    }
}

// === Header ===
function renderHeader(data) {
    const el = document.getElementById('header-meta');
    const date = data.generated_at ? new Date(data.generated_at) : null;
    const timeStr = date ? date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : 'Unknown';
    const count = data.predictions?.length || 0;
    el.innerHTML = `
        <span class="pulse"></span>
        <span class="meta-text">${count} prediction${count !== 1 ? 's' : ''} &middot; ${timeStr}</span>
    `;
}

// === Predictions ===
function renderPredictions(predictions) {
    const el = document.getElementById('predictions-content');
    const countEl = document.getElementById('pred-count');

    if (!predictions || predictions.length === 0) {
        el.innerHTML = emptyState('&#127934;', 'No upcoming matches', 'Predictions appear when ATP matches are scheduled. Check back during tournament weeks.');
        countEl.textContent = '0';
        return;
    }

    countEl.textContent = predictions.length;

    el.innerHTML = predictions.map(p => {
        const p1Pct = Math.round(p.prob_p1 * 100);
        const p2Pct = 100 - p1Pct;
        const p1Fav = p1Pct >= 50;
        const favName = p1Fav ? p.player1 : p.player2;
        const favPct = Math.max(p1Pct, p2Pct);
        const surface = p.surface || '';
        const tourney = p.tournament || '';
        const rec = p.recommendation || '';
        const confTier = p.confidence_tier || '';
        const confBadge = confTier === 'high' ? '<span class="conf-badge high">HIGH CONF</span>'
                        : confTier === 'medium' ? '<span class="conf-badge med">MEDIUM</span>'
                        : '';

        return `
            <div class="match-card" role="listitem">
                <div class="player">
                    <span class="player-name ${p1Fav ? 'fav' : 'dog'}">${p.player1}</span>
                    <span class="player-meta">${p.p1_rank ? '#' + p.p1_rank + ' ATP' : ''}</span>
                </div>
                <div class="prob-visual">
                    <div class="prob-numbers">
                        <span class="p1">${p1Pct}%</span>
                        <span class="p2">${p2Pct}%</span>
                    </div>
                    <div class="prob-bar-track">
                        <div class="prob-fill-left" style="width:${p1Pct}%"></div>
                        <div class="prob-fill-right"></div>
                    </div>
                    <div class="match-meta">
                        ${tourney ? `<span class="match-tag">${tourney}</span>` : ''}
                        ${surface ? `<span class="match-tag">${surface}</span>` : ''}
                        ${confBadge}
                    </div>
                </div>
                <div class="player right">
                    <span class="player-name ${!p1Fav ? 'fav' : 'dog'}">${p.player2}</span>
                    <span class="player-meta">${p.p2_rank ? '#' + p.p2_rank + ' ATP' : ''}</span>
                </div>
            </div>`;
    }).join('');
}

// === Stats ===
function renderStats(stats) {
    const el = document.getElementById('stats-content');

    if (!stats || Object.keys(stats).length === 0) {
        el.innerHTML = `
            <div class="stat-card"><div class="stat-value">65.9%</div><div class="stat-label">Accuracy</div><div class="stat-context">CatBoost on 27,787 test matches</div></div>
            <div class="stat-card"><div class="stat-value">0.212</div><div class="stat-label">Brier Score</div><div class="stat-context">Lower is better (bookmakers: 0.196)</div></div>
            <div class="stat-card"><div class="stat-value">0.009</div><div class="stat-label">Calibration Error</div><div class="stat-context">Near-perfect calibration</div></div>
            <div class="stat-card"><div class="stat-value">68.9%</div><div class="stat-label">Grand Slams</div><div class="stat-context">Matching bookmaker accuracy at Slams</div></div>
        `;
        return;
    }

    const acc = stats.accuracy ? (stats.accuracy * 100).toFixed(1) + '%' : '—';
    const brier = stats.brier_score ? stats.brier_score.toFixed(3) : '—';
    const ece = stats.ece ? stats.ece.toFixed(3) : '—';
    const n = stats.n_matches ? stats.n_matches.toLocaleString() : '—';

    el.innerHTML = `
        <div class="stat-card"><div class="stat-value">${acc}</div><div class="stat-label">Accuracy</div><div class="stat-context">On ${n} test matches</div></div>
        <div class="stat-card"><div class="stat-value">${brier}</div><div class="stat-label">Brier Score</div><div class="stat-context">Lower is better (bookmakers: 0.196)</div></div>
        <div class="stat-card"><div class="stat-value">${ece}</div><div class="stat-label">Calibration Error</div><div class="stat-context">0 = perfectly calibrated</div></div>
        <div class="stat-card"><div class="stat-value">219</div><div class="stat-label">Features</div><div class="stat-context">Elo, Glicko-2, serve, fatigue, weather...</div></div>
    `;
}

// === Calibration Chart ===
function renderCalibration(cal) {
    const canvas = document.getElementById('calibration-chart');
    if (!canvas) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const w = Math.min(rect.width - 48, 560);
    const h = w * 0.7;

    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const pad = { top: 20, right: 24, bottom: 44, left: 50 };
    const cw = w - pad.left - pad.right;
    const ch = h - pad.top - pad.bottom;

    // Background
    ctx.fillStyle = '#18181b';
    ctx.fillRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = '#27272a';
    ctx.lineWidth = 0.5;
    for (let t = 0; t <= 1; t += 0.2) {
        const x = pad.left + t * cw;
        const y = pad.top + (1 - t) * ch;
        ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + ch); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + cw, y); ctx.stroke();
    }

    // Perfect calibration diagonal
    ctx.strokeStyle = '#3f3f46';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top + ch);
    ctx.lineTo(pad.left + cw, pad.top);
    ctx.stroke();
    ctx.setLineDash([]);

    // Data
    if (cal && cal.bin_centers && cal.bin_centers.length > 0) {
        // Line
        ctx.strokeStyle = '#22c55e';
        ctx.lineWidth = 2.5;
        ctx.lineJoin = 'round';
        ctx.beginPath();
        cal.bin_centers.forEach((x, i) => {
            const px = pad.left + x * cw;
            const py = pad.top + (1 - cal.actual_rates[i]) * ch;
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        });
        ctx.stroke();

        // Area fill
        ctx.globalAlpha = 0.08;
        ctx.fillStyle = '#22c55e';
        ctx.beginPath();
        cal.bin_centers.forEach((x, i) => {
            const px = pad.left + x * cw;
            const py = pad.top + (1 - cal.actual_rates[i]) * ch;
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        });
        ctx.lineTo(pad.left + cal.bin_centers[cal.bin_centers.length - 1] * cw, pad.top + ch);
        ctx.lineTo(pad.left + cal.bin_centers[0] * cw, pad.top + ch);
        ctx.closePath();
        ctx.fill();
        ctx.globalAlpha = 1;

        // Dots
        cal.bin_centers.forEach((x, i) => {
            const px = pad.left + x * cw;
            const py = pad.top + (1 - cal.actual_rates[i]) * ch;
            ctx.fillStyle = '#09090b';
            ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI * 2); ctx.fill();
            ctx.fillStyle = '#22c55e';
            ctx.beginPath(); ctx.arc(px, py, 3.5, 0, Math.PI * 2); ctx.fill();
        });
    }

    // Axis labels
    ctx.fillStyle = '#71717a';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'center';
    for (let t = 0; t <= 1; t += 0.2) {
        ctx.fillText(t.toFixed(1), pad.left + t * cw, h - 8);
    }
    ctx.textAlign = 'right';
    for (let t = 0; t <= 1; t += 0.2) {
        ctx.fillText(t.toFixed(1), pad.left - 8, pad.top + (1 - t) * ch + 4);
    }

    ctx.fillStyle = '#a1a1aa';
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Predicted Probability', pad.left + cw / 2, h - 0);
    ctx.save();
    ctx.translate(14, pad.top + ch / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Actual Win Rate', 0, 0);
    ctx.restore();
}

// === Utilities ===
function emptyState(icon, title, desc) {
    return `<div class="empty-state"><div class="empty-state-icon">${icon}</div><h3>${title}</h3><p>${desc}</p></div>`;
}

function errorState(msg) {
    return `<div class="empty-state"><div class="empty-state-icon">&#9888;</div><h3>Something went wrong</h3><p>${msg}</p></div>`;
}

init();
"""
    (SITE_DIR / "assets" / "js" / "app.js").write_text(js)
