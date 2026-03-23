"""Static site generator for GitHub Pages deployment.

Generates an interactive prediction dashboard as a static HTML site.
Predictions are pre-computed and embedded as JSON.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

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

    # Generate predictions JSON
    pred_data = {
        "generated_at": datetime.now().isoformat(),
        "model_version": "0.1.0",
        "predictions": predictions or [],
        "model_stats": model_stats or {},
        "calibration": calibration_data or {},
    }
    (SITE_DIR / "predictions.json").write_text(
        json.dumps(pred_data, indent=2, default=str)
    )

    # Generate HTML
    _write_index_html()
    _write_css()
    _write_js()


def _write_index_html():
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tennis Predictor</title>
    <link rel="stylesheet" href="assets/css/style.css">
</head>
<body>
    <header>
        <h1>Tennis Predictor</h1>
        <p class="subtitle">ATP Match Predictions &mdash; Calibrated Probabilities</p>
    </header>

    <main>
        <section id="status" class="card">
            <h2>System Status</h2>
            <div id="status-content">Loading...</div>
        </section>

        <section id="predictions" class="card">
            <h2>Upcoming Predictions</h2>
            <div id="predictions-content">Loading...</div>
        </section>

        <section id="performance" class="card">
            <h2>Model Performance</h2>
            <div class="metrics-grid" id="metrics-content">Loading...</div>
        </section>

        <section id="calibration" class="card">
            <h2>Calibration Plot</h2>
            <p class="desc">A calibrated model's predicted probabilities match actual outcomes.
            The diagonal line represents perfect calibration.</p>
            <canvas id="calibration-chart" width="500" height="400"></canvas>
        </section>

        <section id="recent" class="card">
            <h2>Recent Results</h2>
            <div id="recent-content">Loading...</div>
        </section>
    </main>

    <footer>
        <p>Built with temporal integrity. No data leakage. Calibration over accuracy.</p>
        <p>Data: <a href="https://github.com/JeffSackmann/tennis_atp">JeffSackmann/tennis_atp</a>
         | Weather: <a href="https://open-meteo.com/">Open-Meteo</a>
         | Court Speed: <a href="https://courtspeed.com/">CourtSpeed.com</a></p>
    </footer>

    <script src="assets/js/app.js"></script>
</body>
</html>"""
    (SITE_DIR / "index.html").write_text(html)


def _write_css():
    css = """* { margin: 0; padding: 0; box-sizing: border-box; }

:root {
    --bg: #0a0e17;
    --card-bg: #111827;
    --text: #e5e7eb;
    --text-muted: #9ca3af;
    --accent: #10b981;
    --accent-dim: #065f46;
    --danger: #ef4444;
    --border: #1f2937;
}

body {
    font-family: 'SF Mono', 'Fira Code', monospace;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 2rem;
    max-width: 1200px;
    margin: 0 auto;
}

header {
    text-align: center;
    margin-bottom: 2rem;
    padding: 2rem;
    border-bottom: 1px solid var(--border);
}

header h1 {
    font-size: 2rem;
    color: var(--accent);
    letter-spacing: 2px;
    text-transform: uppercase;
}

.subtitle {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-top: 0.5rem;
}

.card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.card h2 {
    color: var(--accent);
    font-size: 1.1rem;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
}

.metric {
    background: var(--bg);
    padding: 1rem;
    border-radius: 6px;
    text-align: center;
}

.metric .value {
    font-size: 1.8rem;
    font-weight: bold;
    color: var(--accent);
}

.metric .label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}

.prediction-row {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: center;
    padding: 0.75rem;
    border-bottom: 1px solid var(--border);
    gap: 1rem;
}

.prediction-row:last-child { border-bottom: none; }

.player-name { font-weight: bold; }
.player-name.favorite { color: var(--accent); }
.player-name.underdog { color: var(--text-muted); }

.prob-bar {
    display: flex;
    height: 24px;
    border-radius: 4px;
    overflow: hidden;
    background: var(--bg);
}

.prob-bar .fill {
    background: var(--accent);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: bold;
    color: var(--bg);
    min-width: 30px;
}

.prob-bar .fill.underdog { background: var(--accent-dim); color: var(--text); }

.desc {
    color: var(--text-muted);
    font-size: 0.8rem;
    margin-bottom: 1rem;
}

canvas { max-width: 100%; }

footer {
    text-align: center;
    padding: 2rem;
    color: var(--text-muted);
    font-size: 0.75rem;
}

footer a { color: var(--accent); text-decoration: none; }
footer a:hover { text-decoration: underline; }

.tag {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    font-size: 0.7rem;
    font-weight: bold;
}

.tag.correct { background: var(--accent-dim); color: var(--accent); }
.tag.wrong { background: #7f1d1d; color: var(--danger); }
.tag.upset { background: #78350f; color: #fbbf24; }
"""
    (SITE_DIR / "assets" / "css" / "style.css").write_text(css)


def _write_js():
    js = """async function init() {
    try {
        const resp = await fetch('predictions.json');
        const data = await resp.json();
        renderStatus(data);
        renderPredictions(data.predictions);
        renderMetrics(data.model_stats);
        renderCalibration(data.calibration);
    } catch (e) {
        document.getElementById('status-content').textContent =
            'No prediction data available. Run the pipeline first.';
    }
}

function renderStatus(data) {
    const el = document.getElementById('status-content');
    el.innerHTML = `
        <div class="metrics-grid">
            <div class="metric">
                <div class="value">${data.model_version || '0.1.0'}</div>
                <div class="label">Model Version</div>
            </div>
            <div class="metric">
                <div class="value">${data.predictions?.length || 0}</div>
                <div class="label">Predictions</div>
            </div>
            <div class="metric">
                <div class="value">${new Date(data.generated_at).toLocaleDateString()}</div>
                <div class="label">Last Updated</div>
            </div>
        </div>
    `;
}

function renderPredictions(predictions) {
    const el = document.getElementById('predictions-content');
    if (!predictions || predictions.length === 0) {
        el.textContent = 'No upcoming predictions. Run the prediction pipeline.';
        return;
    }

    el.innerHTML = predictions.map(p => {
        const p1Pct = Math.round(p.prob_p1 * 100);
        const p2Pct = 100 - p1Pct;
        const fav = p1Pct >= 50 ? 'p1' : 'p2';
        return `
            <div class="prediction-row">
                <span class="player-name ${fav === 'p1' ? 'favorite' : 'underdog'}">${p.player1}</span>
                <div class="prob-bar" style="width:200px">
                    <div class="fill" style="width:${p1Pct}%">${p1Pct}%</div>
                    <div class="fill underdog" style="width:${p2Pct}%">${p2Pct}%</div>
                </div>
                <span class="player-name ${fav === 'p2' ? 'favorite' : 'underdog'}">${p.player2}</span>
            </div>
        `;
    }).join('');
}

function renderMetrics(stats) {
    const el = document.getElementById('metrics-content');
    if (!stats || Object.keys(stats).length === 0) {
        el.textContent = 'Train the model to see performance metrics.';
        return;
    }

    el.innerHTML = `
        <div class="metric">
            <div class="value">${(stats.accuracy * 100).toFixed(1)}%</div>
            <div class="label">Accuracy</div>
        </div>
        <div class="metric">
            <div class="value">${stats.brier_score?.toFixed(4) || 'N/A'}</div>
            <div class="label">Brier Score</div>
        </div>
        <div class="metric">
            <div class="value">${stats.ece?.toFixed(4) || 'N/A'}</div>
            <div class="label">Calibration Error</div>
        </div>
        <div class="metric">
            <div class="value">${stats.n_matches?.toLocaleString() || 'N/A'}</div>
            <div class="label">Matches Evaluated</div>
        </div>
    `;
}

function renderCalibration(cal) {
    const canvas = document.getElementById('calibration-chart');
    if (!canvas || !cal || !cal.bin_centers) return;

    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const pad = 50;

    ctx.fillStyle = '#111827';
    ctx.fillRect(0, 0, w, h);

    // Perfect calibration line
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(pad, h - pad);
    ctx.lineTo(w - pad, pad);
    ctx.stroke();
    ctx.setLineDash([]);

    // Plot actual vs predicted
    if (cal.bin_centers.length > 0) {
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 2;
        ctx.beginPath();
        cal.bin_centers.forEach((x, i) => {
            const px = pad + x * (w - 2 * pad);
            const py = h - pad - cal.actual_rates[i] * (h - 2 * pad);
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        });
        ctx.stroke();

        // Dots
        cal.bin_centers.forEach((x, i) => {
            const px = pad + x * (w - 2 * pad);
            const py = h - pad - cal.actual_rates[i] * (h - 2 * pad);
            ctx.fillStyle = '#10b981';
            ctx.beginPath();
            ctx.arc(px, py, 4, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    // Axes labels
    ctx.fillStyle = '#9ca3af';
    ctx.font = '11px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Predicted Probability', w / 2, h - 10);
    ctx.save();
    ctx.translate(15, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Actual Win Rate', 0, 0);
    ctx.restore();

    // Tick marks
    for (let t = 0; t <= 1; t += 0.2) {
        const x = pad + t * (w - 2 * pad);
        const y = h - pad - t * (h - 2 * pad);
        ctx.fillText(t.toFixed(1), x, h - pad + 15);
        ctx.fillText(t.toFixed(1), pad - 20, y + 4);
    }
}

init();
"""
    (SITE_DIR / "assets" / "js" / "app.js").write_text(js)
