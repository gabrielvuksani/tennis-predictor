async function init() {
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
        const conf = Math.abs(p1Pct - 50) * 2;
        const surface = p.surface || '';
        const tourney = p.tournament || '';

        return `
            <div class="match-card" role="listitem">
                <div class="player">
                    <span class="player-name ${p1Fav ? 'fav' : 'dog'}">${p.player1}</span>
                    <span class="player-meta">${p.p1_rank ? 'Rank #' + p.p1_rank : ''}</span>
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
                    </div>
                </div>
                <div class="player right">
                    <span class="player-name ${!p1Fav ? 'fav' : 'dog'}">${p.player2}</span>
                    <span class="player-meta">${p.p2_rank ? 'Rank #' + p.p2_rank : ''}</span>
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
