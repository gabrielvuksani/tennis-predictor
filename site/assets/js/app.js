async function init() {
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
