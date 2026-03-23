# Tennis Predictor

## Architecture

Calibration-first ATP tennis match prediction system. Optimizes Brier score, not accuracy.

### Core Principle: Zero Data Leakage

The `TemporalGuard` (`src/tennis_predictor/temporal/guard.py`) enforces that all features
use only pre-match data. The phosphenq model's critical flaw was ELO data leakage —
our architecture makes this impossible.

**Rule: extract features BEFORE updating state, never after.**

### Key Commands

```bash
source .venv/bin/activate

# Full pipeline (recommended)
tennis-predict full-pipeline --start-year 1991

# Individual steps
tennis-predict ingest
tennis-predict build-features
tennis-predict train --model ensemble
tennis-predict evaluate
tennis-predict status
```

### Project Structure

- `src/tennis_predictor/temporal/guard.py` — TemporalGuard (most critical file)
- `src/tennis_predictor/pipeline.py` — Full pipeline orchestrator
- `src/tennis_predictor/models/` — Baselines, GBMs, GNN, ensemble
- `src/tennis_predictor/data/` — Data loaders (Sackmann, odds, weather, court speed, news)
- `src/tennis_predictor/evaluation/metrics.py` — Brier score, calibration, ROI simulation
- `src/tennis_predictor/online/learner.py` — Self-learning with drift detection

### Data Sources (all free)

- JeffSackmann/tennis_atp — Match results, rankings, stats (CC BY-NC-SA)
- tennis-data.co.uk — Historical betting odds (10+ bookmakers, 2001+)
- Open-Meteo — Weather (no API key, data back to 1940)
- CourtSpeed.com — Court Pace Index (CC BY 4.0)
- Tennis Abstract — Surface speed ratings (1991+)
- ATP Tour RSS — Injury/withdrawal news

### Testing

```bash
pytest tests/ -v
```

The TemporalGuard tests are the most important — they verify zero data leakage.

### Conventions

- Python 3.11+, virtual env in `.venv`
- Temporal validation only (never random cross-validation)
- Brier score is the primary metric, accuracy is secondary
- All features must flow through TemporalGuard
