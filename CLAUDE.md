# Tennis Predictor

## Architecture

Calibration-first ATP tennis match prediction system. Optimizes Brier score, not accuracy.
Bloomberg-style trading dashboard UI. **Current: 66.87% accuracy, 0.2068 Brier on 8,801 ATP matches.**

### Core Principle: Zero Data Leakage

The `TemporalGuard` (`src/tennis_predictor/temporal/guard.py`) enforces that all features
use only pre-match data. **Rule: extract features BEFORE updating state, never after.**

### Key Commands

```bash
source .venv/bin/activate

# Full pipeline (recommended — ATP main tour only)
tennis-predict full-pipeline --start-year 1991

# Individual steps
tennis-predict ingest
tennis-predict build-features
tennis-predict train --model ensemble
tennis-predict evaluate
tennis-predict status

# CI triggers
gh workflow run "Weekly Deep Retrain"                    # Full retrain (~20 min)
gh workflow run "Daily Predictions" --field mode=full    # Daily predictions
gh workflow run "Daily Predictions" --field mode=quick   # Quick deploy
```

### Project Structure

- `src/tennis_predictor/temporal/guard.py` — TemporalGuard (most critical file). 5 Elo systems: overall, surface, serve, return, recent, recent-surface
- `src/tennis_predictor/pipeline.py` — Full pipeline orchestrator. Uses `tour_level_only=True` for ATP main tour
- `src/tennis_predictor/models/ensemble.py` — Stacking ensemble (XGB+LGBM+CatBoost → LightGBM meta-learner)
- `src/tennis_predictor/models/gbm.py` — XGBoost, LightGBM, CatBoost wrappers
- `src/tennis_predictor/features/advanced.py` — 100+ advanced features (serve/return, EWMA, set-level, surface-specific)
- `src/tennis_predictor/features/selection.py` — Feature importance analysis (for diagnostics, NOT for training filtering)
- `src/tennis_predictor/data/odds_merge.py` — Betting odds merge (3-strategy cascade, 96.9% coverage on ATP)
- `src/tennis_predictor/data/charting.py` — Match Charting Project loader (point-by-point for 973 players)
- `src/tennis_predictor/data/` — Other loaders: Sackmann, odds, weather, court speed, news, sentiment, schedule
- `src/tennis_predictor/evaluation/metrics.py` — Brier, calibration, ROI, stratified eval
- `src/tennis_predictor/online/learner.py` — Self-learning with ADWIN drift detection
- `src/tennis_predictor/predict_live.py` — Live prediction (Flashscore scraping → ensemble inference)
- `src/tennis_predictor/web/generate.py` — Static site generator (HTML/CSS/JS as Python string literals)
- `src/tennis_predictor/config.py` — Path constants (delegates tunable params to HP)
- `src/tennis_predictor/hyperparams.py` — Central HP singleton (single source of truth for all params)

### Data Sources (all free)

- JeffSackmann/tennis_atp — Match results, rankings, stats (CC BY-NC-SA)
- tennis-data.co.uk — Historical betting odds, 96.9% ATP coverage 2001+ (10+ bookmakers)
- JeffSackmann/tennis_MatchChartingProject — Point-by-point data for 5000+ matches
- TennisMyLife/TML-Database — Gap-fill for 2025-2026 matches
- Open-Meteo — Weather (no API key, data back to 1940)
- Tennis Abstract — Surface speed ratings (1991+)
- ATP Tour RSS — Injury/withdrawal news
- Reddit r/tennis — Sentiment analysis (via PRAW)

### Testing

```bash
pytest tests/ -v    # 85 tests across 5 files
```

Test files: `test_temporal_guard.py`, `test_models.py`, `test_evaluation.py`, `test_features.py`, `test_data.py`

### Critical Lessons (DO NOT IGNORE)

1. **Never do feature selection before training.** GBMs handle 280+ features via built-in regularization. Explicit selection (auto_select_features) HURT accuracy by 0.4%. The selection.py utility is for analysis only.

2. **Never filter features by NaN rate.** A feature with 70% NaN can still be #3 most important (as odds proved). GBMs handle NaN natively.

3. **Odds are the #3 most important feature.** The date parsing bug (`dayfirst=True` on ISO dates) destroyed 61% of odds data. ALWAYS use `_parse_date_flexible()` — try ISO first, then dayfirst.

4. **Always evaluate on ATP main tour only** (`tour_level_only=True`). Including Challengers (68.5% of all data) tanks accuracy and is unfair against published benchmarks.

5. **Binary Elo > margin-weighted Elo.** FiveThirtyEight proved margin-of-victory hurts tennis predictions. `HP.elo.use_margin_weighting` defaults to False.

### Conventions

- Python 3.11+, virtual env in `.venv`
- Temporal validation only (never random cross-validation)
- Brier score is the primary metric, accuracy is secondary
- All features must flow through TemporalGuard
- All tunable parameters in `hyperparams.py` HP singleton — no magic numbers in code
- Config.py is a compatibility layer that delegates to HP
- Free data sources only, no paid APIs
- CI: weekly retrain Sunday 03:00 UTC, daily predictions 06:00/14:00 UTC
