# Tennis Predictor

**AI-powered ATP tennis match prediction system.** Self-learning, 230+ features, calibrated probabilities, fully autonomous.

**[Live Predictions →](https://gabrielvuksani.github.io/tennis-predictor/)**

---

## Results

Evaluated on **27,787 test matches** (2023-2024), with zero data leakage:

| Model | Accuracy | Brier Score | ECE |
|-------|----------|-------------|-----|
| Elo Baseline | 64.2% | 0.221 | 0.029 |
| XGBoost | 65.7% | 0.212 | 0.023 |
| LightGBM | 65.6% | 0.212 | 0.020 |
| **CatBoost** | **65.9%** | **0.212** | **0.012** |
| IBM SlamTracker | 63.8% | — | — |
| phosphenq model | 66.3% | — | — |
| Bookmakers | ~72% | ~0.196 | — |

**Grand Slams: 68.9% accuracy, 0.200 Brier** — matching bookmaker calibration at majors.

When the model is confident (>70%): **76.0% accuracy.**

---

## How It Works

### Architecture

```
Data (12 sources)  →  Features (230+)  →  Model (Stacking Ensemble)  →  Predictions
     ↓                      ↓                       ↓                        ↓
 JeffSackmann         Elo/Glicko-2          XGBoost + LightGBM        Live predictions
 Flashscore           Rolling stats         + CatBoost stacking       on GitHub Pages
 Open-Meteo           H2H, fatigue          Isotonic calibration      (updated 2x daily)
 Reddit r/tennis      Court speed, weather   TemporalGuard             Self-learning
 Bovada odds          Sentiment, streaks    (zero data leakage)        Drift detection
 Tennis Abstract      Common opponents
 ATP RSS              Intransitivity
```

### The TemporalGuard

The core innovation. The phosphenq model's critical flaw was data leakage — ELO ratings updated with match results were fed to the classifier BEFORE the match was predicted. Our `TemporalGuard` makes this **architecturally impossible**:

1. Features are extracted **before** state is updated
2. State is updated **after** features are extracted
3. The guard enforces this ordering and raises `TemporalLeakageError` on violation

### Self-Learning

The system improves itself through five mechanisms:

1. **Elo/Glicko-2 ratings** update after every match result (margin-weighted WElo)
2. **Rolling statistics** shift as new results enter the 5/10/20/50 match windows
3. **Daily model retrain** on expanding dataset with time-decay weighting
4. **ADWIN drift detection** monitors prediction errors → triggers emergency retrain
5. **Prediction tracking** matches outcomes to yesterday's predictions → feedback loop

### Key Research Applied

| Technique | Source | Impact |
|-----------|--------|--------|
| Calibration-first optimization | Walsh & Joshi (2024) | +34.69% ROI vs accuracy-optimized |
| Confidence-based filtering | Clegg GNN (2025) | 3.26% ROI on targeted subset |
| Common-opponent analysis | Knottenbelt (2012) | 3.8% ROI on 2,173 matches |
| Margin-weighted Elo (WElo) | Angelini et al. | Improved Brier by ~0.005 |
| Intransitivity detection | Clegg (2025) | Bookmakers weakest on A>B>C>A cycles |

---

## Features (230+)

| Category | Count | Examples |
|----------|-------|---------|
| **Rating systems** | 14 | Elo, surface Elo, serve/return Elo, Glicko-2 (rating, RD, volatility) |
| **Rolling statistics** | 54 | Win rate, serve %, break %, ace rate (5/10/20/50 windows) |
| **EWMA stats** | 6 | Exponentially weighted win rate, surface win rate, serve % |
| **Serve/return dominance** | 14 | 2nd serve win%, DF rate, hold%, break%, SPW+RPW, efficiency |
| **Head-to-head** | 4 | Total H2H, surface H2H, win percentages |
| **Fatigue/scheduling** | 11 | Days since last, ACWR, matches in 7/14/30d, hours played |
| **Momentum/streaks** | 8 | Win/loss streak, form velocity, form gradient, EWMA momentum |
| **Tournament context** | 12 | Surface, level, round, draw size, best-of-5, seeding |
| **Player profile** | 12 | Age, height, handedness, distance from peak, career stage |
| **Common opponents** | 4 | Shared opponent count, comparative win rates |
| **Weather/environment** | 8 | Temperature, wind, humidity, altitude, indoor flag |
| **Court speed** | 3 | Court pace index, serve×speed interaction |
| **Intransitivity** | 1 | Graph-based cycle detection score |
| **Sentiment** | 7 | Reddit sentiment, injury signal, momentum signal |
| **Line movements** | 5 | Direction, magnitude, sharp money signal |
| **Betting odds** | 4 | Implied probabilities, model-vs-odds disagreement |
| **Supplementary** | ~60 | Home advantage, retirement risk, surface adaptation, level win rate, etc. |

---

## Data Sources (12)

All free, no API keys required:

| Source | What | Refresh |
|--------|------|---------|
| [JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp) | Match results, stats, rankings (1991-present) | Daily `git pull` |
| [Flashscore.ninja](https://www.flashscore.com) | Upcoming match schedules, live scores | 2x daily scrape |
| [Bovada API](https://www.bovada.lv) | Backup schedule + betting odds | 2x daily |
| [tennis-data.co.uk](http://www.tennis-data.co.uk) | Historical closing odds (10+ bookmakers) | Weekly |
| [Open-Meteo](https://open-meteo.com) | Weather (temp, wind, humidity) | Daily (cached) |
| [Tennis Abstract](https://www.tennisabstract.com) | Court speed ratings | Weekly scrape |
| [CourtSpeed.com](https://courtspeed.com) | Court Pace Index | CC BY 4.0 CSV |
| ATP Tour RSS | Injury/withdrawal news | Daily |
| Reddit r/tennis | Sentiment, injury chatter | Cached per player |
| Elo/Glicko-2 state | Live player ratings (9,279 tracked) | After every match |
| Intransitivity graph | Matchup cycle detection | Weekly recompute |
| Line movement snapshots | Sharp money signals | Per-prediction |

---

## Automation

Runs indefinitely on GitHub Actions (free for public repos):

| Schedule | Time | Duration | What Runs |
|----------|------|----------|-----------|
| **Morning** | 06:00 UTC daily | ~20 min | Self-learn → retrain → predict → deploy |
| **Afternoon** | 14:00 UTC daily | ~30 sec | Re-scrape → predict → deploy |
| **Weekly** | Sunday 03:00 UTC | ~40 min | All data sources → full pipeline → evaluate → deploy |

Self-healing: drift detected → auto-retrain. Scraper fails → fallback chain (Flashscore → Bovada → Tennis Explorer). Workflow fails → GitHub email notification.

---

## Quick Start

```bash
# Clone
git clone https://github.com/gabrielvuksani/tennis-predictor.git
cd tennis-predictor

# Setup
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install xlrd openpyxl rapidfuzz

# Run full pipeline (trains on 308K matches, ~30 min)
tennis-predict full-pipeline

# Generate live predictions
tennis-predict live

# Run daily update (self-learn + predict)
python scripts/daily_update.py

# Quick refresh (predictions only)
python scripts/daily_update.py --quick

# Check system status
tennis-predict status
```

---

## Configuration

All 150+ tunable parameters live in `config/hyperparams.yaml`:

```yaml
elo:
  k_factor_base: 250.0
  surface_weight: 0.6      # 60% surface Elo, 40% overall
  bo5_multiplier: 1.10

features:
  ewma_alpha: 0.15
  peak_age: 25.5
  rolling_windows: [5, 10, 20, 50]

model:
  xgb_learning_rate: 0.05
  cat_depth: 6

betting:
  min_edge: 0.05
  kelly_fraction: 0.25
```

No magic numbers in code — everything imports from `tennis_predictor.hyperparams.HP`.

---

## Project Structure

```
tennis-predictor/
├── config/hyperparams.yaml          # All tunable parameters
├── src/tennis_predictor/
│   ├── temporal/guard.py            # TemporalGuard (anti-leakage)
│   ├── temporal/validation.py       # Expanding-window temporal CV
│   ├── features/advanced.py         # 80+ derived features
│   ├── models/
│   │   ├── baseline.py              # Elo, Rank, Odds baselines
│   │   ├── gbm.py                   # XGBoost, LightGBM, CatBoost
│   │   ├── ensemble.py              # Stacking meta-learner
│   │   └── gnn.py                   # Graph neural network (optional)
│   ├── data/
│   │   ├── sackmann.py              # JeffSackmann data loader
│   │   ├── schedule.py              # Flashscore/Bovada/Tennis Explorer
│   │   ├── weather.py               # Open-Meteo integration
│   │   ├── court_speed.py           # Tennis Abstract scraper
│   │   ├── odds.py                  # Betting odds loader
│   │   ├── news.py                  # ATP RSS + injury detection
│   │   ├── sentiment.py             # Reddit r/tennis sentiment
│   │   └── line_movements.py        # Betting line tracker
│   ├── evaluation/metrics.py        # Brier, calibration, ROI, upset metrics
│   ├── online/learner.py            # Self-learning + drift detection
│   ├── pipeline.py                  # Full pipeline orchestrator
│   ├── predict_live.py              # Live prediction engine
│   ├── hyperparams.py               # Dynamic parameter system
│   └── web/generate.py              # GitHub Pages site generator
├── scripts/daily_update.py          # Daily automation script
├── site/                            # Static site (deployed to Pages)
├── tests/                           # TemporalGuard leak prevention tests
└── .github/workflows/               # 2 automated workflows
```

---

## How Predictions Are Made

For each upcoming match:

1. **Scrape** match schedule from Flashscore (player names, ranks, surface, tournament)
2. **Enrich** with Reddit sentiment (injury signals, momentum) and line movements (sharp money)
3. **Look up** both players in the rating state (Elo, Glicko-2, surface ratings for 9,279 players)
4. **Blend** surface-weighted Elo probability (60% surface Elo, 40% overall)
5. **Adjust** for unknown players (shrink toward 50% if no data)
6. **Tag** with edge signals (high confidence, intransitive matchup, sharp money divergence)
7. **Save** to site JSON and prediction history (for next-day accuracy tracking)

---

## License

Data: [JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp) — CC BY-NC-SA 4.0 (non-commercial).

Code: MIT

---

*Built with zero data leakage, calibration over accuracy, and the principle that profit comes from knowing when you have an edge — not from being right on average.*
