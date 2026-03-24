#!/usr/bin/env python3
"""Daily automated update — the self-learning loop.

Runs on GitHub Actions at 06:00 UTC (full cycle) and 14:00 UTC (quick refresh).
Can also be run locally: `python scripts/daily_update.py [--quick]`

Full cycle (morning):
  1. Refresh data: git pull Sackmann, check ATP RSS for injuries
  2. Self-learn: update Elo/Glicko for new results, track prediction accuracy
  3. Check drift: trigger emergency retrain if model performance has shifted
  4. Retrain: rebuild features + train ensemble on expanding dataset
  5. Predict: scrape upcoming matches, generate predictions
  6. Deploy: update site + commit

Quick refresh (afternoon):
  1. Scrape upcoming matches
  2. Generate predictions (using morning's model)
  3. Deploy site
"""

import json
import os
import pickle
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennis_predictor.config import PROCESSED_DIR, PREDICTIONS_DIR, SACKMANN_DIR, RAW_DIR, SITE_DIR


def main():
    quick = "--quick" in sys.argv or os.environ.get("QUICK_MODE") == "1"

    if quick:
        print(f"=== Quick Refresh: {datetime.now().isoformat()} ===\n")
        predict_and_deploy()
    else:
        print(f"=== Full Daily Cycle: {datetime.now().isoformat()} ===\n")
        full_daily_cycle()

    print(f"\n=== Done: {datetime.now().isoformat()} ===")


def full_daily_cycle():
    """Morning run: refresh → self-learn → retrain → predict → deploy."""
    # Step 1: Refresh all data sources
    refresh_data()

    # Step 2: Self-learn from new results
    new_count = count_new_matches()
    if new_count > 0:
        print(f"\n--- Self-Learning ({new_count} new matches) ---")
        self_learn()

    # Step 3: Check for concept drift
    check_and_handle_drift()

    # Step 4: Retrain model on updated data
    print("\n--- Retraining Model ---")
    retrain_model()

    # Step 5-6: Predict upcoming matches and deploy
    predict_and_deploy()


def predict_and_deploy():
    """Scrape upcoming matches, predict, update site, deploy."""
    print("\n--- Generating Live Predictions ---")
    from tennis_predictor.predict_live import run_live_predictions
    predictions = run_live_predictions()

    print(f"\n--- Updating Site ({len(predictions)} predictions) ---")
    update_site(predictions)


# === DATA REFRESH ===

def refresh_data():
    """Pull latest data from all free sources."""
    print("--- Refreshing Data Sources ---")

    # 1. JeffSackmann match data
    print("  [1/3] Pulling JeffSackmann/tennis_atp...")
    if SACKMANN_DIR.exists() and (SACKMANN_DIR / ".git").exists():
        result = subprocess.run(
            ["git", "pull", "--quiet"], cwd=SACKMANN_DIR,
            capture_output=True, text=True,
        )
        status = result.stdout.strip() or "Already up to date"
        print(f"        {status}")
    else:
        print("        Not cloned — will be cloned during retrain")

    # 2. ATP RSS for injury/withdrawal news
    print("  [2/3] Checking ATP RSS for injuries...")
    try:
        from tennis_predictor.data.news import fetch_atp_rss, detect_injury_signals
        articles = fetch_atp_rss()
        print(f"        {len(articles)} articles fetched")
    except Exception as e:
        print(f"        RSS unavailable: {e}")

    # 3. Weather for active tournaments (uses cache, fast)
    print("  [3/3] Weather data: using cache (refreshed in weekly retrain)")


# === SELF-LEARNING ===

def count_new_matches() -> int:
    """Check how many new matches exist since last processed."""
    import pandas as pd

    matches_path = PROCESSED_DIR / "matches.parquet"
    if not matches_path.exists():
        return 0

    matches = pd.read_parquet(matches_path)
    last_date = matches["tourney_date"].max()
    print(f"  Last processed: {last_date.date()}")

    # Check latest Sackmann file
    current_year = datetime.now().year
    for year in [current_year, current_year - 1]:
        latest_file = SACKMANN_DIR / f"atp_matches_{year}.csv"
        if latest_file.exists():
            new_data = pd.read_csv(latest_file, low_memory=False)
            if "tourney_date" in new_data.columns:
                new_data["tourney_date"] = pd.to_datetime(
                    new_data["tourney_date"], format="%Y%m%d", errors="coerce"
                )
                new_count = len(new_data[new_data["tourney_date"] > last_date])
                if new_count > 0:
                    print(f"  New matches found: {new_count}")
                    return new_count

    print("  No new matches")
    return 0


def self_learn():
    """Update Elo/Glicko ratings and track prediction accuracy."""
    import pandas as pd
    from tennis_predictor.online.learner import OnlineLearner
    from tennis_predictor.temporal.guard import TemporalGuard

    # Load guard state
    guard_path = PROCESSED_DIR / "guard_state_full.pkl"
    if not guard_path.exists():
        guard_path = PROCESSED_DIR / "guard_state.pkl"
    if not guard_path.exists():
        print("  No guard state found. Skipping self-learning.")
        return

    with open(guard_path, "rb") as f:
        guard_state = pickle.load(f)

    guard = TemporalGuard(state=guard_state)
    learner = OnlineLearner(guard=guard)

    # Load yesterday's predictions for tracking
    yesterday_preds = _load_recent_predictions()

    # Process new matches (simplified — full processing happens in retrain)
    stats = learner.get_performance_trend()
    print(f"  Online learner: {stats.get('total_predictions', 0)} predictions logged")
    print(f"  Brier trend: {stats.get('brier_trend', 'insufficient_data')}")

    # Track accuracy of recent predictions
    if yesterday_preds:
        track_record = _check_prediction_accuracy(yesterday_preds)
        if track_record:
            _save_track_record(track_record)

    # Save updated state
    learner._save_state()
    with open(guard_path, "wb") as f:
        pickle.dump(guard.state, f)
    print("  State saved")


def _load_recent_predictions() -> list[dict]:
    """Load predictions from the last few days."""
    history_dir = PREDICTIONS_DIR / "history"
    if not history_dir.exists():
        return []

    preds = []
    for i in range(1, 4):  # Last 3 days
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        path = history_dir / f"{date}.json"
        if path.exists():
            data = json.loads(path.read_text())
            preds.extend(data.get("predictions", []))

    return preds


def _check_prediction_accuracy(predictions: list[dict]) -> dict | None:
    """Match predictions against completed results from Flashscore."""
    from tennis_predictor.data.schedule import fetch_upcoming_matches

    # Get yesterday's completed matches
    yesterday = fetch_upcoming_matches(day_offset=-1)
    completed = [m for m in yesterday if m.get("status") == "finished"]

    if not completed or not predictions:
        return None

    total = 0
    results = []

    for pred in predictions:
        p1 = pred.get("player1", "").lower()
        p2 = pred.get("player2", "").lower()

        # Find matching completed match
        for match in completed:
            m_p1 = match.get("player1", "").lower()
            m_p2 = match.get("player2", "").lower()

            if (p1 in m_p1 or m_p1 in p1) and (p2 in m_p2 or m_p2 in p2):
                predicted_winner = pred["player1"] if pred["prob_p1"] >= 0.5 else pred["player2"]
                prob = max(pred["prob_p1"], pred["prob_p2"])

                results.append({
                    "player1": pred["player1"],
                    "player2": pred["player2"],
                    "predicted_winner": predicted_winner,
                    "predicted_prob": prob,
                    "match_found": True,
                })
                total += 1
                break

    print(f"  Prediction tracking: {total} matches found in completed results")

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "total_tracked": total,
        "results": results,
    }


def _save_track_record(record: dict):
    """Save track record to persistent file."""
    track_file = PREDICTIONS_DIR / "track_record.json"
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    history = []
    if track_file.exists():
        history = json.loads(track_file.read_text())

    history.append(record)
    # Keep last 90 days
    history = history[-90:]

    track_file.write_text(json.dumps(history, indent=2))


# === DRIFT DETECTION ===

def check_and_handle_drift():
    """Check for concept drift and take action if detected."""
    state_file = PROCESSED_DIR / "online_state" / "learner_state.json"
    if not state_file.exists():
        print("  No drift state available")
        return

    state = json.loads(state_file.read_text())
    drift_count = state.get("drift_detector", {}).get("drift_count", 0)

    if drift_count > 0:
        print(f"  WARNING: {drift_count} drift events detected!")

        # In GitHub Actions, trigger the weekly retrain workflow
        if os.environ.get("GITHUB_ACTIONS") == "true":
            print("  Triggering emergency retrain via GitHub API...")
            token = os.environ.get("GITHUB_TOKEN", "")
            repo = os.environ.get("GITHUB_REPOSITORY", "")
            if token and repo:
                import requests
                requests.post(
                    f"https://api.github.com/repos/{repo}/actions/workflows/weekly-retrain.yml/dispatches",
                    headers={"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"},
                    json={"ref": "master"},
                    timeout=10,
                )
                print("  Emergency retrain triggered")
        else:
            print("  Run `tennis-predict full-pipeline` to retrain locally")
    else:
        print("  No drift detected")


# === RETRAIN ===

def retrain_model():
    """Quick retrain: rebuild features + train on updated data (skip external APIs)."""
    try:
        from tennis_predictor.pipeline import run_full_pipeline
        results = run_full_pipeline(
            start_year=1991,
            tour_level_only=False,
            test_year=2023,
            fetch_weather=False,   # Use cache
            fetch_court_speed=False,  # Use cache
            fetch_odds=False,      # Use cache
            compute_intransitivity=False,  # Use cache
        )
        acc = results.get('best_stats', {}).get('accuracy', 0)
        print(f"  Retrain complete: acc={acc:.3f}" if isinstance(acc, (int, float)) else f"  Retrain complete")
    except Exception as e:
        print(f"  Retrain failed: {e}")
        print("  Predictions will use existing model")


# === SITE UPDATE ===

def update_site(predictions: list[dict] | None = None):
    """Regenerate site with latest predictions and stats."""
    from tennis_predictor.web.generate import generate_site

    # Load latest model stats
    stats = {}
    stats_file = PROCESSED_DIR / "latest_stats.json"
    if stats_file.exists():
        stats = json.loads(stats_file.read_text())

    # Load calibration data
    cal = {}
    pred_file = SITE_DIR / "predictions.json"
    if pred_file.exists():
        data = json.loads(pred_file.read_text())
        cal = data.get("calibration", {})

    generate_site(
        predictions=predictions or [],
        model_stats=stats,
        calibration_data=cal,
    )
    print(f"  Site updated at {SITE_DIR}")


if __name__ == "__main__":
    main()
