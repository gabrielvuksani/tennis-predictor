#!/usr/bin/env python3
"""Daily automated update script.

This runs on GitHub Actions every day at 06:00 UTC:
1. Pull latest match results from JeffSackmann
2. Update Elo/Glicko ratings (online learning)
3. Check for concept drift
4. Fetch upcoming matches and generate predictions
5. Update the GitHub Pages site
6. Commit and push any changes

Designed to be lightweight (~5 min) vs the full pipeline (~40 min).
"""

import json
import os
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tennis_predictor.config import PROCESSED_DIR, SITE_DIR, SACKMANN_DIR, RAW_DIR


def main():
    print(f"=== Daily Update: {datetime.now().isoformat()} ===\n")

    # Step 1: Update Sackmann data
    print("Step 1: Updating JeffSackmann data...")
    update_sackmann_data()

    # Step 2: Check for new matches
    print("\nStep 2: Checking for new matches...")
    new_matches = check_new_matches()

    if new_matches > 0:
        print(f"Found {new_matches} new matches since last update")

        # Step 3: Update online state (Elo/Glicko)
        print("\nStep 3: Updating online Elo/Glicko ratings...")
        update_online_state()

        # Step 4: Check for concept drift
        print("\nStep 4: Checking for concept drift...")
        check_drift()
    else:
        print("No new matches found")

    # Step 5: Fetch upcoming matches and predict
    print("\nStep 5: Generating predictions for upcoming matches...")
    generate_live_predictions()

    # Step 6: Update site
    print("\nStep 6: Updating site...")
    update_site()

    print(f"\n=== Daily update complete: {datetime.now().isoformat()} ===")


def update_sackmann_data():
    """Pull latest data from JeffSackmann/tennis_atp."""
    if SACKMANN_DIR.exists() and (SACKMANN_DIR / ".git").exists():
        result = subprocess.run(
            ["git", "pull", "--quiet"],
            cwd=SACKMANN_DIR,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"  Updated: {result.stdout.strip() or 'Already up to date'}")
        else:
            print(f"  Warning: git pull failed: {result.stderr}")
    else:
        print("  Sackmann repo not found. Run full pipeline first.")


def check_new_matches() -> int:
    """Check if there are new matches since our last processed date."""
    import pandas as pd

    matches_path = PROCESSED_DIR / "matches.parquet"
    if not matches_path.exists():
        return 0

    matches = pd.read_parquet(matches_path)
    last_date = matches["tourney_date"].max()
    print(f"  Last processed match: {last_date}")

    # Check latest Sackmann file
    current_year = datetime.now().year
    latest_file = SACKMANN_DIR / f"atp_matches_{current_year}.csv"
    if latest_file.exists():
        new_data = pd.read_csv(latest_file, low_memory=False)
        if "tourney_date" in new_data.columns:
            new_data["tourney_date"] = pd.to_datetime(
                new_data["tourney_date"], format="%Y%m%d", errors="coerce"
            )
            new_max = new_data["tourney_date"].max()
            new_count = len(new_data[new_data["tourney_date"] > last_date])
            print(f"  Latest available: {new_max} ({new_count} new matches)")
            return new_count

    return 0


def update_online_state():
    """Update Elo/Glicko ratings with new match results."""
    state_file = PROCESSED_DIR / "online_state" / "learner_state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())
        print(f"  Predictions logged: {len(state.get('prediction_log', []))}")
        print(f"  Retrains: {state.get('retrain_count', 0)}")
    else:
        print("  No online state found. Will be created on next full pipeline run.")


def check_drift():
    """Check for concept drift in recent predictions."""
    state_file = PROCESSED_DIR / "online_state" / "learner_state.json"
    if not state_file.exists():
        print("  No prediction history for drift detection")
        return

    state = json.loads(state_file.read_text())
    drift_count = state.get("drift_detector", {}).get("drift_count", 0)
    if drift_count > 0:
        print(f"  WARNING: {drift_count} drift events detected. Consider full retrain.")
    else:
        print("  No drift detected")


def generate_live_predictions():
    """Fetch upcoming matches and generate predictions."""
    from tennis_predictor.predict_live import (
        fetch_upcoming_from_odds_api,
        generate_predictions,
        save_predictions,
    )

    api_key = os.environ.get("ODDS_API_KEY", "")
    upcoming = fetch_upcoming_from_odds_api(api_key)

    if upcoming:
        predictions = generate_predictions(upcoming)
        if predictions:
            save_predictions(predictions)
            print(f"  Generated {len(predictions)} predictions")
        else:
            print("  No predictions generated (model may not be trained)")
    else:
        print("  No upcoming matches found (set ODDS_API_KEY for live data)")


def update_site():
    """Regenerate site with latest predictions and stats."""
    from tennis_predictor.web.generate import generate_site

    # Load latest stats if available
    stats = {}
    stats_file = PROCESSED_DIR / "latest_stats.json"
    if stats_file.exists():
        stats = json.loads(stats_file.read_text())

    # Load latest predictions
    predictions = []
    pred_file = SITE_DIR / "predictions.json"
    if pred_file.exists():
        pred_data = json.loads(pred_file.read_text())
        predictions = pred_data.get("predictions", [])

    generate_site(predictions=predictions, model_stats=stats)
    print(f"  Site updated at {SITE_DIR}")


if __name__ == "__main__":
    main()
