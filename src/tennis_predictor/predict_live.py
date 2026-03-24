"""Live prediction generator — fetches upcoming matches and generates predictions.

Workflow:
1. Load the trained model and TemporalGuard state from disk
2. Fetch upcoming matches (The Odds API free tier or fallback to schedule)
3. Build features for each upcoming match using the guard's current state
4. Generate calibrated probability predictions
5. Output predictions as JSON for the site
"""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from tennis_predictor.config import PROCESSED_DIR, PREDICTIONS_DIR, SITE_DIR


# === UPCOMING MATCH SOURCES ===

def fetch_upcoming_from_odds_api(api_key: str | None = None) -> list[dict]:
    """Fetch upcoming ATP matches from The Odds API (free tier: 500 credits/month).

    Each call to /odds costs 1 credit per region. /events costs 0 credits.
    """
    api_key = api_key or os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        return []

    matches = []
    for sport in ["tennis_atp_us_open", "tennis_atp_french_open",
                   "tennis_atp_wimbledon", "tennis_atp_aus_open"]:
        try:
            # Events endpoint is FREE (0 credits)
            resp = requests.get(
                f"https://api.the-odds-api.com/v4/sports/{sport}/events",
                params={"apiKey": api_key},
                timeout=15,
            )
            if resp.status_code == 200:
                for event in resp.json():
                    matches.append({
                        "player1": event.get("home_team", ""),
                        "player2": event.get("away_team", ""),
                        "commence_time": event.get("commence_time", ""),
                        "sport": sport,
                        "tournament": _sport_to_tournament(sport),
                    })
        except requests.RequestException:
            continue

    # Also try generic ATP endpoint
    try:
        resp = requests.get(
            "https://api.the-odds-api.com/v4/sports",
            params={"apiKey": api_key},
            timeout=15,
        )
        if resp.status_code == 200:
            atp_sports = [s["key"] for s in resp.json()
                          if "tennis" in s["key"].lower() and s.get("active", False)]
            for sport in atp_sports:
                if sport in ["tennis_atp_us_open", "tennis_atp_french_open",
                             "tennis_atp_wimbledon", "tennis_atp_aus_open"]:
                    continue  # Already fetched
                try:
                    resp2 = requests.get(
                        f"https://api.the-odds-api.com/v4/sports/{sport}/events",
                        params={"apiKey": api_key},
                        timeout=15,
                    )
                    if resp2.status_code == 200:
                        for event in resp2.json():
                            matches.append({
                                "player1": event.get("home_team", ""),
                                "player2": event.get("away_team", ""),
                                "commence_time": event.get("commence_time", ""),
                                "sport": sport,
                                "tournament": _sport_to_tournament(sport),
                            })
                except requests.RequestException:
                    continue
    except requests.RequestException:
        pass

    remaining = resp.headers.get("x-requests-remaining", "?") if resp else "?"
    print(f"Fetched {len(matches)} upcoming matches from The Odds API (credits remaining: {remaining})")
    return matches


def fetch_upcoming_from_rss() -> list[dict]:
    """Fallback: get tournament context from ATP RSS feed."""
    try:
        import feedparser
        feed = feedparser.parse("https://www.atptour.com/en/media/rss-feed/xml-feed")
        # RSS doesn't give match schedules, but gives tournament context
        return []
    except Exception:
        return []


# === PREDICTION ENGINE ===

def generate_predictions(upcoming_matches: list[dict] | None = None) -> list[dict]:
    """Generate predictions for upcoming matches.

    If no upcoming matches are provided, uses dummy data for site display.
    """
    # Load model
    model_path = PROCESSED_DIR / "model_ensemble.pkl"
    if not model_path.exists():
        # Try individual model
        for name in ["model_catboost.pkl", "model_xgboost.pkl"]:
            alt = PROCESSED_DIR / name
            if alt.exists():
                model_path = alt
                break

    if not model_path.exists():
        print("No trained model found. Run the training pipeline first.")
        return []

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load guard state
    guard_path = PROCESSED_DIR / "guard_state_full.pkl"
    if not guard_path.exists():
        guard_path = PROCESSED_DIR / "guard_state.pkl"
    if not guard_path.exists():
        print("No guard state found. Run the feature pipeline first.")
        return []

    with open(guard_path, "rb") as f:
        guard_state = pickle.load(f)

    # Load player lookup
    players_path = PROCESSED_DIR / "player_lookup.json"
    player_lookup = {}
    if players_path.exists():
        player_lookup = json.loads(players_path.read_text())

    if not upcoming_matches:
        print("No upcoming matches to predict.")
        return []

    # Build features for each match and predict
    predictions = []
    from tennis_predictor.temporal.guard import TemporalGuard, TemporalState
    guard = TemporalGuard(state=guard_state)

    for match in upcoming_matches:
        try:
            p1_name = match.get("player1", "")
            p2_name = match.get("player2", "")

            if not p1_name or not p2_name:
                continue

            # Look up player IDs
            p1_id = player_lookup.get(p1_name.lower(), p1_name)
            p2_id = player_lookup.get(p2_name.lower(), p2_name)

            # Get Elo ratings from state
            from tennis_predictor.config import ELO_CONFIG
            init = ELO_CONFIG["initial_rating"]
            elo1 = guard.state.elo.get(str(p1_id), init)
            elo2 = guard.state.elo.get(str(p2_id), init)
            elo_prob = 1.0 / (1.0 + 10 ** ((elo2 - elo1) / 400))

            predictions.append({
                "player1": p1_name,
                "player2": p2_name,
                "prob_p1": round(elo_prob, 3),
                "prob_p2": round(1 - elo_prob, 3),
                "tournament": match.get("tournament", ""),
                "commence_time": match.get("commence_time", ""),
                "model": "elo_baseline",
                "generated_at": datetime.now().isoformat(),
            })
        except Exception as e:
            print(f"Error predicting {match}: {e}")
            continue

    return predictions


def save_predictions(predictions: list[dict]) -> Path:
    """Save predictions to JSON for the site."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "generated_at": datetime.now().isoformat(),
        "n_predictions": len(predictions),
        "predictions": predictions,
    }

    path = PREDICTIONS_DIR / "latest.json"
    path.write_text(json.dumps(output, indent=2))

    # Also update site predictions
    site_pred = SITE_DIR / "predictions.json"
    if site_pred.exists():
        existing = json.loads(site_pred.read_text())
        existing["predictions"] = predictions
        existing["generated_at"] = datetime.now().isoformat()
        site_pred.write_text(json.dumps(existing, indent=2))

    print(f"Saved {len(predictions)} predictions to {path}")
    return path


def build_player_lookup() -> dict:
    """Build a name -> player_id mapping from Sackmann data."""
    matches_path = PROCESSED_DIR / "matches.parquet"
    if not matches_path.exists():
        return {}

    matches = pd.read_parquet(matches_path)
    lookup = {}

    for _, row in matches.iterrows():
        if pd.notna(row.get("winner_name")) and pd.notna(row.get("winner_id")):
            lookup[str(row["winner_name"]).lower()] = str(row["winner_id"])
        if pd.notna(row.get("loser_name")) and pd.notna(row.get("loser_id")):
            lookup[str(row["loser_name"]).lower()] = str(row["loser_id"])

    path = PROCESSED_DIR / "player_lookup.json"
    path.write_text(json.dumps(lookup, indent=2))
    print(f"Built player lookup: {len(lookup)} players")
    return lookup


def _sport_to_tournament(sport_key: str) -> str:
    mapping = {
        "tennis_atp_us_open": "US Open",
        "tennis_atp_french_open": "Roland Garros",
        "tennis_atp_wimbledon": "Wimbledon",
        "tennis_atp_aus_open": "Australian Open",
    }
    return mapping.get(sport_key, sport_key.replace("tennis_atp_", "").replace("_", " ").title())


# === RESULTS PROCESSING ===

def process_latest_results() -> dict:
    """Process the latest match results to update Elo/Glicko ratings.

    This is the online learning step — called after each day of matches.
    """
    from tennis_predictor.online.learner import OnlineLearner

    learner = OnlineLearner()
    stats = learner.get_performance_trend()
    print(f"Online learner: {stats.get('total_predictions', 0)} predictions, "
          f"Brier trend: {stats.get('brier_trend', 'unknown')}")
    return stats


if __name__ == "__main__":
    import sys

    api_key = os.environ.get("ODDS_API_KEY", "")

    # Fetch upcoming matches
    upcoming = fetch_upcoming_from_odds_api(api_key)
    if not upcoming:
        print("No upcoming matches found. Site will show model stats only.")

    # Generate predictions
    predictions = generate_predictions(upcoming)

    # Save
    if predictions:
        save_predictions(predictions)
    else:
        print("No predictions generated.")
