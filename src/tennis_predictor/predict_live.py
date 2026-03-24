"""Live prediction generator — fetches upcoming matches and generates predictions.

Workflow:
1. Scrape upcoming matches from Flashscore (no API key needed)
2. Load trained model + TemporalGuard state (Elo/Glicko ratings)
3. Build features for each match using player ratings
4. Generate calibrated probability predictions
5. Output as JSON for the GitHub Pages site
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tennis_predictor.config import PROCESSED_DIR, PREDICTIONS_DIR, SITE_DIR
from tennis_predictor.hyperparams import HP


def run_live_predictions() -> list[dict]:
    """Full pipeline: scrape matches → predict → save → return."""
    from tennis_predictor.data.schedule import fetch_upcoming_matches

    print("=== Live Prediction Generator ===\n")

    # 1. Fetch upcoming matches (today + tomorrow, deduplicated)
    print("Fetching upcoming ATP matches...")
    today = fetch_upcoming_matches(day_offset=0)
    tomorrow = fetch_upcoming_matches(day_offset=1)
    all_matches = today + tomorrow

    # Deduplicate by player pair
    seen = set()
    upcoming = []
    for m in all_matches:
        key = (m["player1"], m["player2"])
        if key not in seen:
            seen.add(key)
            upcoming.append(m)

    print(f"Found {len(today)} today + {len(tomorrow)} tomorrow = {len(upcoming)} unique")

    if not upcoming:
        print("No upcoming matches found.")
        return []

    # 2. Enrich with sentiment and line movements
    print("Enriching with sentiment and line data...")
    upcoming = _enrich_matches(upcoming)

    # 3. Load model and state
    guard_state, player_lookup = _load_state()

    # 4. Look up LIVE rankings from ESPN (current, not stale Sackmann data)
    from tennis_predictor.data.rankings import fetch_live_rankings, get_player_rank
    live_ranks = fetch_live_rankings()
    sackmann_ranks = _build_ranking_lookup()  # Fallback for players outside top 150

    for match in upcoming:
        # Try ESPN first (current), fall back to Sackmann (stale but broader)
        r1 = get_player_rank(match["player1"], live_ranks)
        if r1 is None:
            pid1 = _find_player_id(match["player1"], player_lookup)
            r1 = sackmann_ranks.get(pid1) if pid1 else None
        match["p1_rank"] = r1

        r2 = get_player_rank(match["player2"], live_ranks)
        if r2 is None:
            pid2 = _find_player_id(match["player2"], player_lookup)
            r2 = sackmann_ranks.get(pid2) if pid2 else None
        match["p2_rank"] = r2

    # 5. Generate predictions
    predictions = []
    for match in upcoming:
        pred = _predict_match(match, guard_state, player_lookup)
        if pred:
            predictions.append(pred)

    # Sort by confidence (most confident first)
    predictions.sort(key=lambda p: p["confidence"], reverse=True)

    # Apply selective prediction analysis
    from tennis_predictor.models.selective import compute_edge_signals
    predictions = compute_edge_signals(predictions)

    high_conf = sum(1 for p in predictions if p.get("recommendation") == "strong_predict")
    print(f"\nGenerated {len(predictions)} predictions ({high_conf} high-confidence)")

    # 4. Save
    save_predictions(predictions)

    return predictions


def _load_state() -> tuple:
    """Load guard state and player lookup."""
    # Load guard state (has Elo/Glicko ratings for all players)
    guard_state = None
    for path in [PROCESSED_DIR / "guard_state_full.pkl", PROCESSED_DIR / "guard_state.pkl"]:
        if path.exists():
            with open(path, "rb") as f:
                guard_state = pickle.load(f)
            print(f"Loaded rating state from {path.name}")
            print(f"  Players tracked: {len(guard_state.elo)}")
            break

    # Load or build player lookup (name -> player_id)
    player_lookup = {}
    lookup_path = PROCESSED_DIR / "player_lookup.json"
    if lookup_path.exists():
        player_lookup = json.loads(lookup_path.read_text())
        print(f"Loaded player lookup: {len(player_lookup)} names")
    else:
        player_lookup = build_player_lookup()

    return guard_state, player_lookup


def _predict_match(
    match: dict,
    guard_state,
    player_lookup: dict,
) -> dict | None:
    """Generate a prediction for a single upcoming match."""
    p1_name = match.get("player1", "")
    p2_name = match.get("player2", "")
    surface = match.get("surface", "Hard")
    tournament = match.get("tournament", "")

    if not p1_name or not p2_name:
        return None

    # Look up player IDs
    p1_id = _find_player_id(p1_name, player_lookup)
    p2_id = _find_player_id(p2_name, player_lookup)

    if guard_state is None:
        # Fallback: use rankings if available
        p1_rank = match.get("p1_rank")
        p2_rank = match.get("p2_rank")
        if p1_rank and p2_rank:
            # Simple logistic model from rank difference
            rank_diff = p2_rank - p1_rank  # Positive = p1 is better
            prob_p1 = 1.0 / (1.0 + 10 ** (-rank_diff / 250))
        else:
            return None
    else:
        init = HP.elo.initial_rating

        # Get Elo ratings
        elo1 = guard_state.elo.get(p1_id, init) if p1_id else init
        elo2 = guard_state.elo.get(p2_id, init) if p2_id else init

        # Surface Elo
        surf_elo1 = guard_state.elo_surface.get((p1_id, surface), init) if p1_id else init
        surf_elo2 = guard_state.elo_surface.get((p2_id, surface), init) if p2_id else init

        # Blended probability (60% surface Elo, 40% overall Elo)
        elo_prob = 1.0 / (1.0 + 10 ** ((elo2 - elo1) / 400))
        surf_prob = 1.0 / (1.0 + 10 ** ((surf_elo2 - surf_elo1) / 400))
        prob_p1 = 0.4 * elo_prob + 0.6 * surf_prob

        # Adjust toward 50% if we have no data on a player (uncertainty)
        if p1_id is None or elo1 == init:
            prob_p1 = 0.4 * prob_p1 + 0.6 * 0.5
        if p2_id is None or elo2 == init:
            prob_p1 = 0.4 * prob_p1 + 0.6 * 0.5

        # Blend with point-level simulation (if serve/return stats available)
        try:
            from tennis_predictor.models.point_sim import get_point_sim_prediction
            p1_hist = guard_state.match_history.get(p1_id, []) if p1_id else []
            p2_hist = guard_state.match_history.get(p2_id, []) if p2_id else []

            p1_spw = np.nanmean([m.get("serve_pts_won_pct", np.nan) for m in p1_hist[-20:]]) if p1_hist else None
            p1_rpw = np.nanmean([m.get("return_pts_won_pct", np.nan) for m in p1_hist[-20:]]) if p1_hist else None
            p2_spw = np.nanmean([m.get("serve_pts_won_pct", np.nan) for m in p2_hist[-20:]]) if p2_hist else None
            p2_rpw = np.nanmean([m.get("return_pts_won_pct", np.nan) for m in p2_hist[-20:]]) if p2_hist else None

            best_of = 5 if match.get("tournament", "").lower() in [
                "australian open", "roland garros", "wimbledon", "us open"
            ] else 3

            sim_prob = get_point_sim_prediction(p1_spw, p1_rpw, p2_spw, p2_rpw, best_of)
            if sim_prob is not None and not np.isnan(sim_prob):
                # Blend: 70% Elo-based, 30% point simulation
                prob_p1 = 0.7 * prob_p1 + 0.3 * sim_prob
        except Exception:
            pass  # Point sim is supplementary; don't fail if it errors

    prob_p1 = max(0.05, min(0.95, prob_p1))

    # Confidence level
    confidence = abs(prob_p1 - 0.5) * 2  # 0 = toss-up, 1 = certain

    return {
        "player1": p1_name,
        "player2": p2_name,
        "prob_p1": round(prob_p1, 3),
        "prob_p2": round(1 - prob_p1, 3),
        "tournament": tournament,
        "surface": surface,
        "start_time": match.get("start_time", ""),
        "status": match.get("status", "upcoming"),
        "p1_rank": match.get("p1_rank"),
        "p2_rank": match.get("p2_rank"),
        "confidence": round(confidence, 3),
        "intransitivity_score": match.get("intransitivity_score", 0),
        "sharp_signal": match.get("sharp_signal", 0),
        "model": "elo_surface_blend",
        "generated_at": datetime.now().isoformat(),
    }


def _build_ranking_lookup() -> dict[str, int]:
    """Build player_id → ATP ranking from the most recent Sackmann data."""
    matches_path = PROCESSED_DIR / "matches.parquet"
    if not matches_path.exists():
        return {}

    try:
        matches = pd.read_parquet(matches_path)
        # Get the last known ranking for each player (from winner and loser columns)
        rankings = {}
        # Sort by date, take the most recent rank for each player
        recent = matches.sort_values("tourney_date", ascending=False).head(20000)

        for _, row in recent.iterrows():
            wid = str(row.get("winner_id", ""))
            lid = str(row.get("loser_id", ""))
            wr = row.get("winner_rank")
            lr = row.get("loser_rank")

            if wid and pd.notna(wr) and wid not in rankings:
                rankings[wid] = int(wr)
            if lid and pd.notna(lr) and lid not in rankings:
                rankings[lid] = int(lr)

        return rankings
    except Exception:
        return {}


def _enrich_matches(matches: list[dict]) -> list[dict]:
    """Add sentiment and line movement data to matches."""
    # Sentiment (Reddit)
    try:
        from tennis_predictor.data.sentiment import get_player_sentiment
        for match in matches:
            p1_sent = get_player_sentiment(match["player1"])
            p2_sent = get_player_sentiment(match["player2"])
            match["p1_sentiment"] = p1_sent.get("sentiment_score", 0)
            match["p2_sentiment"] = p2_sent.get("sentiment_score", 0)
            match["p1_injury_signal"] = p1_sent.get("injury_signal", 0)
            match["p2_injury_signal"] = p2_sent.get("injury_signal", 0)
            match["p1_momentum_signal"] = p1_sent.get("momentum_signal", 0)
            match["p2_momentum_signal"] = p2_sent.get("momentum_signal", 0)
            match["sentiment_diff"] = match["p1_sentiment"] - match["p2_sentiment"]
    except Exception as e:
        print(f"  Sentiment enrichment skipped: {e}")

    # Line movements (Bovada odds snapshots)
    try:
        from tennis_predictor.data.line_movements import track_line_movements, get_line_features
        movements = track_line_movements(matches)
        for match in matches:
            lf = get_line_features(match["player1"], match["player2"])
            match.update(lf)
    except Exception as e:
        print(f"  Line movement enrichment skipped: {e}")

    return matches


def _find_player_id(name: str, player_lookup: dict) -> str | None:
    """Find player ID from name, handling various formats.

    Flashscore uses "Last I." format (e.g., "Sinner J.", "Fritz T.").
    Sackmann uses "First Last" format (e.g., "Jannik Sinner", "Taylor Fritz").

    Strategy:
    1. Direct lookup
    2. Convert "Last I." → find "First Last" where first starts with I and last matches
    3. Fallback: unique last name match (only if exactly 1 candidate)
    """
    if not name:
        return None

    name_lower = name.strip().lower()

    # Direct lookup
    if name_lower in player_lookup:
        return player_lookup[name_lower]

    parts = name_lower.split()
    if not parts:
        return None

    # Handle "Last I." format (most common from Flashscore)
    # e.g., "Fritz T." → last_name="fritz", initial="t"
    # e.g., "Etcheverry T. M." → last_name could be multi-word
    # Strategy: try initial-based matching first (most precise)

    # Find the initial (last part ending with ".")
    initial = None
    last_parts = []
    for i, part in enumerate(parts):
        if part.endswith(".") and len(part) <= 3:
            initial = part[0]
            last_parts = parts[:i]
            break
    else:
        # No initial found — try "Last First" reordering
        last_parts = parts

    if initial and last_parts:
        last_name = " ".join(last_parts)
        # Find all Sackmann names where last name matches AND first initial matches
        candidates = [
            (k, v) for k, v in player_lookup.items()
            if k.endswith(" " + last_name) and k.split()[0].startswith(initial)
        ]
        if len(candidates) == 1:
            return candidates[0][1]
        # If multiple candidates (e.g., two "A. Zverev"), try longer initial match
        if len(candidates) > 1:
            # Return the one with highest Elo (most active/famous player)
            # We don't have Elo here, so pick the shorter name (usually the famous one)
            candidates.sort(key=lambda x: len(x[0]))
            return candidates[0][1]

    # Fallback: try "Last First" → "First Last" reordering
    if len(parts) >= 2:
        reordered = " ".join(parts[1:]).rstrip(".") + " " + parts[0]
        if reordered in player_lookup:
            return player_lookup[reordered]

    # Last resort: unique last name match (only if exactly 1 candidate)
    last_name = parts[0].rstrip(".")
    candidates = [
        (k, v) for k, v in player_lookup.items()
        if k.split()[-1] == last_name
    ]
    if len(candidates) == 1:
        return candidates[0][1]

    return None


def save_predictions(predictions: list[dict]) -> Path:
    """Save predictions to site JSON and daily history (for tracking accuracy)."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "generated_at": datetime.now().isoformat(),
        "n_predictions": len(predictions),
        "predictions": predictions,
    }

    # Save latest
    latest_path = PREDICTIONS_DIR / "latest.json"
    latest_path.write_text(json.dumps(output, indent=2, default=str))

    # Save daily history (for prediction tracking / feedback loop)
    history_dir = PREDICTIONS_DIR / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    history_path = history_dir / f"{today}.json"
    history_path.write_text(json.dumps(output, indent=2, default=str))

    # Update site predictions.json
    site_path = SITE_DIR / "predictions.json"
    if site_path.exists():
        existing = json.loads(site_path.read_text())
        existing["predictions"] = predictions
        existing["generated_at"] = datetime.now().isoformat()
        site_path.write_text(json.dumps(existing, indent=2, default=str))
    else:
        SITE_DIR.mkdir(parents=True, exist_ok=True)
        site_path.write_text(json.dumps(output, indent=2, default=str))

    print(f"Saved {len(predictions)} predictions to {site_path} + history/{today}.json")
    return site_path


def build_player_lookup() -> dict:
    """Build a name → player_id mapping from Sackmann data."""
    matches_path = PROCESSED_DIR / "matches.parquet"
    if not matches_path.exists():
        print("No matches.parquet found. Run pipeline first.")
        return {}

    matches = pd.read_parquet(matches_path)
    lookup = {}

    for col_name, col_id in [("winner_name", "winner_id"), ("loser_name", "loser_id")]:
        for _, row in matches[[col_name, col_id]].drop_duplicates().iterrows():
            if pd.notna(row[col_name]) and pd.notna(row[col_id]):
                lookup[str(row[col_name]).lower().strip()] = str(row[col_id])

    path = PROCESSED_DIR / "player_lookup.json"
    path.write_text(json.dumps(lookup, indent=2))
    print(f"Built player lookup: {len(lookup)} names")
    return lookup


if __name__ == "__main__":
    predictions = run_live_predictions()
    if predictions:
        print(f"\n{'='*60}")
        print(f"{'Player 1':<25} {'Player 2':<25} {'Prob':>6} {'Conf':>5}")
        print(f"{'-'*60}")
        for p in predictions[:20]:
            fav = p["player1"] if p["prob_p1"] >= 0.5 else p["player2"]
            prob = max(p["prob_p1"], p["prob_p2"])
            print(f"{p['player1']:<25} {p['player2']:<25} {prob:>5.0%} {p['confidence']:>5.1%}")
