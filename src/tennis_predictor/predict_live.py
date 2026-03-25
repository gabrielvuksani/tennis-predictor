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

from tennis_predictor.config import PROCESSED_DIR, PREDICTIONS_DIR, SITE_DIR, SACKMANN_DIR
from tennis_predictor.hyperparams import HP


# --- Player bio cache (age, height, handedness from Sackmann atp_players.csv) ---
_PLAYER_BIO_CACHE: dict | None = None


def _load_player_bio() -> dict:
    """Load player bio data (dob, height, hand) keyed by player_id string.

    Returns a dict: player_id_str -> {"dob": Timestamp|NaT, "ht": float|nan, "hand": str}
    Cached after first call.
    """
    global _PLAYER_BIO_CACHE
    if _PLAYER_BIO_CACHE is not None:
        return _PLAYER_BIO_CACHE

    _PLAYER_BIO_CACHE = {}
    players_path = SACKMANN_DIR / "atp_players.csv"
    if not players_path.exists():
        return _PLAYER_BIO_CACHE

    try:
        players = pd.read_csv(players_path, dtype={"player_id": str})
        players["dob"] = pd.to_datetime(players["dob"], format="%Y%m%d", errors="coerce")
        for _, row in players.iterrows():
            pid = str(row["player_id"])
            _PLAYER_BIO_CACHE[pid] = {
                "dob": row["dob"],
                "ht": float(row["height"]) if pd.notna(row.get("height")) else np.nan,
                "hand": str(row.get("hand", "U")) if pd.notna(row.get("hand")) else "U",
            }
    except Exception:
        pass

    return _PLAYER_BIO_CACHE


# --- Court speed cache (loaded once from local cache files) ---
_COURT_SPEED_CACHE: pd.DataFrame | None = None


def _load_court_speed_data() -> pd.DataFrame:
    """Load court speed data from local cache. No network requests.

    Reads the cached JSON files that load_court_speed_history() writes.
    Returns empty DataFrame if no cached data exists.
    """
    global _COURT_SPEED_CACHE
    if _COURT_SPEED_CACHE is not None:
        return _COURT_SPEED_CACHE

    from tennis_predictor.config import CACHE_DIR

    cache_dir = CACHE_DIR / "court_speed"
    frames = []
    if cache_dir.exists():
        for f in sorted(cache_dir.glob("ta_speed_*.json")):
            try:
                data = json.loads(f.read_text())
                if data:
                    frames.append(pd.DataFrame(data))
            except Exception:
                continue

    _COURT_SPEED_CACHE = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["tournament", "surface", "year", "speed_rating"]
    )
    return _COURT_SPEED_CACHE


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
        key = tuple(sorted([m["player1"], m["player2"]]))
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

    # Regenerate site with predictions + model stats
    from tennis_predictor.web.generate import generate_site
    stats = {}
    stats_file = PROCESSED_DIR / "latest_stats.json"
    if stats_file.exists():
        stats = json.loads(stats_file.read_text())
    generate_site(predictions=predictions, model_stats=stats)

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
            rank_diff = p2_rank - p1_rank
            prob_p1 = 1.0 / (1.0 + 10 ** (-rank_diff / 250))
        else:
            return None
        model_used = "rank_fallback"
    else:
        # Try to use the TRAINED ENSEMBLE MODEL (highest accuracy)
        prob_p1, model_used = _predict_with_ensemble(match, guard_state, p1_id, p2_id, surface)

        if prob_p1 is None:
            # Fallback: Elo blend
            init = HP.elo.initial_rating
            elo1 = guard_state.elo.get(p1_id, init) if p1_id else init
            elo2 = guard_state.elo.get(p2_id, init) if p2_id else init
            surf_elo1 = guard_state.elo_surface.get((p1_id, surface), init) if p1_id else init
            surf_elo2 = guard_state.elo_surface.get((p2_id, surface), init) if p2_id else init
            elo_prob = 1.0 / (1.0 + 10 ** ((elo2 - elo1) / 400))
            surf_prob = 1.0 / (1.0 + 10 ** ((surf_elo2 - surf_elo1) / 400))
            prob_p1 = 0.4 * elo_prob + 0.6 * surf_prob
            if p1_id is None or elo1 == init:
                prob_p1 = 0.4 * prob_p1 + 0.6 * 0.5
            if p2_id is None or elo2 == init:
                prob_p1 = 0.4 * prob_p1 + 0.6 * 0.5
            model_used = "elo_blend"

    prob_p1 = max(0.05, min(0.95, prob_p1))

    # Confidence level
    confidence = abs(prob_p1 - 0.5) * 2  # 0 = toss-up, 1 = certain

    # === Build detailed analysis data ===
    detail = _build_match_detail(guard_state, p1_id, p2_id, p1_name, p2_name, surface)

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
        "model": model_used,
        "generated_at": datetime.now().isoformat(),
        "detail": detail,
    }


def _build_match_detail(guard_state, p1_id, p2_id, p1_name, p2_name, surface) -> dict:
    """Extract comprehensive stats for the match detail view."""
    if guard_state is None:
        return {}

    from tennis_predictor.hyperparams import HP
    init = HP.elo.initial_rating

    def _player_stats(pid, name):
        """Extract all available stats for a player."""
        if not pid:
            return {"name": name}

        history = guard_state.match_history.get(pid, [])
        recent = history[-10:] if history else []
        recent_20 = history[-20:] if history else []

        # Elo ratings
        elo = guard_state.elo.get(pid, init)
        surf_elo = guard_state.elo_surface.get((pid, surface), init)
        serve_elo = guard_state.elo_serve.get(pid, init)
        return_elo = guard_state.elo_return.get(pid, init)

        # Glicko-2
        g2 = guard_state.glicko2.get(pid, (init, 350, 0.06))

        # Recent form
        wins_5 = sum(1 for m in history[-5:] if m.get("won")) if len(history) >= 5 else None
        wins_10 = sum(1 for m in recent if m.get("won")) if len(recent) >= 5 else None

        # Surface form
        surf_matches = [m for m in recent_20 if m.get("surface") == surface]
        surf_wins = sum(1 for m in surf_matches if m.get("won")) if surf_matches else None
        surf_total = len(surf_matches) if surf_matches else 0

        # Serve stats (averages from last 20)
        def _avg(key):
            vals = [m.get(key) for m in recent_20 if m.get(key) is not None and not (isinstance(m.get(key), float) and np.isnan(m.get(key)))]
            return round(float(np.mean(vals)), 3) if vals else None

        # Streaks
        win_streak = 0
        for m in reversed(history):
            if m.get("won"):
                win_streak += 1
            else:
                break

        loss_streak = 0
        for m in reversed(history):
            if not m.get("won"):
                loss_streak += 1
            else:
                break

        # Days since last match
        days_since = None
        if history:
            last_date = history[-1].get("date")
            if last_date is not None:
                try:
                    days_since = (pd.Timestamp.now() - pd.Timestamp(last_date)).days
                except Exception:
                    pass

        return {
            "name": name,
            "elo": round(elo),
            "surface_elo": round(surf_elo),
            "serve_elo": round(serve_elo),
            "return_elo": round(return_elo),
            "glicko2_rating": round(g2[0]),
            "glicko2_rd": round(g2[1]),
            "total_matches": len(history),
            "form_last5": f"{wins_5}/{5}" if wins_5 is not None else None,
            "form_last10": f"{wins_10}/{len(recent)}" if wins_10 is not None else None,
            "surface_record": f"{surf_wins}/{surf_total}" if surf_wins is not None else None,
            "win_streak": win_streak,
            "loss_streak": loss_streak,
            "days_since_last": days_since,
            "first_serve_pct": _avg("first_serve_pct"),
            "first_serve_won": _avg("first_serve_won_pct"),
            "second_serve_won": _avg("second_serve_won_pct"),
            "ace_rate": _avg("ace_rate"),
            "df_rate": _avg("df_rate"),
            "bp_save_pct": _avg("bp_save_pct"),
            "return_pts_won": _avg("return_pts_won_pct"),
            "serve_pts_won": _avg("serve_pts_won_pct"),
        }

    p1_stats = _player_stats(p1_id, p1_name)
    p2_stats = _player_stats(p2_id, p2_name)

    # H2H record
    h2h = {"total": 0, "p1_wins": 0, "p2_wins": 0}
    if p1_id and p2_id:
        key = (min(p1_id, p2_id), max(p1_id, p2_id))
        h2h_data = guard_state.h2h.get(key, {})
        if h2h_data:
            total = h2h_data.get("total", 0)
            if key[0] == p1_id:
                p1w = h2h_data.get("wins_a", 0)
            else:
                p1w = h2h_data.get("wins_b", 0)
            h2h = {"total": total, "p1_wins": p1w, "p2_wins": total - p1w}

    # Key factors (what's driving this prediction)
    factors = []
    if p1_stats.get("elo") and p2_stats.get("elo"):
        diff = p1_stats["elo"] - p2_stats["elo"]
        if abs(diff) > 200:
            stronger = p1_name if diff > 0 else p2_name
            factors.append(f"Elo advantage: {stronger} ({abs(diff)} pts)")
        surf_diff = (p1_stats.get("surface_elo", 0) or 0) - (p2_stats.get("surface_elo", 0) or 0)
        if abs(surf_diff) > 150:
            stronger = p1_name if surf_diff > 0 else p2_name
            factors.append(f"Surface specialist: {stronger} on {surface}")
    if p1_stats.get("win_streak", 0) >= 5:
        factors.append(f"{p1_name} on {p1_stats['win_streak']}-match win streak")
    if p2_stats.get("win_streak", 0) >= 5:
        factors.append(f"{p2_name} on {p2_stats['win_streak']}-match win streak")
    if h2h["total"] >= 3:
        leader = p1_name if h2h["p1_wins"] > h2h["p2_wins"] else p2_name
        factors.append(f"H2H: {leader} leads {max(h2h['p1_wins'],h2h['p2_wins'])}-{min(h2h['p1_wins'],h2h['p2_wins'])}")
    if p1_stats.get("days_since_last") and p1_stats["days_since_last"] > 14:
        factors.append(f"Rust risk: {p1_name} hasn't played in {p1_stats['days_since_last']} days")
    if p2_stats.get("days_since_last") and p2_stats["days_since_last"] > 14:
        factors.append(f"Rust risk: {p2_name} hasn't played in {p2_stats['days_since_last']} days")

    return {
        "p1": p1_stats,
        "p2": p2_stats,
        "h2h": h2h,
        "factors": factors[:5],  # Top 5 key factors
    }


def _predict_with_ensemble(
    match: dict, guard_state, p1_id: str | None, p2_id: str | None, surface: str,
) -> tuple[float | None, str]:
    """Try to predict using the trained ensemble/CatBoost model.

    Builds a feature vector from the guard state and runs it through
    the trained model. Returns (probability, model_name) or (None, "") if fails.
    """
    try:
        # Load model (cached after first call)
        model = _get_cached_model()
        if model is None:
            return None, ""

        # Build a mock match Series that the TemporalGuard can extract features from
        from tennis_predictor.temporal.guard import TemporalGuard
        mock_guard = TemporalGuard(state=guard_state)

        # --- Look up player bio (age, height, handedness) from Sackmann data ---
        now = pd.Timestamp.now()
        bio = _load_player_bio()
        p1_bio = bio.get(p1_id, {}) if p1_id else {}
        p2_bio = bio.get(p2_id, {}) if p2_id else {}

        p1_age = np.nan
        if p1_bio.get("dob") is not None and pd.notna(p1_bio["dob"]):
            p1_age = (now - p1_bio["dob"]).days / 365.25
        p2_age = np.nan
        if p2_bio.get("dob") is not None and pd.notna(p2_bio["dob"]):
            p2_age = (now - p2_bio["dob"]).days / 365.25

        p1_ht = p1_bio.get("ht", np.nan)
        p2_ht = p2_bio.get("ht", np.nan)
        p1_hand = p1_bio.get("hand", "U")
        p2_hand = p2_bio.get("hand", "U")

        # --- Look up court speed from local cache ---
        court_speed = np.nan
        tourney_name = match.get("tournament", "")
        try:
            from tennis_predictor.data.court_speed import get_tournament_speed
            speed_data = _load_court_speed_data()
            if len(speed_data) > 0:
                court_speed = get_tournament_speed(tourney_name, now.year, speed_data)
        except Exception:
            pass

        match_series = pd.Series({
            "match_id": f"live_{match.get('player1', '')}_{match.get('player2', '')}",
            "p1_id": p1_id or "",
            "p2_id": p2_id or "",
            "tourney_date": now,
            "tourney_name": tourney_name,
            "tourney_level": "A",
            "surface": surface,
            "round": "R32",
            "best_of": 3,
            "draw_size": 32,
            "p1_rank": match.get("p1_rank", np.nan),
            "p2_rank": match.get("p2_rank", np.nan),
            "p1_rank_points": np.nan,
            "p2_rank_points": np.nan,
            "p1_age": p1_age,
            "p2_age": p2_age,
            "p1_ht": p1_ht,
            "p2_ht": p2_ht,
            "p1_entry": "",
            "p2_entry": "",
            "p1_seed": "",
            "p2_seed": "",
            "p1_hand": p1_hand,
            "p2_hand": p2_hand,
            "p1_ioc": "",
            "p2_ioc": "",
            "minutes": np.nan,
            "retirement": False,
            "n_sets": np.nan,
            "y": 0,
            # Stats (not available for upcoming matches)
            "w_ace": np.nan, "w_df": np.nan, "w_svpt": np.nan,
            "w_1stIn": np.nan, "w_1stWon": np.nan, "w_2ndWon": np.nan,
            "w_SvGms": np.nan, "w_bpSaved": np.nan, "w_bpFaced": np.nan,
            "l_ace": np.nan, "l_df": np.nan, "l_svpt": np.nan,
            "l_1stIn": np.nan, "l_1stWon": np.nan, "l_2ndWon": np.nan,
            "l_SvGms": np.nan, "l_bpSaved": np.nan, "l_bpFaced": np.nan,
            # Supplementary
            "court_speed": court_speed,
            "weather_temp_max": np.nan, "weather_temp_min": np.nan,
            "weather_precipitation": np.nan, "weather_wind_max": np.nan,
            "weather_wind_gust_max": np.nan, "weather_altitude": np.nan,
            "weather_is_indoor": np.nan,
            "odds_implied_p1": np.nan, "odds_implied_p2": np.nan,
            "odds_diff": np.nan, "elo_vs_odds_diff": np.nan,
            "intransitivity_score": np.nan,
            "p1_sentiment": np.nan, "p2_sentiment": np.nan,
            "p1_injury_signal": np.nan, "p2_injury_signal": np.nan,
            "p1_momentum_signal": np.nan, "p2_momentum_signal": np.nan,
            "sentiment_diff": np.nan,
            "line_direction": np.nan, "line_magnitude": np.nan,
            "sharp_signal": np.nan, "opening_implied_p1": np.nan,
            "current_implied_p1": np.nan,
        })

        # Extract features using the guard (reads from Elo/Glicko/history state)
        # We need to bypass the duplicate check since this is a live prediction
        mock_guard._processed_match_ids = set()  # Allow processing
        mock_guard._last_extracted_match = None
        features = mock_guard.extract_pre_match_state(match_series)

        # Build DataFrame and align columns to model's expected order
        feature_df = pd.DataFrame([features])

        # Try to load selected features from pipeline (Change 5: feature selection)
        selected_features_path = PROCESSED_DIR / "selected_features.json"
        expected_cols = None
        if selected_features_path.exists():
            try:
                expected_cols = json.loads(selected_features_path.read_text())
            except Exception:
                pass

        # Fallback: get expected features from the first base model
        if expected_cols is None:
            if hasattr(model, "final_base_models_") and model.final_base_models_:
                base_model = model.final_base_models_[0][1]
                if hasattr(base_model, "feature_names_"):
                    expected_cols = base_model.feature_names_

        if expected_cols is not None:
            for col in expected_cols:
                if col not in feature_df.columns:
                    feature_df[col] = np.nan
            feature_df = feature_df[expected_cols]

        # Run through model
        proba = model.predict_proba(feature_df)
        prob_p1 = float(proba[0, 1])

        model_name = type(model).__name__.lower()
        if "stacking" in model_name or "ensemble" in model_name:
            model_name = "stacking_ensemble"
        elif "catboost" in model_name:
            model_name = "catboost"
        elif "xgboost" in model_name or "xgb" in model_name:
            model_name = "xgboost"
        else:
            model_name = "trained_model"

        return prob_p1, model_name

    except Exception as e:
        # Silently fall back to Elo blend
        return None, ""


_MODEL_CACHE = {"model": None, "loaded": False}


def _get_cached_model():
    """Load and cache the trained model."""
    if _MODEL_CACHE["loaded"]:
        return _MODEL_CACHE["model"]

    _MODEL_CACHE["loaded"] = True

    for name in ["model_ensemble.pkl", "model_catboost.pkl", "model_xgboost.pkl"]:
        path = PROCESSED_DIR / name
        if path.exists():
            try:
                with open(path, "rb") as f:
                    model = pickle.load(f)
                _MODEL_CACHE["model"] = model
                print(f"  Loaded trained model: {name}")
                return model
            except Exception:
                continue

    return None


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
