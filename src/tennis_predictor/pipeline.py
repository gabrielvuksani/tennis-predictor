"""Full prediction pipeline — integrates all data sources and trains the ensemble.

This is the main orchestrator that:
1. Loads match data (JeffSackmann)
2. Merges betting odds (tennis-data.co.uk)
3. Adds weather data (Open-Meteo)
4. Adds court speed data (Tennis Abstract)
5. Computes intransitivity scores (GNN-lite)
6. Builds features chronologically via TemporalGuard
7. Trains the stacking ensemble
8. Evaluates with full metrics
9. Generates the static site
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from tennis_predictor.config import PROCESSED_DIR, CACHE_DIR
from tennis_predictor.hyperparams import HP


def run_full_pipeline(
    start_year: int = 1991,
    end_year: int | None = None,
    tour_level_only: bool = False,
    test_year: int = 2023,
    fetch_weather: bool = True,
    fetch_court_speed: bool = True,
    fetch_odds: bool = True,
    compute_intransitivity: bool = True,
) -> dict:
    """Run the complete pipeline end-to-end.

    Args:
        start_year: First year of match data.
        end_year: Last year (None = all available).
        tour_level_only: If True, exclude challengers/futures for fair comparison.
        test_year: Year to start the test set.
        fetch_weather: Whether to fetch weather data.
        fetch_court_speed: Whether to fetch court speed data.
        fetch_odds: Whether to fetch/merge betting odds.
        compute_intransitivity: Whether to compute intransitivity scores.

    Returns:
        Dict with evaluation results and model paths.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # === STEP 1: Load match data ===
    print("=" * 60)
    print("STEP 1: Loading match data")
    print("=" * 60)
    from tennis_predictor.data.sackmann import clone_or_update_repo, load_matches, create_pairwise_rows

    clone_or_update_repo()
    matches = load_matches(
        start_year=start_year,
        end_year=end_year,
        include_qual_chall=not tour_level_only,
        include_futures=False,
    )

    if tour_level_only:
        matches = matches[matches["tourney_level"].isin(["G", "M", "F", "A", "250", "500", "1000"])].copy()
        matches = matches.reset_index(drop=True)
        print(f"Filtered to tour-level: {len(matches):,} matches")

    # === STEP 2: Merge betting odds ===
    if fetch_odds:
        print("\n" + "=" * 60)
        print("STEP 2: Merging betting odds")
        print("=" * 60)
        try:
            from tennis_predictor.data.odds_merge import merge_odds_with_matches
            matches = merge_odds_with_matches(matches)
            odds_count = matches["odds_implied_w"].notna().sum()
            print(f"Odds successfully merged: {odds_count:,} matches with odds")
            if odds_count == 0:
                print("WARNING: Odds merge produced 0 matches — check tennis-data.co.uk availability")
        except Exception as e:
            print(f"WARNING: Odds merge failed: {e}")
            print("Continuing without odds data.")
    if "odds_implied_w" not in matches.columns:
        matches["odds_implied_w"] = np.nan
        matches["odds_implied_l"] = np.nan
        matches["odds_pinnacle_w"] = np.nan
        matches["odds_pinnacle_l"] = np.nan

    # === STEP 3: Add court speed ===
    if fetch_court_speed:
        print("\n" + "=" * 60)
        print("STEP 3: Adding court speed data")
        print("=" * 60)
        matches = _add_court_speed(matches)
    else:
        matches["court_speed"] = np.nan

    # === STEP 4: Add weather (batched by tournament) ===
    if fetch_weather:
        print("\n" + "=" * 60)
        print("STEP 4: Adding weather data")
        print("=" * 60)
        matches = _add_weather(matches)
    else:
        for col in ["weather_temp_max", "weather_temp_min", "weather_precipitation",
                     "weather_wind_max", "weather_wind_gust_max", "weather_altitude",
                     "weather_is_indoor"]:
            matches[col] = np.nan

    # === STEP 5: Create pairwise rows ===
    print("\n" + "=" * 60)
    print("STEP 5: Creating pairwise match rows")
    print("=" * 60)
    pairwise = create_pairwise_rows(matches)

    # Carry over stat columns and supplementary data
    stat_cols = [c for c in matches.columns if c.startswith(("w_", "l_"))]
    supp_cols = ["court_speed", "weather_temp_max", "weather_temp_min",
                 "weather_precipitation", "weather_wind_max", "weather_wind_gust_max",
                 "weather_altitude", "weather_is_indoor"]
    for col in stat_cols + supp_cols:
        if col in matches.columns:
            pairwise[col] = matches[col].values

    # Map odds to pairwise format (need to swap based on who is p1)
    rng = np.random.RandomState(42)
    swap = rng.randint(0, 2, size=len(matches)).astype(bool)
    pairwise["odds_implied_p1"] = np.where(
        swap, matches["odds_implied_w"].values, matches["odds_implied_l"].values
    )
    pairwise["odds_implied_p2"] = np.where(
        swap, matches["odds_implied_l"].values, matches["odds_implied_w"].values
    )
    # Keep raw odds for ROI simulation
    pairwise["odds_decimal_p1"] = np.where(
        swap, matches.get("odds_pinnacle_w", pd.Series(np.nan, index=matches.index)).values,
        matches.get("odds_pinnacle_l", pd.Series(np.nan, index=matches.index)).values
    )
    pairwise["odds_decimal_p2"] = np.where(
        swap, matches.get("odds_pinnacle_l", pd.Series(np.nan, index=matches.index)).values,
        matches.get("odds_pinnacle_w", pd.Series(np.nan, index=matches.index)).values
    )

    # === STEP 6: Compute intransitivity scores ===
    if compute_intransitivity:
        print("\n" + "=" * 60)
        print("STEP 6: Computing intransitivity scores")
        print("=" * 60)
        pairwise = _compute_intransitivity(pairwise, matches)

    # === STEP 7: Build features chronologically ===
    print("\n" + "=" * 60)
    print("STEP 7: Building features chronologically")
    print("=" * 60)
    from tennis_predictor.temporal.guard import TemporalGuard
    from tennis_predictor.temporal.validation import build_features_chronologically

    guard = TemporalGuard()
    X, y, guard = build_features_chronologically(pairwise, guard=guard)

    # Save intermediate outputs
    X.to_parquet(PROCESSED_DIR / "features_full.parquet", index=False)
    np.save(PROCESSED_DIR / "targets_full.npy", y)
    pairwise[["match_id", "tourney_date", "tourney_level", "p1_name", "p2_name",
              "odds_decimal_p1", "odds_decimal_p2"]].to_parquet(
        PROCESSED_DIR / "match_metadata.parquet", index=False
    )
    with open(PROCESSED_DIR / "guard_state_full.pkl", "wb") as f:
        pickle.dump(guard.state, f)

    print(f"Feature matrix: {X.shape}")
    print(f"Target balance: {y.mean():.4f}")

    # === STEP 7b: NaN rate logging (informational only) ===
    # Note: GBMs handle NaN natively — explicit filtering hurts accuracy.
    nan_rates = X.isna().mean()
    high_nan = nan_rates[nan_rates > 0.5]
    if len(high_nan):
        print(f"\n{len(high_nan)} features have >50% NaN (handled natively by GBMs):")

    # === STEP 8: Train and evaluate ===
    print("\n" + "=" * 60)
    print("STEP 8: Training and evaluating models")
    print("=" * 60)
    results = _train_and_evaluate(X, y, pairwise, test_year=test_year)

    # === STEP 9: Generate site ===
    print("\n" + "=" * 60)
    print("STEP 9: Generating site")
    print("=" * 60)
    from tennis_predictor.web.generate import generate_site
    generate_site(
        model_stats=results.get("best_stats", {}),
        calibration_data=results.get("calibration", {}),
    )
    print("Site generated at site/")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"{'=' * 60}")

    return results


def _add_court_speed(matches: pd.DataFrame) -> pd.DataFrame:
    """Add court speed data from Tennis Abstract."""
    from tennis_predictor.data.court_speed import load_court_speed_history, get_tournament_speed

    speed_data = load_court_speed_history(
        start_year=max(2010, matches["tourney_date"].dt.year.min()),
        end_year=matches["tourney_date"].dt.year.max(),
    )

    if len(speed_data) == 0:
        print("Warning: No court speed data available.")
        matches["court_speed"] = np.nan
        return matches

    print(f"Loaded {len(speed_data)} court speed records")

    speeds = []
    for _, row in tqdm(matches.iterrows(), total=len(matches), desc="Mapping court speed",
                       mininterval=2):
        year = row["tourney_date"].year if not pd.isna(row["tourney_date"]) else 0
        name = str(row.get("tourney_name", ""))
        speeds.append(get_tournament_speed(name, year, speed_data))

    matches["court_speed"] = speeds
    has_speed = pd.Series(speeds).notna().sum()
    print(f"Matches with court speed: {has_speed:,} / {len(matches):,} ({has_speed/len(matches):.1%})")

    return matches


def _add_weather(matches: pd.DataFrame, min_weather_year: int = 2015) -> pd.DataFrame:
    """Add weather data, batched by tournament to minimize API calls.

    Only fetches weather for tournaments from min_weather_year onward to avoid
    excessive API calls for historical data that won't meaningfully impact training.
    """
    from tennis_predictor.data.weather import get_match_weather

    weather_cols = ["weather_temp_max", "weather_temp_min", "weather_precipitation",
                    "weather_wind_max", "weather_wind_gust_max", "weather_altitude",
                    "weather_is_indoor"]

    # Only fetch weather for recent tournaments
    recent_mask = matches["tourney_date"].dt.year >= min_weather_year
    tournaments = matches[recent_mask].groupby("tourney_id").first()[
        ["tourney_name", "tourney_date"]
    ].reset_index()
    print(f"Fetching weather for {len(tournaments)} tournaments ({min_weather_year}+)")

    # Cache weather per tournament
    weather_cache: dict[str, dict] = {}
    cache_file = CACHE_DIR / "weather_tourney_cache.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if cache_file.exists():
        weather_cache = json.loads(cache_file.read_text())

    fetched = 0
    for _, t in tqdm(tournaments.iterrows(), total=len(tournaments),
                     desc="Fetching weather", mininterval=2):
        tid = str(t["tourney_id"])
        if tid in weather_cache:
            continue

        weather = get_match_weather(
            str(t["tourney_name"]),
            pd.Timestamp(t["tourney_date"]),
        )
        weather_cache[tid] = weather
        fetched += 1

        # Save cache periodically
        if fetched % 100 == 0:
            cache_file.write_text(json.dumps(weather_cache, default=str))

    # Save final cache
    cache_file.write_text(json.dumps(weather_cache, default=str))
    print(f"Fetched weather for {fetched} new tournaments ({len(weather_cache)} total cached)")

    # Map weather to matches
    for col in weather_cols:
        matches[col] = matches["tourney_id"].map(
            lambda tid: weather_cache.get(str(tid), {}).get(col, np.nan)
        )

    has_weather = matches["weather_temp_max"].notna().sum()
    print(f"Matches with weather: {has_weather:,} / {len(matches):,} ({has_weather/len(matches):.1%})")

    return matches


def _compute_intransitivity(pairwise: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Compute intransitivity scores for each matchup.

    This is a lightweight version of the GNN approach from Clegg (2025).
    Instead of a full GNN, we compute a graph-based intransitivity metric:

    For players A and B about to play:
    1. Find all common opponents C (players both A and B have played recently)
    2. For each C: check if A beat C and C beat B (or vice versa) — this forms
       a potential intransitive triple
    3. The intransitivity score = fraction of common opponents creating cycles

    High intransitivity = the matchup is hard to predict from rankings alone,
    which is where bookmakers are weakest.
    """
    print("Computing intransitivity from match history...")

    # Build win graph from recent matches (last 2 years)
    max_date = pairwise["tourney_date"].max()
    lookback_days = HP.pipeline.intransitivity_lookback_days
    cutoff = max_date - pd.Timedelta(days=lookback_days)

    # We'll compute intransitivity in a rolling fashion
    # For efficiency, pre-build adjacency info from original matches
    win_graph: dict[str, set[str]] = {}  # player -> set of players they beat

    scores = np.full(len(pairwise), np.nan)
    window_matches: list[tuple[str, str, pd.Timestamp]] = []  # (winner, loser, date)

    for idx, row in tqdm(pairwise.iterrows(), total=len(pairwise),
                         desc="Intransitivity", mininterval=2):
        p1 = str(row.get("p1_id", ""))
        p2 = str(row.get("p2_id", ""))
        match_date = row.get("tourney_date", pd.NaT)

        if not p1 or not p2 or pd.isna(match_date):
            continue

        # Remove old matches from window
        cutoff_date = match_date - pd.Timedelta(days=lookback_days)
        while window_matches and window_matches[0][2] < cutoff_date:
            old_w, old_l, _ = window_matches.pop(0)
            if old_w in win_graph:
                win_graph[old_w].discard(old_l)

        # Find common opponents
        p1_beaten = win_graph.get(p1, set())
        p2_beaten = win_graph.get(p2, set())

        # Players who beat p1 or p2
        p1_lost_to = {w for w, beaten in win_graph.items() if p1 in beaten}
        p2_lost_to = {w for w, beaten in win_graph.items() if p2 in beaten}

        # Common opponents: players both have played
        p1_opponents = p1_beaten | p1_lost_to
        p2_opponents = p2_beaten | p2_lost_to
        common = p1_opponents & p2_opponents

        if len(common) >= 2:
            cycles = 0
            for c in common:
                # Check for cycle: A > C > B or B > C > A
                a_beat_c = c in p1_beaten
                c_beat_b = p2 in win_graph.get(c, set())
                b_beat_c = c in p2_beaten
                c_beat_a = p1 in win_graph.get(c, set())

                if (a_beat_c and c_beat_b) or (b_beat_c and c_beat_a):
                    # This is a transitive chain, not intransitive
                    pass
                elif (a_beat_c and b_beat_c) or (c_beat_a and c_beat_b):
                    # Both beat C or both lost to C — no cycle
                    pass
                else:
                    # Intransitive: A>C but B>C in opposite direction
                    cycles += 1

            scores[idx] = cycles / len(common)

        # Update win graph with this match result
        y = row.get("y", np.nan)
        if not pd.isna(y):
            winner = p1 if y == 1 else p2
            loser = p2 if y == 1 else p1
            if winner not in win_graph:
                win_graph[winner] = set()
            win_graph[winner].add(loser)
            window_matches.append((winner, loser, match_date))

    pairwise["intransitivity_score"] = scores
    has_score = pd.Series(scores).notna().sum()
    print(f"Matches with intransitivity: {has_score:,} / {len(pairwise):,}")

    return pairwise


def _train_and_evaluate(
    X: pd.DataFrame,
    y: np.ndarray,
    pairwise: pd.DataFrame,
    test_year: int = 2023,
) -> dict:
    """Train all models and evaluate comprehensively."""
    from tennis_predictor.models.baseline import EloBaseline, RankBaseline
    from tennis_predictor.models.gbm import XGBoostPredictor, LightGBMPredictor, CatBoostPredictor
    from tennis_predictor.models.ensemble import StackingEnsemble
    from tennis_predictor.evaluation.metrics import (
        full_evaluation, compare_models, calibration_curve,
        brier_score, accuracy, log_loss, upset_metrics, roi_simulation,
    )

    # Temporal split
    dates = pairwise["tourney_date"]
    train_mask = dates < f"{test_year}-01-01"
    test_mask = dates >= f"{test_year}-01-01"

    X_train_raw, y_train_raw = X[train_mask.values], y[train_mask.values]
    X_test, y_test = X[test_mask.values], y[test_mask.values]

    # Filter retirements from training (they are noise — the result doesn't reflect skill)
    train_pairwise_all = pairwise[train_mask.values]
    retirement_mask = train_pairwise_all.get("retirement", pd.Series(False)).values.astype(bool)
    non_ret = ~retirement_mask
    X_train = X_train_raw[non_ret]
    y_train = y_train_raw[non_ret]

    # Save feature names for live prediction alignment
    selected_features_path = PROCESSED_DIR / "selected_features.json"
    with open(selected_features_path, "w") as f:
        json.dump(list(X_train.columns), f, indent=2)
    print(f"Saved {X_train.shape[1]} feature names for live prediction alignment")

    # Time-decay sample weighting: recent matches matter more
    train_dates = train_pairwise_all.loc[non_ret, "tourney_date"]
    max_train_date = train_dates.max()
    days_ago = (max_train_date - train_dates).dt.days.values.astype(float)
    gamma = HP.pipeline.time_decay_gamma  # Per-day decay (~1 year half-life)
    sample_weights = gamma ** days_ago
    sample_weights = sample_weights / sample_weights.mean()  # Normalize to mean=1

    print(f"Train: {len(X_train):,} matches (up to {test_year-1}, {retirement_mask.sum():,} retirements excluded)")
    print(f"Test:  {len(X_test):,} matches ({test_year}+)")

    # Detect upsets: player with higher rank number (lower ranked) wins
    test_pairwise = pairwise[test_mask.values]
    is_upset = np.zeros(len(y_test), dtype=bool)
    p1_rank = test_pairwise.get("p1_rank", pd.Series(np.nan))
    p2_rank = test_pairwise.get("p2_rank", pd.Series(np.nan))
    # Upset = lower-ranked player (higher rank number) won
    for i, (r1, r2, result) in enumerate(zip(p1_rank.values, p2_rank.values, y_test)):
        if not pd.isna(r1) and not pd.isna(r2):
            if (r1 > r2 and result == 1) or (r2 > r1 and result == 0):
                is_upset[i] = True

    # Get odds for ROI simulation
    odds_p1 = test_pairwise.get("odds_decimal_p1", pd.Series(np.nan)).values
    odds_p2 = test_pairwise.get("odds_decimal_p2", pd.Series(np.nan)).values

    evaluations = []

    # === Baselines ===
    print("\nTraining baselines...")
    for name, model_cls in [("Elo Baseline", EloBaseline), ("Rank Baseline", RankBaseline)]:
        model = model_cls()
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_test)[:, 1]
        ev = full_evaluation(y_test, pred, is_upset=is_upset,
                             odds_p1=odds_p1, odds_p2=odds_p2, label=name)
        evaluations.append(ev)

    # === Individual GBMs (with time-decay sample weighting) ===
    models = {}
    for name, cls in [("XGBoost", XGBoostPredictor),
                      ("LightGBM", LightGBMPredictor),
                      ("CatBoost", CatBoostPredictor)]:
        print(f"Training {name}...")
        t0 = time.time()
        model = cls()
        try:
            model.fit(X_train, y_train, sample_weight=sample_weights)
        except TypeError:
            model.fit(X_train, y_train)
        pred = model.predict_proba(X_test)[:, 1]
        models[name] = (model, pred)
        ev = full_evaluation(y_test, pred, is_upset=is_upset,
                             odds_p1=odds_p1, odds_p2=odds_p2, label=name)
        evaluations.append(ev)
        print(f"  {name}: acc={ev['accuracy']:.4f}, brier={ev['brier_score']:.4f} ({time.time()-t0:.1f}s)")

    # === Stacking Ensemble ===
    print("Training Stacking Ensemble...")
    t0 = time.time()
    ensemble = StackingEnsemble(calibrate=True)
    ensemble.fit(X_train, y_train)
    ensemble_pred = ensemble.predict_proba(X_test)[:, 1]
    ev_ensemble = full_evaluation(y_test, ensemble_pred, is_upset=is_upset,
                                  odds_p1=odds_p1, odds_p2=odds_p2, label="Stacking Ensemble")
    evaluations.append(ev_ensemble)
    print(f"  Ensemble: acc={ev_ensemble['accuracy']:.4f}, brier={ev_ensemble['brier_score']:.4f} ({time.time()-t0:.1f}s)")

    # === Results ===
    print("\n" + "=" * 70)
    table = compare_models(evaluations)
    print(table.to_string(index=False))
    print("=" * 70)

    print("\nBenchmarks:")
    print("  phosphenq (post-fix):   66.3% accuracy")
    print("  IBM SlamTracker:        63.8% accuracy")
    print("  Bookmakers:            ~72% accuracy, ~0.196 Brier")

    # Upset analysis
    if is_upset.any():
        print(f"\nUpset Analysis (n={is_upset.sum():,} upsets, {is_upset.mean():.1%} of test):")
        for ev in evaluations:
            um = ev.get("upset_metrics", {})
            if um.get("n_upsets", 0) > 0:
                print(f"  {ev['label']}: upset_acc={um['upset_accuracy']:.3f}, "
                      f"detection={um['upset_detection_rate']:.3f}")

    # ROI analysis
    print("\nROI Analysis (Kelly fraction=0.25, min_edge=5%):")
    for ev in evaluations:
        roi = ev.get("roi", {})
        if roi.get("n_bets", 0) > 0:
            print(f"  {ev['label']}: {roi['n_bets']} bets, ROI={roi['roi_pct']:.1f}%, "
                  f"win_rate={roi['win_rate']:.1%}")

    # Feature importance from best single model
    best_single = models.get("CatBoost", models.get("XGBoost", (None, None)))
    if best_single[0] and hasattr(best_single[0], "feature_importances"):
        print("\nTop 20 Features (CatBoost):")
        for i, (name, imp) in enumerate(list(best_single[0].feature_importances.items())[:20]):
            print(f"  {i+1:2d}. {name}: {imp:.4f}")

    # Stratified evaluation
    from tennis_predictor.evaluation.metrics import stratified_evaluation, print_stratified_evaluation
    test_meta = test_pairwise[["tourney_date", "tourney_level"]].copy()
    test_meta["surface"] = pairwise[test_mask.values]["surface"].values if "surface" in pairwise.columns else "Unknown"
    test_meta["round"] = pairwise[test_mask.values]["round"].values if "round" in pairwise.columns else "Unknown"
    strat = stratified_evaluation(y_test, ensemble_pred, test_meta, label="Stacking Ensemble")
    print_stratified_evaluation(strat)

    # Save best model
    model_path = PROCESSED_DIR / "model_ensemble.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(ensemble, f)
    print(f"\nEnsemble saved to {model_path}")

    # Prepare return data
    best_stats = {
        "accuracy": ev_ensemble["accuracy"],
        "brier_score": ev_ensemble["brier_score"],
        "ece": ev_ensemble["ece"],
        "n_matches": ev_ensemble["n_matches"],
    }

    return {
        "evaluations": evaluations,
        "best_stats": best_stats,
        "calibration": ev_ensemble["calibration"],
        "ensemble_weights": ensemble.base_model_weights,
        "stratified": strat,
    }
