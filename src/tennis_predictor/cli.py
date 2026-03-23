"""Command-line interface for the tennis predictor."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

from tennis_predictor.config import (
    PROCESSED_DIR,
    PREDICTIONS_DIR,
    RAW_DIR,
    SACKMANN_DIR,
)


@click.group()
def main():
    """Tennis Predictor — State-of-the-art ATP match prediction."""
    pass


@main.command()
@click.option("--start-year", default=1991, help="First year to load")
@click.option("--end-year", default=None, type=int, help="Last year to load")
def ingest(start_year: int, end_year: int | None):
    """Download and ingest all data sources."""
    from tennis_predictor.data.sackmann import clone_or_update_repo, load_matches

    click.echo("=== Data Ingestion ===\n")

    # 1. Clone/update JeffSackmann repo
    click.echo("Step 1: Fetching JeffSackmann tennis_atp data...")
    clone_or_update_repo()

    # 2. Load and clean matches
    click.echo(f"\nStep 2: Loading matches ({start_year}-{end_year or 'latest'})...")
    matches = load_matches(start_year=start_year, end_year=end_year)

    # Save processed matches
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output = PROCESSED_DIR / "matches.parquet"
    matches.to_parquet(output, index=False)
    click.echo(f"\nSaved {len(matches):,} matches to {output}")

    # 3. Summary stats
    click.echo(f"\n=== Data Summary ===")
    click.echo(f"Matches: {len(matches):,}")
    click.echo(f"Date range: {matches['tourney_date'].min()} to {matches['tourney_date'].max()}")
    click.echo(f"Unique players: {pd.concat([matches['winner_id'], matches['loser_id']]).nunique():,}")
    click.echo(f"Surfaces: {matches['surface'].value_counts().to_dict()}")
    click.echo(f"Tournament levels: {matches['tourney_level'].value_counts().to_dict()}")

    # Stats completeness
    has_stats = matches["w_ace"].notna().sum()
    click.echo(f"Matches with stats: {has_stats:,} ({has_stats/len(matches)*100:.1f}%)")


@main.command()
@click.option("--start-year", default=1991, help="First year for features")
@click.option("--end-year", default=None, type=int, help="Last year")
def build_features(start_year: int, end_year: int | None):
    """Build the feature matrix chronologically (respecting temporal boundaries)."""
    from tennis_predictor.data.sackmann import create_pairwise_rows, load_matches
    from tennis_predictor.temporal.guard import TemporalGuard
    from tennis_predictor.temporal.validation import build_features_chronologically

    click.echo("=== Building Feature Matrix ===\n")

    # Load matches
    matches_file = PROCESSED_DIR / "matches.parquet"
    if matches_file.exists():
        click.echo("Loading cached matches...")
        matches = pd.read_parquet(matches_file)
    else:
        click.echo("Loading matches from source...")
        matches = load_matches(start_year=start_year, end_year=end_year)

    # Create pairwise rows (random player1/player2 assignment)
    click.echo("Creating pairwise match rows...")
    pairwise = create_pairwise_rows(matches)

    # Merge original stats columns for feature computation
    stat_cols = [c for c in matches.columns if c.startswith(("w_", "l_"))]
    for col in stat_cols:
        pairwise[col] = matches[col].values

    click.echo(f"Processing {len(pairwise):,} matches chronologically...")
    click.echo("(This computes Elo, rolling stats, H2H, etc. — may take a few minutes)\n")

    # Build features with TemporalGuard
    guard = TemporalGuard()
    X, y, guard = build_features_chronologically(pairwise, guard=guard)

    # Save outputs
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
    np.save(PROCESSED_DIR / "targets.npy", y)
    with open(PROCESSED_DIR / "guard_state.pkl", "wb") as f:
        pickle.dump(guard.state, f)

    click.echo(f"\n=== Feature Matrix Built ===")
    click.echo(f"Samples: {len(X):,}")
    click.echo(f"Features: {X.shape[1]}")
    click.echo(f"Feature names: {list(X.columns[:20])}...")
    click.echo(f"Target balance: {y.mean():.3f} (should be ~0.5)")
    click.echo(f"Guard stats: {guard.stats}")
    click.echo(f"\nSaved to {PROCESSED_DIR}")


@main.command()
@click.option("--model", "model_type", default="ensemble",
              type=click.Choice(["elo", "xgboost", "lightgbm", "catboost", "ensemble"]))
@click.option("--validate/--no-validate", default=True, help="Run temporal validation")
def train(model_type: str, validate: bool):
    """Train the prediction model."""
    from tennis_predictor.evaluation.metrics import (
        accuracy,
        brier_score,
        compare_models,
        full_evaluation,
    )
    from tennis_predictor.models.baseline import EloBaseline
    from tennis_predictor.models.ensemble import StackingEnsemble, create_default_ensemble
    from tennis_predictor.models.gbm import (
        CatBoostPredictor,
        LightGBMPredictor,
        XGBoostPredictor,
    )
    from tennis_predictor.temporal.validation import (
        generate_temporal_folds,
        temporal_backtest,
    )

    click.echo("=== Model Training ===\n")

    # Load features
    X = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    y = np.load(PROCESSED_DIR / "targets.npy")

    click.echo(f"Loaded {len(X):,} samples with {X.shape[1]} features")

    # Select model
    model_map = {
        "elo": lambda: EloBaseline(),
        "xgboost": lambda: XGBoostPredictor(),
        "lightgbm": lambda: LightGBMPredictor(),
        "catboost": lambda: CatBoostPredictor(),
        "ensemble": lambda: create_default_ensemble(),
    }

    if validate:
        click.echo("\nRunning temporal validation...")

        # Create a temporary DataFrame with dates for fold generation
        dates_file = PROCESSED_DIR / "matches.parquet"
        if dates_file.exists():
            dates_df = pd.read_parquet(dates_file)[["tourney_date"]]
            # Align indices
            dates_df = dates_df.iloc[:len(X)].reset_index(drop=True)
            X = X.reset_index(drop=True)
        else:
            click.echo("Warning: No date info for folds. Using simple split.")
            validate = False

    if validate:
        folds = generate_temporal_folds(dates_df)
        click.echo(f"Generated {len(folds)} temporal folds\n")

        results = temporal_backtest(
            pd.DataFrame({"tourney_date": dates_df["tourney_date"]}),
            model_factory=model_map[model_type],
            folds=folds,
        )

        # Aggregate results
        all_y_true = np.concatenate([r["y_true"] for r in results])
        all_y_pred = np.concatenate([r["y_pred_proba"] for r in results])

        ev = full_evaluation(all_y_true, all_y_pred, label=model_type)
        click.echo(f"\n=== Temporal Validation Results ({model_type}) ===")
        click.echo(f"Accuracy:    {ev['accuracy']:.4f}")
        click.echo(f"Brier Score: {ev['brier_score']:.4f}")
        click.echo(f"Log Loss:    {ev['log_loss']:.4f}")
        click.echo(f"ECE:         {ev['ece']:.4f}")

    # Train final model on all data
    click.echo(f"\nTraining final {model_type} model on all data...")
    model = model_map[model_type]()
    model.fit(X, y)

    # Save model
    model_path = PROCESSED_DIR / f"model_{model_type}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    click.echo(f"Model saved to {model_path}")

    # Feature importance (for GBM models)
    if hasattr(model, "feature_importances"):
        importances = model.feature_importances
        if importances:
            click.echo(f"\nTop 20 features:")
            for i, (name, imp) in enumerate(list(importances.items())[:20]):
                click.echo(f"  {i+1:2d}. {name}: {imp:.4f}")


@main.command()
@click.option("--model", "model_type", default="ensemble")
def predict(model_type: str):
    """Generate predictions for upcoming matches."""
    click.echo("=== Generating Predictions ===\n")

    model_path = PROCESSED_DIR / f"model_{model_type}.pkl"
    if not model_path.exists():
        click.echo(f"Error: No trained model found at {model_path}")
        click.echo("Run 'tennis-predict train' first.")
        sys.exit(1)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    click.echo(f"Loaded {model_type} model")
    click.echo("Prediction generation for live matches requires The Odds API key.")
    click.echo("Set ODDS_API_KEY environment variable to enable live predictions.")


@main.command()
def evaluate():
    """Run comprehensive evaluation of all models."""
    from tennis_predictor.evaluation.metrics import compare_models, full_evaluation
    from tennis_predictor.models.baseline import EloBaseline, RankBaseline

    click.echo("=== Comprehensive Model Evaluation ===\n")

    X = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    y = np.load(PROCESSED_DIR / "targets.npy")

    # Use last 20% as hold-out test set (temporal)
    split = int(len(X) * 0.8)
    X_test = X.iloc[split:]
    y_test = y[split:]

    evaluations = []

    # Baselines
    for name, model_cls in [("Rank Baseline", RankBaseline), ("Elo Baseline", EloBaseline)]:
        model = model_cls()
        model.fit(X_test, y_test)
        proba = model.predict_proba(X_test)[:, 1]
        evaluations.append(full_evaluation(y_test, proba, label=name))

    # Trained models
    for model_type in ["xgboost", "lightgbm", "catboost", "ensemble"]:
        model_path = PROCESSED_DIR / f"model_{model_type}.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            proba = model.predict_proba(X_test)[:, 1]
            evaluations.append(full_evaluation(y_test, proba, label=model_type))

    table = compare_models(evaluations)
    click.echo(table.to_string(index=False))


@main.command()
def status():
    """Show system status and data freshness."""
    click.echo("=== Tennis Predictor Status ===\n")

    # Check data
    matches_file = PROCESSED_DIR / "matches.parquet"
    if matches_file.exists():
        matches = pd.read_parquet(matches_file)
        click.echo(f"Matches loaded: {len(matches):,}")
        click.echo(f"Date range: {matches['tourney_date'].min()} to "
                    f"{matches['tourney_date'].max()}")
    else:
        click.echo("No matches loaded. Run 'tennis-predict ingest' first.")

    # Check features
    features_file = PROCESSED_DIR / "features.parquet"
    if features_file.exists():
        X = pd.read_parquet(features_file)
        click.echo(f"Features built: {X.shape[0]:,} samples, {X.shape[1]} features")
    else:
        click.echo("No features built. Run 'tennis-predict build-features' first.")

    # Check models
    for model_type in ["elo", "xgboost", "lightgbm", "catboost", "ensemble"]:
        model_path = PROCESSED_DIR / f"model_{model_type}.pkl"
        if model_path.exists():
            click.echo(f"Model [{model_type}]: trained ✓")

    # Check Sackmann repo
    if SACKMANN_DIR.exists():
        click.echo(f"\nSackmann repo: present at {SACKMANN_DIR}")
    else:
        click.echo("\nSackmann repo: not cloned")

    # Online learner state
    learner_state = PROCESSED_DIR / "online_state" / "learner_state.json"
    if learner_state.exists():
        state = json.loads(learner_state.read_text())
        click.echo(f"\nOnline learner: {len(state.get('prediction_log', []))} predictions logged")
        click.echo(f"Retrains: {state.get('retrain_count', 0)}")


@main.command()
@click.option("--start-year", default=1991, help="First year of data")
@click.option("--tour-only/--all-levels", default=False, help="Tour-level matches only")
@click.option("--test-year", default=2023, help="Year to start test set")
@click.option("--weather/--no-weather", default=True, help="Fetch weather data")
@click.option("--court-speed/--no-court-speed", default=True, help="Fetch court speed")
@click.option("--odds/--no-odds", default=True, help="Fetch betting odds")
def full_pipeline(start_year, tour_only, test_year, weather, court_speed, odds):
    """Run the complete pipeline: ingest, features, train, evaluate, deploy."""
    from tennis_predictor.pipeline import run_full_pipeline
    run_full_pipeline(
        start_year=start_year,
        tour_level_only=tour_only,
        test_year=test_year,
        fetch_weather=weather,
        fetch_court_speed=court_speed,
        fetch_odds=odds,
    )


if __name__ == "__main__":
    main()
