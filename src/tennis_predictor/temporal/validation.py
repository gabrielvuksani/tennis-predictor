"""Temporal validation framework.

Implements expanding-window and rolling-window temporal cross-validation.
NEVER uses random splits — all splits respect chronological order.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from tennis_predictor.config import VALIDATION_CONFIG
from tennis_predictor.temporal.guard import TemporalGuard, TemporalState


@dataclass
class TemporalFold:
    """A single temporal validation fold."""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_indices: np.ndarray
    test_indices: np.ndarray

    @property
    def train_size(self) -> int:
        return len(self.train_indices)

    @property
    def test_size(self) -> int:
        return len(self.test_indices)


def generate_temporal_folds(
    matches: pd.DataFrame,
    min_training_years: int | None = None,
    validation_window_months: int | None = None,
    expanding: bool | None = None,
    gap_days: int | None = None,
) -> list[TemporalFold]:
    """Generate temporal cross-validation folds.

    Each fold has:
    - Training set: all matches before the test period (expanding) or a fixed window (rolling)
    - Gap: a small gap to prevent same-tournament leakage
    - Test set: matches in the validation window

    Args:
        matches: DataFrame with 'tourney_date' column, sorted chronologically.
        min_training_years: Minimum years of data before first test fold.
        validation_window_months: Duration of each test fold in months.
        expanding: If True, training set grows. If False, fixed rolling window.
        gap_days: Days of gap between train and test.
    """
    config = VALIDATION_CONFIG
    min_training_years = min_training_years or config["min_training_years"]
    validation_window_months = validation_window_months or config["validation_window_months"]
    expanding = expanding if expanding is not None else config["expanding_window"]
    gap_days = gap_days or config["gap_days"]

    dates = matches["tourney_date"]
    min_date = dates.min()
    max_date = dates.max()

    # First test period starts after min_training_years
    first_test_start = min_date + pd.DateOffset(years=min_training_years)
    if first_test_start >= max_date:
        raise ValueError(
            f"Not enough data for validation. Data spans {min_date} to {max_date}, "
            f"but min_training_years={min_training_years} pushes first test past data end."
        )

    folds = []
    fold_id = 0
    test_start = first_test_start

    while test_start < max_date:
        test_end = test_start + pd.DateOffset(months=validation_window_months)
        if test_end > max_date:
            test_end = max_date

        # Training period
        train_end = test_start - timedelta(days=gap_days)
        if expanding:
            train_start = min_date
        else:
            # Rolling window: same size as would be at first fold
            rolling_window = first_test_start - min_date
            train_start = max(min_date, train_end - rolling_window)

        # Get indices
        train_mask = (dates >= train_start) & (dates <= train_end)
        test_mask = (dates >= test_start) & (dates < test_end)

        train_idx = matches.index[train_mask].values
        test_idx = matches.index[test_mask].values

        if len(train_idx) > 0 and len(test_idx) > 0:
            folds.append(TemporalFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_indices=train_idx,
                test_indices=test_idx,
            ))
            fold_id += 1

        test_start = test_end

    return folds


def build_features_chronologically(
    matches: pd.DataFrame,
    guard: TemporalGuard | None = None,
    progress: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, TemporalGuard]:
    """Process all matches chronologically, extracting features and updating state.

    This is the ONLY correct way to build the feature matrix. Each match's
    features are extracted from pre-match state, then the state is updated.

    Args:
        matches: Pairwise match DataFrame sorted by date (from create_pairwise_rows).
        guard: Existing TemporalGuard to continue from (or None for fresh start).
        progress: Show progress bar.

    Returns:
        Tuple of (feature_matrix, target_array, guard_with_final_state).
    """
    if guard is None:
        guard = TemporalGuard()

    all_features = []
    targets = []

    iterator = matches.iterrows()
    if progress:
        iterator = tqdm(iterator, total=len(matches), desc="Building features")

    for idx, match in iterator:
        # 1. Extract features BEFORE updating state (temporal safety)
        features = guard.extract_pre_match_state(match)

        # 2. Record features and target
        all_features.append(features)
        targets.append(match["y"])

        # 3. NOW update state with match result
        guard.update_state(match, match["y"])

    feature_df = pd.DataFrame(all_features, index=matches.index)
    target_arr = np.array(targets, dtype=np.float32)

    return feature_df, target_arr, guard


def temporal_backtest(
    matches: pd.DataFrame,
    model_factory,
    folds: list[TemporalFold] | None = None,
    progress: bool = True,
) -> list[dict]:
    """Run a full temporal backtest.

    For each fold:
    1. Build features chronologically for training data
    2. Train model
    3. Build features for test data (continuing from training state)
    4. Predict and evaluate

    Args:
        matches: Pairwise match DataFrame sorted by date.
        model_factory: Callable that returns a fresh model instance.
        folds: Pre-computed folds (or None to auto-generate).
        progress: Show progress bar.

    Returns:
        List of fold result dicts with predictions and metrics.
    """
    if folds is None:
        folds = generate_temporal_folds(matches)

    results = []

    for fold in tqdm(folds, desc="Backtesting folds", disable=not progress):
        # Build features for training period
        train_matches = matches.loc[fold.train_indices]
        X_train, y_train, guard = build_features_chronologically(
            train_matches, progress=False
        )

        # Build features for test period (continuing from training state)
        test_matches = matches.loc[fold.test_indices]
        X_test, y_test, guard = build_features_chronologically(
            test_matches, guard=guard, progress=False
        )

        # Train model
        model = model_factory()
        model.fit(X_train, y_train)

        # Predict
        y_pred_proba = model.predict_proba(X_test)
        if y_pred_proba.ndim > 1:
            y_pred_proba = y_pred_proba[:, 1]

        results.append({
            "fold_id": fold.fold_id,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            "train_size": fold.train_size,
            "test_size": fold.test_size,
            "y_true": y_test,
            "y_pred_proba": y_pred_proba,
            "test_indices": fold.test_indices,
        })

    return results
