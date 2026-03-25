"""Feature selection utilities — importance-based feature ranking and pruning.

Uses native GBM feature importances (gain-based) to rank features. This is
faster and more stable than SHAP for large feature sets, and correlates well
with SHAP for tree-based models.

Not integrated into the main pipeline — call from CLI or notebook:
    from tennis_predictor.features.selection import auto_select_features
    selected, importances = auto_select_features(X_train, y_train, top_n=60)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def select_features(
    model: Any,
    X: pd.DataFrame,
    y: Any = None,
    top_n: int = 60,
    min_importance: float = 0.001,
) -> tuple[list[str], dict[str, float]]:
    """Select top N features by model importance.

    Works with any model that exposes feature importances via either:
    - A `feature_importances` property returning dict[str, float]
      (our GBM wrappers: XGBoostPredictor, LightGBMPredictor, CatBoostPredictor)
    - A `feature_importances_` attribute returning an array (sklearn convention)

    Args:
        model: A fitted model with feature importances.
        X: Feature matrix (DataFrame with named columns).
        y: Ignored — present for API consistency.
        top_n: Maximum number of features to keep.
        min_importance: Minimum importance threshold; features below this are dropped
            even if top_n has not been reached.

    Returns:
        Tuple of (selected_feature_names, importance_dict) where importance_dict
        maps every feature name to its importance score, sorted descending.
    """
    # Extract importances — support both our wrapper and sklearn interfaces
    if hasattr(model, "feature_importances") and callable(
        getattr(type(model), "feature_importances", None).__get__  # property check
    ):
        importance_dict = dict(model.feature_importances)
    elif hasattr(model, "feature_importances_"):
        raw = model.feature_importances_
        if isinstance(X, pd.DataFrame):
            names = list(X.columns)
        else:
            names = [f"feat_{i}" for i in range(len(raw))]
        importance_dict = dict(zip(names, raw))
    else:
        raise ValueError(
            f"Model {type(model).__name__} does not expose feature importances. "
            "Expected a `feature_importances` property or `feature_importances_` attribute."
        )

    # Sort descending by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Apply both top_n and min_importance filters
    selected = []
    for name, score in sorted_features:
        if len(selected) >= top_n:
            break
        if score >= min_importance:
            selected.append(name)

    dropped = [name for name, _ in sorted_features if name not in selected]

    # Print summary
    print(f"\n{'='*60}")
    print(f"Feature Selection Summary")
    print(f"{'='*60}")
    print(f"Total features:    {len(importance_dict)}")
    print(f"Selected features: {len(selected)} (top_n={top_n}, min_importance={min_importance})")
    print(f"Dropped features:  {len(dropped)}")
    print(f"\nTop 20 selected features:")
    for i, name in enumerate(selected[:20]):
        score = importance_dict[name]
        print(f"  {i+1:3d}. {name:<40s} {score:.6f}")
    if len(selected) > 20:
        print(f"  ... and {len(selected) - 20} more")
    if dropped:
        print(f"\nBottom 10 dropped features:")
        for name in dropped[-10:]:
            score = importance_dict[name]
            print(f"       {name:<40s} {score:.6f}")
    print(f"{'='*60}\n")

    return selected, dict(sorted_features)


def auto_select_features(
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    top_n: int = 60,
    min_importance: float = 0.001,
    iterations: int = 100,
) -> tuple[list[str], dict[str, float]]:
    """Train a quick CatBoost model and select top features.

    This is a convenience function for rapid feature selection. It trains
    CatBoost with few iterations (just enough to estimate importances)
    and returns the top N features ranked by gain.

    Args:
        X: Feature matrix.
        y: Binary target.
        top_n: Maximum number of features to keep.
        min_importance: Minimum importance threshold.
        iterations: CatBoost iterations for importance estimation (default 100).

    Returns:
        Tuple of (selected_feature_names, importance_dict).
    """
    from tennis_predictor.models.gbm import CatBoostPredictor

    print(f"Training CatBoost ({iterations} iterations) for feature importance estimation...")
    model = CatBoostPredictor(
        iterations=iterations,
        depth=6,
        learning_rate=0.1,
        verbose=0,
    )
    model.fit(X, y if isinstance(y, np.ndarray) else np.asarray(y))

    return select_features(model, X, top_n=top_n, min_importance=min_importance)
