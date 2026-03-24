"""Gradient Boosting Models — XGBoost, LightGBM, CatBoost.

These are the workhorses of the prediction system. Research consistently shows
gradient boosting dominates for structured/tabular sports prediction tasks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from tennis_predictor.config import MODEL_CONFIG


class XGBoostPredictor(BaseEstimator, ClassifierMixin):
    """XGBoost classifier with tennis-tuned defaults."""

    def __init__(self, **kwargs):
        self.params = {**MODEL_CONFIG["xgboost"], **kwargs}
        self.model = None

    def fit(self, X, y, eval_set=None, sample_weight=None, **kwargs):
        import xgboost as xgb

        self.classes_ = np.array([0, 1])

        X_clean = _prepare_features(X)
        self.feature_names_ = list(X_clean.columns) if hasattr(X_clean, "columns") else None

        params = {k: v for k, v in self.params.items()
                  if k != "early_stopping_rounds"}
        early_stopping = self.params.get("early_stopping_rounds", 50)

        # Apply monotone constraints for logical relationships
        if "monotone_constraints" not in params and self.feature_names_:
            constraints = _build_monotone_constraints(self.feature_names_)
            if constraints:
                params["monotone_constraints"] = constraints

        self.model = xgb.XGBClassifier(**params)

        fit_params = {}
        if eval_set is not None:
            X_eval, y_eval = eval_set
            X_eval = _prepare_features(X_eval)
            fit_params["eval_set"] = [(X_eval, y_eval)]
            if early_stopping:
                self.model.set_params(early_stopping_rounds=early_stopping)
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight

        self.model.fit(X_clean, y, **fit_params)
        return self

    def predict_proba(self, X):
        X_clean = _prepare_features(X)
        return self.model.predict_proba(X_clean)

    def predict(self, X):
        X_clean = _prepare_features(X)
        return self.model.predict(X_clean)

    @property
    def feature_importances(self) -> dict[str, float]:
        if self.model is None or self.feature_names_ is None:
            return {}
        importances = self.model.feature_importances_
        return dict(sorted(
            zip(self.feature_names_, importances),
            key=lambda x: x[1], reverse=True
        ))


class LightGBMPredictor(BaseEstimator, ClassifierMixin):
    """LightGBM classifier with tennis-tuned defaults."""

    def __init__(self, **kwargs):
        self.params = {**MODEL_CONFIG["lightgbm"], **kwargs}
        self.model = None

    def fit(self, X, y, eval_set=None, sample_weight=None, **kwargs):
        import lightgbm as lgb

        self.classes_ = np.array([0, 1])

        X_clean = _prepare_features(X)
        self.feature_names_ = list(X_clean.columns) if hasattr(X_clean, "columns") else None

        params = {k: v for k, v in self.params.items()}

        self.model = lgb.LGBMClassifier(**params)

        fit_params = {}
        if eval_set is not None:
            X_eval, y_eval = eval_set
            X_eval = _prepare_features(X_eval)
            fit_params["eval_set"] = [(X_eval, y_eval)]
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight

        self.model.fit(X_clean, y, **fit_params)
        return self

    def predict_proba(self, X):
        X_clean = _prepare_features(X)
        return self.model.predict_proba(X_clean)

    def predict(self, X):
        X_clean = _prepare_features(X)
        return self.model.predict(X_clean)

    @property
    def feature_importances(self) -> dict[str, float]:
        if self.model is None or self.feature_names_ is None:
            return {}
        importances = self.model.feature_importances_
        return dict(sorted(
            zip(self.feature_names_, importances),
            key=lambda x: x[1], reverse=True
        ))


class CatBoostPredictor(BaseEstimator, ClassifierMixin):
    """CatBoost classifier with tennis-tuned defaults."""

    def __init__(self, **kwargs):
        self.params = {**MODEL_CONFIG["catboost"], **kwargs}
        self.model = None

    def fit(self, X, y, eval_set=None, sample_weight=None, **kwargs):
        from catboost import CatBoostClassifier

        self.classes_ = np.array([0, 1])

        X_clean = _prepare_features(X)
        self.feature_names_ = list(X_clean.columns) if hasattr(X_clean, "columns") else None

        params = {k: v for k, v in self.params.items()
                  if k != "early_stopping_rounds"}
        early_stopping = self.params.get("early_stopping_rounds", 50)

        self.model = CatBoostClassifier(**params)

        fit_params = {}
        if eval_set is not None:
            X_eval, y_eval = eval_set
            X_eval = _prepare_features(X_eval)
            fit_params["eval_set"] = (X_eval, y_eval)
            if early_stopping:
                self.model.set_params(od_wait=early_stopping)
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight

        self.model.fit(X_clean, y, **fit_params)
        return self

    def predict_proba(self, X):
        X_clean = _prepare_features(X)
        return self.model.predict_proba(X_clean)

    def predict(self, X):
        X_clean = _prepare_features(X)
        return self.model.predict(X_clean)

    @property
    def feature_importances(self) -> dict[str, float]:
        if self.model is None or self.feature_names_ is None:
            return {}
        importances = self.model.feature_importances_
        return dict(sorted(
            zip(self.feature_names_, importances),
            key=lambda x: x[1], reverse=True
        ))


def _build_monotone_constraints(feature_names: list[str]) -> tuple | None:
    """Build monotone constraints for XGBoost.

    Forces logical relationships: higher Elo diff = higher win probability.
    Returns tuple of (1, -1, 0, ...) per feature, or None if not applicable.

    1 = monotonically increasing (higher value → higher P(win))
    -1 = monotonically decreasing
    0 = no constraint
    """
    # Features that should increase win probability when they increase
    increasing = {
        "elo_diff", "surface_elo_diff", "elo_win_prob", "surface_elo_win_prob",
        "glicko2_diff", "glicko2_uncertainty_diff", "serve_elo_diff", "return_elo_diff",
        "h2h_p1_win_pct", "h2h_surface_p1_win_pct",
        "diff_winrate_10", "diff_surface_winrate_10", "diff_dominance_20",
        "diff_avg_1st_serve_won_10", "diff_avg_bp_save_10",
        "diff_ewma_winrate", "diff_winrate_vs_top50",
        "common_opp_winrate_diff", "form_velocity_diff",
        "rank_points_diff", "log_rank_points_diff",
        "odds_implied_p1", "sentiment_diff",
    }

    # Features that should decrease win probability when they increase
    # (e.g., rank_diff: p1_rank - p2_rank; if p1 has higher rank number, worse)
    decreasing = {
        "rank_diff", "log_rank_diff",
    }

    constraints = []
    for name in feature_names:
        if name in increasing:
            constraints.append(1)
        elif name in decreasing:
            constraints.append(-1)
        else:
            constraints.append(0)

    # Only apply if we have some constrained features
    if any(c != 0 for c in constraints):
        return tuple(constraints)
    return None


def _prepare_features(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Clean features for model consumption."""
    if isinstance(X, np.ndarray):
        return pd.DataFrame(X)

    X = X.copy()

    # Replace infinities with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Ensure all columns are numeric
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return X
