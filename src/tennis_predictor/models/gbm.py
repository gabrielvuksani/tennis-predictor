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
