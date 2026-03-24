"""Stacking ensemble and calibration.

The ensemble combines multiple base models (XGBoost, LightGBM, CatBoost)
via a meta-learner. This consistently outperforms any single model.

Calibration is applied as the final layer — we optimize for Brier score,
not accuracy, because calibrated probabilities are what matter for betting.
A study showed calibration-optimized models achieved +34.69% ROI while
accuracy-optimized models achieved -35.17% ROI.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from tennis_predictor.models.gbm import (
    CatBoostPredictor,
    LightGBMPredictor,
    XGBoostPredictor,
)


class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """Stacking ensemble that combines gradient boosting models.

    Architecture:
    1. Base models (XGBoost, LightGBM, CatBoost) produce probability predictions
    2. Meta-learner (Logistic Regression) combines base predictions
    3. Isotonic regression calibrates final probabilities

    The meta-learner is trained on out-of-fold predictions to avoid overfitting.
    """

    # Key features to pass through to meta-learner for context-aware stacking
    PASSTHROUGH_FEATURES = [
        "elo_diff", "surface_elo_diff", "rank_diff", "surface_Hard",
        "surface_Clay", "surface_Grass", "tourney_level_G", "best_of_5",
    ]

    def __init__(
        self,
        base_models: list[tuple[str, BaseEstimator]] | None = None,
        meta_learner: BaseEstimator | None = None,
        n_folds: int = 5,
        calibrate: bool = True,
        passthrough: bool = True,
    ):
        self.base_models = base_models or [
            ("xgboost", XGBoostPredictor()),
            ("lightgbm", LightGBMPredictor()),
            ("catboost", CatBoostPredictor()),
        ]
        self.meta_learner = meta_learner or LogisticRegression(
            C=1.0, max_iter=5000, solver="lbfgs"
        )
        self.passthrough = passthrough
        self.n_folds = n_folds
        self.calibrate = calibrate
        self.calibrator_ = None
        self.fitted_base_models_: list[tuple[str, list[BaseEstimator]]] = []

    def fit(self, X, y, **kwargs):
        self.classes_ = np.array([0, 1])

        n_samples = len(y)
        n_models = len(self.base_models)

        # Generate out-of-fold predictions for meta-learner training
        # Use temporal folds (no shuffling) to prevent leakage
        oof_predictions = np.zeros((n_samples, n_models))
        from sklearn.model_selection import TimeSeriesSplit
        kf = TimeSeriesSplit(n_splits=self.n_folds)

        self.fitted_base_models_ = []

        for model_idx, (name, model_template) in enumerate(self.base_models):
            fold_models = []
            print(f"  Training base model: {name}")

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                if isinstance(X, pd.DataFrame):
                    X_fold_train = X.iloc[train_idx]
                    X_fold_val = X.iloc[val_idx]
                else:
                    X_fold_train = X[train_idx]
                    X_fold_val = X[val_idx]
                y_fold_train = y[train_idx]

                # Clone model for this fold
                import sklearn.base
                model = sklearn.base.clone(model_template)

                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=(X_fold_val, y[val_idx]),
                )

                # Out-of-fold predictions
                proba = model.predict_proba(X_fold_val)
                oof_predictions[val_idx, model_idx] = proba[:, 1]
                fold_models.append(model)

            self.fitted_base_models_.append((name, fold_models))

        # Train meta-learner on out-of-fold predictions + passthrough features
        print("  Training meta-learner...")
        meta_input = oof_predictions
        if self.passthrough and isinstance(X, pd.DataFrame):
            pt_cols = [c for c in self.PASSTHROUGH_FEATURES if c in X.columns]
            if pt_cols:
                pt_data = X[pt_cols].fillna(0).values
                meta_input = np.hstack([oof_predictions, pt_data])
        self.meta_learner.fit(meta_input, y)

        # Calibrate (must use same meta_input shape as training)
        if self.calibrate:
            meta_proba = self.meta_learner.predict_proba(meta_input)[:, 1]
            self.calibrator_ = IsotonicRegression(
                y_min=0.01, y_max=0.99, out_of_bounds="clip"
            )
            self.calibrator_.fit(meta_proba, y)

        # Store passthrough column names for predict
        self._pt_cols = (
            [c for c in self.PASSTHROUGH_FEATURES if c in X.columns]
            if self.passthrough and isinstance(X, pd.DataFrame) else []
        )

        # Retrain all base models on full dataset for final predictions
        self.final_base_models_ = []
        for name, model_template in self.base_models:
            import sklearn.base
            model = sklearn.base.clone(model_template)
            model.fit(X, y)
            self.final_base_models_.append((name, model))

        return self

    def predict_proba(self, X):
        # Get base model predictions
        base_preds = np.zeros((len(X) if hasattr(X, "__len__") else X.shape[0],
                               len(self.final_base_models_)))

        for idx, (name, model) in enumerate(self.final_base_models_):
            proba = model.predict_proba(X)
            base_preds[:, idx] = proba[:, 1]

        # Meta-learner combines predictions + passthrough features
        meta_input = base_preds
        pt_cols = getattr(self, "_pt_cols", [])
        if pt_cols and isinstance(X, pd.DataFrame):
            pt_data = X[pt_cols].fillna(0).values
            meta_input = np.hstack([base_preds, pt_data])
        meta_proba = self.meta_learner.predict_proba(meta_input)[:, 1]

        # Calibrate
        if self.calibrate and self.calibrator_ is not None:
            meta_proba = self.calibrator_.predict(meta_proba)

        meta_proba = np.clip(meta_proba, 0.01, 0.99)
        return np.column_stack([1 - meta_proba, meta_proba])

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    @property
    def base_model_weights(self) -> dict[str, float]:
        """Meta-learner weights for each base model."""
        if hasattr(self.meta_learner, "coef_"):
            weights = self.meta_learner.coef_[0]
            return {name: float(w) for (name, _), w
                    in zip(self.base_models, weights)}
        return {}


class CalibratedModel(BaseEstimator, ClassifierMixin):
    """Wrapper that adds calibration to any model.

    Supports both Platt scaling (sigmoid) and isotonic regression.
    Isotonic is preferred for large datasets (>1000 samples).
    """

    def __init__(
        self,
        base_model: BaseEstimator,
        method: str = "isotonic",
        cv: int = 5,
    ):
        self.base_model = base_model
        self.method = method
        self.cv = cv
        self.calibrated_model_ = None

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.calibrated_model_ = CalibratedClassifierCV(
            estimator=self.base_model,
            method=self.method,
            cv=self.cv,
        )
        self.calibrated_model_.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.calibrated_model_.predict_proba(X)

    def predict(self, X):
        return self.calibrated_model_.predict(X)


def create_default_ensemble(**kwargs) -> StackingEnsemble:
    """Create the default stacking ensemble with tuned hyperparameters."""
    return StackingEnsemble(
        base_models=[
            ("xgboost", XGBoostPredictor(**kwargs.get("xgboost", {}))),
            ("lightgbm", LightGBMPredictor(**kwargs.get("lightgbm", {}))),
            ("catboost", CatBoostPredictor(**kwargs.get("catboost", {}))),
        ],
        calibrate=True,
    )
