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
from lightgbm import LGBMClassifier

from tennis_predictor.models.gbm import (
    CatBoostPredictor,
    LightGBMPredictor,
    XGBoostPredictor,
)


class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """Stacking ensemble that combines gradient boosting models.

    Architecture:
    1. Base models (XGBoost, LightGBM, CatBoost) produce probability predictions
    2. Meta-learner combines base predictions (LightGBM or Logistic Regression)
    3. Isotonic regression calibrates final probabilities

    The meta-learner is trained on out-of-fold predictions to avoid overfitting.

    When ``meta_learner_type='lgbm'`` (default), a LightGBM classifier is used
    so the meta-learner can capture nonlinear interactions between base model
    predictions and passthrough context features (e.g. "trust XGBoost more on
    clay").  When ``meta_learner_type='logistic'``, the original Logistic
    Regression meta-learner is used instead.
    """

    # Key features to pass through to meta-learner for context-aware stacking
    PASSTHROUGH_FEATURES = [
        "elo_diff", "surface_elo_diff", "rank_diff",
        "surface_Hard", "surface_Clay", "surface_Grass",
        "tourney_level_G", "best_of_5",
        # Uncertainty signals
        "glicko2_rd_p1", "glicko2_rd_p2",
        # Market signal (when available)
        "odds_implied_p1",
        # Player history depth
        "p1_match_count", "p2_match_count",
        # H2H context
        "h2h_total_matches",
    ]

    def __init__(
        self,
        base_models: list[tuple[str, BaseEstimator]] | None = None,
        meta_learner: BaseEstimator | None = None,
        meta_learner_type: str = "lgbm",
        n_folds: int = 5,
        calibrate: bool = True,
        passthrough: bool = True,
    ):
        self.base_models = base_models or [
            ("xgboost", XGBoostPredictor()),
            ("lightgbm", LightGBMPredictor()),
            ("catboost", CatBoostPredictor()),
        ]
        self.meta_learner_type = meta_learner_type
        if meta_learner is not None:
            # Explicit meta_learner takes precedence
            self.meta_learner = meta_learner
        elif meta_learner_type == "lgbm":
            self.meta_learner = LGBMClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                reg_lambda=1.0,
                subsample=0.7,
                colsample_bytree=0.7,
                min_child_samples=50,
                verbose=-1,
            )
        else:
            # 'logistic' or any other value falls back to LogisticRegression
            self.meta_learner = LogisticRegression(
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

        # Build meta-features: out-of-fold predictions + passthrough features
        base_pred_columns = [name for name, _ in self.base_models]
        meta_input = pd.DataFrame(oof_predictions, columns=base_pred_columns)

        # Base model disagreement — tells the meta-learner when models disagree
        meta_input['base_disagreement'] = meta_input[base_pred_columns].std(axis=1)

        if self.passthrough and isinstance(X, pd.DataFrame):
            pt_cols = [c for c in self.PASSTHROUGH_FEATURES if c in X.columns]
            if pt_cols:
                pt_data = X[pt_cols].fillna(0).reset_index(drop=True)
                meta_input = pd.concat([meta_input, pt_data], axis=1)

        meta_input = meta_input.values

        # Split meta-features temporally: first 80% trains meta-learner,
        # last 20% is held out for calibration. This prevents the calibrator
        # from overfitting to data the meta-learner was optimized on.
        if self.calibrate:
            split_idx = int(len(y) * 0.8)
            meta_train, meta_cal = meta_input[:split_idx], meta_input[split_idx:]
            y_train, y_cal = y[:split_idx], y[split_idx:]

            print("  Training meta-learner...")
            self.meta_learner.fit(meta_train, y_train)

            print("  Fitting calibrator on held-out meta-predictions...")
            cal_proba = self.meta_learner.predict_proba(meta_cal)[:, 1]
            self.calibrator_ = IsotonicRegression(
                y_min=0.01, y_max=0.99, out_of_bounds="clip"
            )
            self.calibrator_.fit(cal_proba, y_cal)
        else:
            print("  Training meta-learner...")
            self.meta_learner.fit(meta_input, y)

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

        # Meta-learner combines predictions + passthrough features + disagreement
        base_pred_columns = [name for name, _ in self.final_base_models_]
        meta_df = pd.DataFrame(base_preds, columns=base_pred_columns)
        meta_df['base_disagreement'] = meta_df[base_pred_columns].std(axis=1)

        pt_cols = getattr(self, "_pt_cols", [])
        if pt_cols and isinstance(X, pd.DataFrame):
            pt_data = X[pt_cols].fillna(0).reset_index(drop=True)
            meta_df = pd.concat([meta_df, pt_data], axis=1)

        meta_input = meta_df.values
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
        """Meta-learner weights for each base model.

        For Logistic Regression, returns the linear coefficients.
        For LightGBM, returns feature importances for base model columns
        (normalized to sum to 1), which approximate how much each base
        model contributes to the final prediction.
        """
        if hasattr(self.meta_learner, "coef_"):
            weights = self.meta_learner.coef_[0]
            return {name: float(w) for (name, _), w
                    in zip(self.base_models, weights)}
        if hasattr(self.meta_learner, "feature_importances_"):
            importances = self.meta_learner.feature_importances_
            n_base = len(self.base_models)
            base_importances = importances[:n_base]
            total = base_importances.sum()
            if total > 0:
                base_importances = base_importances / total
            return {name: float(w) for (name, _), w
                    in zip(self.base_models, base_importances)}
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
    """Create the default stacking ensemble with tuned hyperparameters.

    Uses a LightGBM meta-learner by default to capture nonlinear interactions
    between base model predictions and passthrough context features.
    Pass ``meta_learner_type='logistic'`` to revert to LogisticRegression.
    """
    return StackingEnsemble(
        base_models=[
            ("xgboost", XGBoostPredictor(**kwargs.get("xgboost", {}))),
            ("lightgbm", LightGBMPredictor(**kwargs.get("lightgbm", {}))),
            ("catboost", CatBoostPredictor(**kwargs.get("catboost", {}))),
        ],
        meta_learner_type=kwargs.get("meta_learner_type", "lgbm"),
        calibrate=True,
    )
