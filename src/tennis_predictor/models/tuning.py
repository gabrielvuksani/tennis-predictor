"""Optuna hyperparameter optimization for the prediction models.

Automatically tunes XGBoost, LightGBM, and CatBoost hyperparameters
using Bayesian optimization with temporal cross-validation.

Usage:
    from tennis_predictor.models.tuning import run_optuna_tuning
    best_params = run_optuna_tuning(X_train, y_train, n_trials=100)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tennis_predictor.evaluation.metrics import brier_score


def run_optuna_tuning(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_trials: int = 100,
    model_type: str = "catboost",
) -> dict:
    """Run Optuna hyperparameter optimization.

    Uses temporal train/val split (last 20% as validation).
    Optimizes for Brier score (lower is better).

    Args:
        X_train: Training features.
        y_train: Training targets.
        n_trials: Number of optimization trials.
        model_type: "xgboost", "lightgbm", or "catboost".

    Returns:
        Best hyperparameters dict.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Temporal split: last 20% as validation
    split = int(len(X_train) * 0.8)
    X_tr = X_train.iloc[:split]
    y_tr = y_train[:split]
    X_val = X_train.iloc[split:]
    y_val = y_train[split:]

    def objective(trial):
        if model_type == "xgboost":
            return _xgb_objective(trial, X_tr, y_tr, X_val, y_val)
        elif model_type == "lightgbm":
            return _lgb_objective(trial, X_tr, y_tr, X_val, y_val)
        elif model_type == "catboost":
            return _cat_objective(trial, X_tr, y_tr, X_val, y_val)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    study = optuna.create_study(direction="minimize", study_name=f"{model_type}_tuning")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest {model_type} Brier score: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    return study.best_params


def _xgb_objective(trial, X_tr, y_tr, X_val, y_val) -> float:
    from tennis_predictor.models.gbm import XGBoostPredictor, _prepare_features

    params = {
        "n_estimators": 1000,
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "early_stopping_rounds": 30,
    }

    model = XGBoostPredictor(**params)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    pred = model.predict_proba(X_val)[:, 1]
    return brier_score(y_val, pred)


def _lgb_objective(trial, X_tr, y_tr, X_val, y_val) -> float:
    from tennis_predictor.models.gbm import LightGBMPredictor

    params = {
        "n_estimators": 1000,
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    model = LightGBMPredictor(**params)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    pred = model.predict_proba(X_val)[:, 1]
    return brier_score(y_val, pred)


def _cat_objective(trial, X_tr, y_tr, X_val, y_val) -> float:
    from tennis_predictor.models.gbm import CatBoostPredictor

    params = {
        "iterations": 1000,
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "early_stopping_rounds": 30,
    }

    model = CatBoostPredictor(**params)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    pred = model.predict_proba(X_val)[:, 1]
    return brier_score(y_val, pred)
