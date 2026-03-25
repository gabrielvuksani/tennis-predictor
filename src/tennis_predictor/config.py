"""Central configuration for the tennis predictor system.

Note: ELO_CONFIG, GLICKO2_CONFIG, and MODEL_CONFIG are compatibility layers
that delegate to the HP singleton in hyperparams.py. HP is the single source
of truth for all tunable parameters.
"""

from pathlib import Path
from tennis_predictor.hyperparams import HP

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
CACHE_DIR = DATA_DIR / "cache"
SITE_DIR = ROOT_DIR / "site"

# Data source paths
SACKMANN_DIR = RAW_DIR / "tennis_atp"
ODDS_DIR = RAW_DIR / "odds"
WEATHER_DIR = RAW_DIR / "weather"
COURT_SPEED_DIR = RAW_DIR / "court_speed"

# Elo configuration — delegates to HP.elo
ELO_CONFIG = {
    "initial_rating": HP.elo.initial_rating,
    "k_factor_base": HP.elo.k_factor_base,
    "k_factor_exponent": HP.elo.k_factor_exponent,
    "k_factor_offset": HP.elo.k_factor_offset,
    "surface_weight": HP.elo.surface_weight,
    "level_multipliers": HP.elo.level_multipliers,
    "bo5_multiplier": HP.elo.bo5_multiplier,
}

# Glicko-2 configuration — delegates to HP.glicko2
GLICKO2_CONFIG = {
    "initial_rating": HP.glicko2.initial_rating,
    "initial_rd": HP.glicko2.initial_rd,
    "initial_vol": HP.glicko2.initial_vol,
    "tau": HP.glicko2.tau,
    "epsilon": HP.glicko2.epsilon,
    "rd_decay_per_day": HP.glicko2.rd_decay_per_day,
    "max_rd": HP.glicko2.max_rd,
    "min_rd": HP.glicko2.min_rd,
}

# Feature engineering
ROLLING_WINDOWS = [5, 10, 20, 50]  # Recent match windows
SURFACE_TYPES = ["Hard", "Clay", "Grass", "Carpet"]
TOURNAMENT_LEVELS = ["G", "M", "F", "A", "C", "S", "D"]

# Model configuration — delegates to HP.model
MODEL_CONFIG = {
    "xgboost": {
        "n_estimators": HP.model.xgb_n_estimators,
        "max_depth": HP.model.xgb_max_depth,
        "learning_rate": HP.model.xgb_learning_rate,
        "subsample": HP.model.xgb_subsample,
        "colsample_bytree": HP.model.xgb_colsample_bytree,
        "min_child_weight": HP.model.xgb_min_child_weight,
        "reg_alpha": HP.model.xgb_reg_alpha,
        "reg_lambda": HP.model.xgb_reg_lambda,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "early_stopping_rounds": HP.model.xgb_early_stopping,
    },
    "lightgbm": {
        "n_estimators": HP.model.lgb_n_estimators,
        "max_depth": HP.model.lgb_max_depth,
        "learning_rate": HP.model.lgb_learning_rate,
        "subsample": HP.model.lgb_subsample,
        "colsample_bytree": HP.model.lgb_colsample_bytree,
        "min_child_samples": HP.model.lgb_min_child_samples,
        "reg_alpha": HP.model.lgb_reg_alpha,
        "reg_lambda": HP.model.lgb_reg_lambda,
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
    },
    "catboost": {
        "iterations": HP.model.cat_iterations,
        "depth": HP.model.cat_depth,
        "learning_rate": HP.model.cat_learning_rate,
        "l2_leaf_reg": HP.model.cat_l2_leaf_reg,
        "subsample": HP.model.cat_subsample,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "verbose": 0,
        "early_stopping_rounds": HP.model.cat_early_stopping,
    },
}

# Temporal validation
VALIDATION_CONFIG = {
    "min_training_years": 5,       # Minimum years of data to start training
    "validation_window_months": 6,  # Each validation fold
    "expanding_window": True,       # Expanding (True) vs rolling (False)
    "gap_days": 1,                  # Gap between train and test to prevent leakage
}

# Online learning
ONLINE_CONFIG = {
    "retrain_frequency_days": 30,
    "drift_detection_method": "adwin",
    "drift_confidence": 0.002,
    "min_samples_for_drift": 100,
}

# Weather API
WEATHER_CONFIG = {
    "api_base": "https://archive-api.open-meteo.com/v1/archive",
    "forecast_base": "https://api.open-meteo.com/v1/forecast",
    "geocoding_base": "https://geocoding-api.open-meteo.com/v1/search",
    "variables": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "windspeed_10m_max",
        "windgusts_10m_max",
    ],
    "hourly_variables": [
        "relativehumidity_2m",
        "temperature_2m",
        "windspeed_10m",
    ],
}
