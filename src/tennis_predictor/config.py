"""Central configuration for the tennis predictor system."""

from pathlib import Path

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

# Elo configuration
ELO_CONFIG = {
    "initial_rating": 1500.0,
    "k_factor_base": 250.0,
    "k_factor_exponent": 0.4,
    "k_factor_offset": 5,
    # Surface-specific Elo uses 60% weight from surface, 40% from overall
    "surface_weight": 0.6,
    # Tournament level multipliers for K-factor
    "level_multipliers": {
        "G": 1.25,   # Grand Slams
        "M": 1.10,   # Masters 1000
        "F": 1.10,   # Tour Finals
        "A": 1.00,   # ATP 500/250
        "C": 0.85,   # Challengers
        "S": 0.70,   # Satellites/ITF
        "D": 0.90,   # Davis Cup
    },
    # Match format multiplier (best-of-5 results are more informative)
    "bo5_multiplier": 1.10,
}

# Glicko-2 configuration
GLICKO2_CONFIG = {
    "initial_rating": 1500.0,
    "initial_rd": 350.0,    # Rating deviation
    "initial_vol": 0.06,    # Volatility
    "tau": 0.5,             # System constant (controls volatility change)
    "epsilon": 0.000001,    # Convergence tolerance
    "rd_decay_per_day": 0.5,  # RD increases when player is inactive
    "max_rd": 350.0,
    "min_rd": 30.0,
}

# Feature engineering
ROLLING_WINDOWS = [5, 10, 20, 50]  # Recent match windows
SURFACE_TYPES = ["Hard", "Clay", "Grass", "Carpet"]
TOURNAMENT_LEVELS = ["G", "M", "F", "A", "C", "S", "D"]

# Model configuration
MODEL_CONFIG = {
    "xgboost": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "early_stopping_rounds": 50,
    },
    "lightgbm": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
    },
    "catboost": {
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "subsample": 0.8,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "verbose": 0,
        "early_stopping_rounds": 50,
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
