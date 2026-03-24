"""Dynamic hyperparameter system — replaces ALL hardcoded magic numbers.

Every tunable parameter lives here. They can be:
1. Set to defaults (current values, for backward compatibility)
2. Loaded from a YAML config file (for manual tuning)
3. Optimized by Optuna (for automated tuning)
4. Learned from data (for parameters like peak_age)

Usage:
    from tennis_predictor.hyperparams import HP
    k_factor = HP.elo.k_factor_base  # 250.0 by default
    HP.load("config/tuned_params.yaml")  # Override with tuned values
"""

from __future__ import annotations

import json
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class EloParams:
    initial_rating: float = 1500.0
    k_factor_base: float = 250.0
    k_factor_exponent: float = 0.4
    k_factor_offset: int = 5
    surface_weight: float = 0.6        # 60% surface Elo, 40% overall
    surface_k_multiplier: float = 1.2   # Higher K for surface (less data)
    serve_k_multiplier: float = 0.8     # Dampened K for serve/return Elo
    bo5_multiplier: float = 1.10
    logistic_scale: float = 400.0       # Standard Elo scale
    # Tournament level multipliers
    level_G: float = 1.25   # Grand Slam
    level_M: float = 1.10   # Masters 1000
    level_F: float = 1.10   # Tour Finals
    level_A: float = 1.00   # ATP 500/250
    level_C: float = 0.85   # Challengers
    level_S: float = 0.70   # Satellites/ITF
    level_D: float = 0.90   # Davis Cup

    @property
    def level_multipliers(self) -> dict[str, float]:
        return {
            "G": self.level_G, "M": self.level_M, "F": self.level_F,
            "A": self.level_A, "C": self.level_C, "S": self.level_S,
            "D": self.level_D,
        }


@dataclass
class Glicko2Params:
    initial_rating: float = 1500.0
    initial_rd: float = 350.0
    initial_vol: float = 0.06
    tau: float = 0.5
    epsilon: float = 1e-6
    rd_decay_per_day: float = 0.5
    max_rd: float = 350.0
    min_rd: float = 30.0
    vol_increase_factor: float = 1.01
    max_vol: float = 0.1


@dataclass
class FeatureParams:
    rolling_windows: list[int] = field(default_factory=lambda: [5, 10, 20, 50])
    match_history_limit: int = 200
    inactivity_days: int = 60
    fatigue_windows_days: list[int] = field(default_factory=lambda: [7, 14, 30])
    ewma_alpha: float = 0.15
    ewma_window: int = 50
    min_matches_ewma: int = 3
    min_matches_variance: int = 5
    min_matches_velocity: int = 10
    velocity_window: int = 5
    form_gradient_window: int = 10
    surface_adaptation_window: int = 5
    peak_age: float = 25.5
    post_peak_age: float = 28.0
    top_opponent_rank: int = 50
    acwr_acute_days: int = 7
    acwr_chronic_days: int = 28
    serve_return_window: int = 20


@dataclass
class ModelParams:
    # XGBoost
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_min_child_weight: int = 5
    xgb_reg_alpha: float = 0.1
    xgb_reg_lambda: float = 1.0
    xgb_early_stopping: int = 50
    # LightGBM
    lgb_n_estimators: int = 500
    lgb_max_depth: int = 6
    lgb_learning_rate: float = 0.05
    lgb_subsample: float = 0.8
    lgb_colsample_bytree: float = 0.8
    lgb_min_child_samples: int = 20
    lgb_reg_alpha: float = 0.1
    lgb_reg_lambda: float = 1.0
    # CatBoost
    cat_iterations: int = 500
    cat_depth: int = 6
    cat_learning_rate: float = 0.05
    cat_l2_leaf_reg: float = 3.0
    cat_subsample: float = 0.8
    cat_early_stopping: int = 50
    # Ensemble
    stacking_folds: int = 5
    meta_learner_C: float = 1.0
    meta_learner_max_iter: int = 5000
    prob_clip_min: float = 0.01
    prob_clip_max: float = 0.99


@dataclass
class OnlineParams:
    retrain_frequency_days: int = 30
    drift_confidence: float = 0.002
    drift_min_window: int = 30
    drift_max_window: int = 1000
    recency_weight_range: tuple[float, float] = (-2.0, 0.0)
    performance_trend_window: int = 100
    trend_threshold: float = 0.01


@dataclass
class BettingParams:
    min_edge: float = 0.05
    kelly_fraction: float = 0.25
    max_bet_multiplier: float = 5.0
    upset_detection_lower: float = 0.4
    upset_detection_upper: float = 0.6


@dataclass
class PipelineParams:
    start_year: int = 1991
    test_year: int = 2023
    min_weather_year: int = 2015
    intransitivity_lookback_days: int = 730
    min_common_opponents: int = 2
    time_decay_gamma: float = 0.9997
    odds_date_window: int = 16
    odds_rank_tolerance: int = 1


@dataclass
class PredictionParams:
    surface_blend_weight: float = 0.6
    unknown_player_shrinkage: float = 0.6
    rank_logistic_scale: float = 250.0


@dataclass
class Hyperparams:
    """Central hyperparameter store. All tunable values live here."""
    elo: EloParams = field(default_factory=EloParams)
    glicko2: Glicko2Params = field(default_factory=Glicko2Params)
    features: FeatureParams = field(default_factory=FeatureParams)
    model: ModelParams = field(default_factory=ModelParams)
    online: OnlineParams = field(default_factory=OnlineParams)
    betting: BettingParams = field(default_factory=BettingParams)
    pipeline: PipelineParams = field(default_factory=PipelineParams)
    prediction: PredictionParams = field(default_factory=PredictionParams)

    def load(self, path: str | Path) -> None:
        """Load hyperparameters from YAML file, overriding defaults."""
        path = Path(path)
        if not path.exists():
            return
        with open(path) as f:
            data = yaml.safe_load(f)
        if not data:
            return
        self._apply(data)

    def save(self, path: str | Path) -> None:
        """Save current hyperparameters to YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def _apply(self, data: dict) -> None:
        """Recursively apply dict values to dataclass fields."""
        for section_name, section_data in data.items():
            if hasattr(self, section_name) and isinstance(section_data, dict):
                section = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)

    def to_dict(self) -> dict:
        return asdict(self)


# Global singleton — import this everywhere
HP = Hyperparams()

# Load user overrides if config file exists
_config_path = Path(__file__).parent.parent.parent / "config" / "hyperparams.yaml"
if _config_path.exists():
    HP.load(_config_path)
