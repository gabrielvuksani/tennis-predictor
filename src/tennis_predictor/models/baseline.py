"""Baseline models for comparison.

The Elo baseline is the simplest model — it converts Elo ratings directly
to win probabilities using the logistic function. This is what we must beat.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from tennis_predictor.hyperparams import HP


class EloBaseline(BaseEstimator, ClassifierMixin):
    """Predicts match outcome purely from Elo difference.

    Uses the logistic formula: P(p1 wins) = 1 / (1 + 10^((elo2 - elo1) / 400))
    """

    def __init__(self, elo_column: str = "elo_diff", scale: float = HP.elo.logistic_scale):
        self.elo_column = elo_column
        self.scale = scale

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        if hasattr(X, "values"):
            elo_diff = X[self.elo_column].values if self.elo_column in X.columns else X.iloc[:, 0].values
        else:
            elo_diff = X[:, 0]

        elo_diff = np.nan_to_num(elo_diff, nan=0.0)
        prob_p1 = 1.0 / (1.0 + 10 ** (-elo_diff / self.scale))
        return np.column_stack([1 - prob_p1, prob_p1])

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class RankBaseline(BaseEstimator, ClassifierMixin):
    """Predicts the higher-ranked player always wins.

    This is the simplest possible baseline.
    """

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        if hasattr(X, "values"):
            rank_diff = X["rank_diff"].values if "rank_diff" in X.columns else np.zeros(len(X))
        else:
            rank_diff = np.zeros(X.shape[0])

        # Lower rank number = better player, so negative diff = p1 is better
        rank_diff = np.nan_to_num(rank_diff, nan=0.0)
        # Simple logistic conversion
        prob_p1 = 1.0 / (1.0 + np.exp(rank_diff / 50))
        return np.column_stack([1 - prob_p1, prob_p1])

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class OddsBaseline(BaseEstimator, ClassifierMixin):
    """Uses bookmaker implied probabilities as predictions.

    This is the benchmark we're trying to match/beat. Bookmakers at ~72% accuracy
    represent the market's aggregated information.
    """

    def __init__(self, prob_column: str = "implied_prob_p1"):
        self.prob_column = prob_column

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        if hasattr(X, "values") and self.prob_column in X.columns:
            prob_p1 = X[self.prob_column].values
        else:
            prob_p1 = np.full(len(X) if hasattr(X, "__len__") else X.shape[0], 0.5)

        prob_p1 = np.nan_to_num(prob_p1, nan=0.5)
        prob_p1 = np.clip(prob_p1, 0.01, 0.99)
        return np.column_stack([1 - prob_p1, prob_p1])

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
