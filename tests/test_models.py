"""Tests for prediction models — GBMs and stacking ensemble.

Verifies that models can be instantiated, fit on small synthetic data,
and produce valid probability predictions (between 0 and 1, summing to ~1).
"""

import numpy as np
import pandas as pd
import pytest


def _make_synthetic_data(n_rows: int = 100, n_features: int = 10, seed: int = 42):
    """Create small synthetic binary classification data for model tests."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.randn(n_rows, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = rng.randint(0, 2, size=n_rows)
    return X, y


class TestXGBoostPredictor:
    """Test XGBoost model fit and predict."""

    def test_fit_and_predict(self):
        from tennis_predictor.models.gbm import XGBoostPredictor

        X, y = _make_synthetic_data()
        model = XGBoostPredictor(n_estimators=10, max_depth=3)
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_probabilities_valid_range(self):
        from tennis_predictor.models.gbm import XGBoostPredictor

        X, y = _make_synthetic_data()
        model = XGBoostPredictor(n_estimators=10, max_depth=3)
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert np.all(proba >= 0.0), "Probabilities must be >= 0"
        assert np.all(proba <= 1.0), "Probabilities must be <= 1"

    def test_probabilities_sum_to_one(self):
        from tennis_predictor.models.gbm import XGBoostPredictor

        X, y = _make_synthetic_data()
        model = XGBoostPredictor(n_estimators=10, max_depth=3)
        model.fit(X, y)

        proba = model.predict_proba(X)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_predict_returns_binary(self):
        from tennis_predictor.models.gbm import XGBoostPredictor

        X, y = _make_synthetic_data()
        model = XGBoostPredictor(n_estimators=10, max_depth=3)
        model.fit(X, y)

        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_feature_importances_after_fit(self):
        from tennis_predictor.models.gbm import XGBoostPredictor

        X, y = _make_synthetic_data()
        model = XGBoostPredictor(n_estimators=10, max_depth=3)
        model.fit(X, y)

        importances = model.feature_importances
        assert len(importances) == X.shape[1]
        assert all(np.isreal(v) for v in importances.values())


class TestLightGBMPredictor:
    """Test LightGBM model fit and predict."""

    def test_fit_and_predict(self):
        from tennis_predictor.models.gbm import LightGBMPredictor

        X, y = _make_synthetic_data()
        model = LightGBMPredictor(n_estimators=10, max_depth=3, verbose=-1)
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_probabilities_valid_range(self):
        from tennis_predictor.models.gbm import LightGBMPredictor

        X, y = _make_synthetic_data()
        model = LightGBMPredictor(n_estimators=10, max_depth=3, verbose=-1)
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_probabilities_sum_to_one(self):
        from tennis_predictor.models.gbm import LightGBMPredictor

        X, y = _make_synthetic_data()
        model = LightGBMPredictor(n_estimators=10, max_depth=3, verbose=-1)
        model.fit(X, y)

        proba = model.predict_proba(X)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


class TestCatBoostPredictor:
    """Test CatBoost model fit and predict."""

    def test_fit_and_predict(self):
        from tennis_predictor.models.gbm import CatBoostPredictor

        X, y = _make_synthetic_data()
        model = CatBoostPredictor(iterations=10, depth=3, verbose=0)
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_probabilities_valid_range(self):
        from tennis_predictor.models.gbm import CatBoostPredictor

        X, y = _make_synthetic_data()
        model = CatBoostPredictor(iterations=10, depth=3, verbose=0)
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_probabilities_sum_to_one(self):
        from tennis_predictor.models.gbm import CatBoostPredictor

        X, y = _make_synthetic_data()
        model = CatBoostPredictor(iterations=10, depth=3, verbose=0)
        model.fit(X, y)

        proba = model.predict_proba(X)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


class TestStackingEnsemble:
    """Test the stacking ensemble with small lightweight base models."""

    def _make_ensemble(self):
        """Create a fast ensemble with tiny base models for testing."""
        from tennis_predictor.models.gbm import (
            XGBoostPredictor,
            LightGBMPredictor,
            CatBoostPredictor,
        )
        from tennis_predictor.models.ensemble import StackingEnsemble

        return StackingEnsemble(
            base_models=[
                ("xgboost", XGBoostPredictor(n_estimators=5, max_depth=2)),
                ("lightgbm", LightGBMPredictor(n_estimators=5, max_depth=2, verbose=-1)),
                ("catboost", CatBoostPredictor(iterations=5, depth=2, verbose=0)),
            ],
            n_folds=2,
            calibrate=True,
            passthrough=False,
        )

    def test_fit_and_predict(self):
        X, y = _make_synthetic_data(n_rows=200, n_features=10)
        ensemble = self._make_ensemble()
        ensemble.fit(X, y)

        proba = ensemble.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_probabilities_valid_range(self):
        X, y = _make_synthetic_data(n_rows=200, n_features=10)
        ensemble = self._make_ensemble()
        ensemble.fit(X, y)

        proba = ensemble.predict_proba(X)
        assert np.all(proba >= 0.0), "Ensemble probabilities must be >= 0"
        assert np.all(proba <= 1.0), "Ensemble probabilities must be <= 1"

    def test_probabilities_sum_to_one(self):
        X, y = _make_synthetic_data(n_rows=200, n_features=10)
        ensemble = self._make_ensemble()
        ensemble.fit(X, y)

        proba = ensemble.predict_proba(X)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_predict_returns_binary(self):
        X, y = _make_synthetic_data(n_rows=200, n_features=10)
        ensemble = self._make_ensemble()
        ensemble.fit(X, y)

        preds = ensemble.predict(X)
        assert set(preds).issubset({0, 1})

    def test_ensemble_without_calibration(self):
        from tennis_predictor.models.gbm import (
            XGBoostPredictor,
            LightGBMPredictor,
        )
        from tennis_predictor.models.ensemble import StackingEnsemble

        ensemble = StackingEnsemble(
            base_models=[
                ("xgboost", XGBoostPredictor(n_estimators=5, max_depth=2)),
                ("lightgbm", LightGBMPredictor(n_estimators=5, max_depth=2, verbose=-1)),
            ],
            n_folds=2,
            calibrate=False,
            passthrough=False,
        )
        X, y = _make_synthetic_data(n_rows=200, n_features=10)
        ensemble.fit(X, y)

        proba = ensemble.predict_proba(X)
        assert proba.shape == (len(X), 2)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


class TestBaselineModels:
    """Test baseline models for basic correctness."""

    def test_elo_baseline_higher_elo_wins(self):
        from tennis_predictor.models.baseline import EloBaseline

        X = pd.DataFrame({"elo_diff": [200.0, -200.0, 0.0]})
        model = EloBaseline()
        model.fit(X)
        proba = model.predict_proba(X)

        # Positive elo_diff => p1 favored => proba[:,1] > 0.5
        assert proba[0, 1] > 0.5
        # Negative elo_diff => p1 underdog => proba[:,1] < 0.5
        assert proba[1, 1] < 0.5
        # Zero elo_diff => coin flip
        assert abs(proba[2, 1] - 0.5) < 1e-6

    def test_rank_baseline_probabilities_valid(self):
        from tennis_predictor.models.baseline import RankBaseline

        X = pd.DataFrame({"rank_diff": [10.0, -10.0, 0.0]})
        model = RankBaseline()
        model.fit(X)
        proba = model.predict_proba(X)

        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)
