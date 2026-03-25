"""Tests for evaluation metrics.

Verifies Brier score, accuracy, calibration curve, and ECE against known
inputs with deterministic expected outputs.
"""

import numpy as np
import pytest

from tennis_predictor.evaluation.metrics import (
    accuracy,
    brier_score,
    calibration_curve,
    expected_calibration_error,
    log_loss,
)


class TestBrierScore:
    """Test Brier score on known inputs."""

    def test_perfect_predictions(self):
        """Perfect predictions should yield Brier score = 0."""
        y_true = np.array([1, 0, 1, 0, 1])
        y_prob = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        assert brier_score(y_true, y_prob) == 0.0

    def test_worst_predictions(self):
        """Completely wrong predictions should yield Brier score = 1."""
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.0, 1.0, 0.0, 1.0])
        assert brier_score(y_true, y_prob) == 1.0

    def test_random_predictions(self):
        """All-0.5 predictions should yield Brier score = 0.25."""
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        assert abs(brier_score(y_true, y_prob) - 0.25) < 1e-10

    def test_known_value(self):
        """Test a hand-calculated Brier score.

        y_true = [1, 0], y_prob = [0.8, 0.3]
        BS = mean((0.8-1)^2, (0.3-0)^2) = mean(0.04, 0.09) = 0.065
        """
        y_true = np.array([1, 0])
        y_prob = np.array([0.8, 0.3])
        expected = (0.04 + 0.09) / 2
        assert abs(brier_score(y_true, y_prob) - expected) < 1e-10

    def test_single_prediction(self):
        """Brier score with a single sample."""
        assert brier_score(np.array([1]), np.array([0.7])) == pytest.approx(0.09)

    def test_returns_float(self):
        result = brier_score(np.array([1, 0]), np.array([0.6, 0.4]))
        assert isinstance(result, float)


class TestAccuracy:
    """Test accuracy metric on known inputs."""

    def test_perfect_accuracy(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2])
        assert accuracy(y_true, y_prob) == 1.0

    def test_zero_accuracy(self):
        """Completely wrong predictions at threshold 0.5."""
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8])
        assert accuracy(y_true, y_prob) == 0.0

    def test_half_accuracy(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.9, 0.9, 0.1, 0.1])
        assert accuracy(y_true, y_prob) == 0.5

    def test_custom_threshold(self):
        """With threshold=0.7, only proba >= 0.7 predicts class 1."""
        y_true = np.array([1, 1, 0])
        y_prob = np.array([0.8, 0.6, 0.3])
        # At threshold 0.7: predictions are [1, 0, 0]
        # Correct: y_true[0]=1 matches pred=1, y_true[1]=1 vs pred=0 (wrong),
        #          y_true[2]=0 matches pred=0
        assert accuracy(y_true, y_prob, threshold=0.7) == pytest.approx(2.0 / 3.0)

    def test_all_same_class(self):
        y_true = np.array([1, 1, 1])
        y_prob = np.array([0.6, 0.7, 0.8])
        assert accuracy(y_true, y_prob) == 1.0


class TestLogLoss:
    """Test log loss metric."""

    def test_perfect_predictions_near_zero(self):
        """Near-perfect predictions should give very low log loss."""
        y_true = np.array([1, 0, 1])
        y_prob = np.array([0.999, 0.001, 0.999])
        result = log_loss(y_true, y_prob)
        assert result < 0.01

    def test_random_predictions(self):
        """All-0.5 predictions should give log loss = -ln(0.5) ~ 0.693."""
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        expected = -np.log(0.5)
        assert abs(log_loss(y_true, y_prob) - expected) < 1e-6


class TestCalibrationCurve:
    """Test calibration curve computation."""

    def test_returns_correct_structure(self):
        y_true = np.random.RandomState(42).randint(0, 2, size=200)
        y_prob = np.random.RandomState(42).uniform(0, 1, size=200)
        result = calibration_curve(y_true, y_prob, n_bins=10)

        assert "bin_centers" in result
        assert "actual_rates" in result
        assert "bin_counts" in result
        assert "n_bins" in result
        assert result["n_bins"] == 10

    def test_bin_counts_sum_to_total(self):
        n = 500
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=n)
        y_prob = rng.uniform(0, 1, size=n)
        result = calibration_curve(y_true, y_prob, n_bins=10)

        assert sum(result["bin_counts"]) == n

    def test_bin_centers_within_range(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=200)
        y_prob = rng.uniform(0, 1, size=200)
        result = calibration_curve(y_true, y_prob, n_bins=10)

        for center in result["bin_centers"]:
            assert 0.0 <= center <= 1.0

    def test_actual_rates_within_range(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=200)
        y_prob = rng.uniform(0, 1, size=200)
        result = calibration_curve(y_true, y_prob, n_bins=10)

        for rate in result["actual_rates"]:
            assert 0.0 <= rate <= 1.0

    def test_fewer_bins(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=100)
        y_prob = rng.uniform(0, 1, size=100)
        result = calibration_curve(y_true, y_prob, n_bins=5)

        assert result["n_bins"] == 5
        assert len(result["bin_centers"]) <= 5

    def test_all_predictions_in_one_bin(self):
        """When all predictions are nearly the same value, most bins are empty."""
        y_true = np.array([1, 0, 1, 0, 1])
        y_prob = np.array([0.55, 0.55, 0.55, 0.55, 0.55])
        result = calibration_curve(y_true, y_prob, n_bins=10)

        # Only one bin should have data
        assert len(result["bin_centers"]) == 1
        assert result["bin_counts"][0] == 5


class TestExpectedCalibrationError:
    """Test ECE metric."""

    def test_perfectly_calibrated(self):
        """If actual rates exactly match predicted rates, ECE should be 0.

        Construct data so each bin has predicted prob = actual win rate.
        """
        # 10 samples per bin at bin centers: 0.05, 0.15, ..., 0.95
        # In each bin, the actual outcome rate matches the predicted prob
        rng = np.random.RandomState(42)
        y_true_parts = []
        y_prob_parts = []

        for i in range(10):
            p = (i + 0.5) / 10  # bin center: 0.05, 0.15, ..., 0.95
            n = 1000  # large n so the actual rate converges to p
            outcomes = rng.binomial(1, p, size=n)
            y_true_parts.append(outcomes)
            y_prob_parts.append(np.full(n, p))

        y_true = np.concatenate(y_true_parts)
        y_prob = np.concatenate(y_prob_parts)

        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        # With 1000 samples per bin, the actual rate should be very close to p
        assert ece < 0.03, f"ECE for near-perfect calibration should be near 0, got {ece}"

    def test_ece_non_negative(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=200)
        y_prob = rng.uniform(0, 1, size=200)
        ece = expected_calibration_error(y_true, y_prob)
        assert ece >= 0.0

    def test_constant_overconfident_predictions(self):
        """Predicting 0.9 for all matches when true rate is 0.5 should give high ECE."""
        y_true = np.array([1, 0] * 100)
        y_prob = np.full(200, 0.9)
        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        # Actual rate = 0.5, predicted = 0.9, so |0.5 - 0.9| = 0.4
        assert ece > 0.3, f"Overconfident predictions should have high ECE, got {ece}"

    def test_empty_data(self):
        """ECE on empty data should return 0."""
        ece = expected_calibration_error(np.array([]), np.array([]), n_bins=10)
        assert ece == 0.0
