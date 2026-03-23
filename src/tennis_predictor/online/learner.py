"""Online learning pipeline for self-improving predictions.

The system learns from every match result:
1. Elo/Glicko-2 ratings update automatically after each match (inherently online)
2. The GBM ensemble retrains periodically on the expanding dataset
3. Concept drift detection triggers emergency retraining when needed
4. Feature importance is recalculated to detect shifts in what matters

This is what makes the model "self-learning" — it never stops improving.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tennis_predictor.config import ONLINE_CONFIG, PROCESSED_DIR
from tennis_predictor.temporal.guard import TemporalGuard


class OnlineLearner:
    """Manages the self-learning lifecycle.

    After every match or batch of matches:
    1. Update Elo/Glicko-2 ratings (instant, always happens)
    2. Log the prediction error (for drift detection)
    3. If enough new data: retrain the ensemble
    4. If drift detected: emergency retrain with recency weighting

    State is persisted to disk so the system remembers across sessions.
    """

    def __init__(
        self,
        guard: TemporalGuard | None = None,
        model: Any = None,
        state_dir: Path | None = None,
    ):
        self.guard = guard or TemporalGuard()
        self.model = model
        self.state_dir = state_dir or PROCESSED_DIR / "online_state"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Prediction log for drift detection
        self.prediction_log: list[dict] = []
        self.last_retrain_date: datetime | None = None
        self.retrain_count: int = 0

        # Drift detector state
        self.drift_detector = ADWINDriftDetector(
            confidence=ONLINE_CONFIG["drift_confidence"]
        )

        # Load persisted state if available
        self._load_state()

    def process_match(
        self,
        match: pd.Series,
        actual_result: int,
        predicted_proba: float | None = None,
    ) -> dict:
        """Process a new match result.

        Args:
            match: Match data row.
            actual_result: 1 if player1 won, 0 if player2 won.
            predicted_proba: Our model's pre-match prediction (if available).

        Returns:
            Dict with update details and any drift warnings.
        """
        result = {
            "match_id": match.get("match_id", "unknown"),
            "elo_updated": False,
            "drift_detected": False,
            "retrain_triggered": False,
        }

        # 1. Update Elo/Glicko-2 (always)
        try:
            features = self.guard.extract_pre_match_state(match)
            self.guard.update_state(match, actual_result)
            result["elo_updated"] = True
        except Exception as e:
            result["error"] = str(e)

        # 2. Log prediction error (if we made a prediction)
        if predicted_proba is not None:
            error = (predicted_proba - actual_result) ** 2  # Brier score contribution
            self.prediction_log.append({
                "timestamp": datetime.now().isoformat(),
                "match_date": str(match.get("tourney_date", "")),
                "predicted": predicted_proba,
                "actual": actual_result,
                "error": error,
            })

            # 3. Check for drift
            drift = self.drift_detector.update(error)
            if drift:
                result["drift_detected"] = True
                result["drift_message"] = (
                    "Concept drift detected — model performance has shifted. "
                    "Emergency retrain recommended."
                )

        # 4. Check if periodic retrain is due
        retrain_freq = ONLINE_CONFIG["retrain_frequency_days"]
        if self.last_retrain_date is not None:
            days_since = (datetime.now() - self.last_retrain_date).days
            if days_since >= retrain_freq:
                result["retrain_triggered"] = True
                result["retrain_reason"] = f"Periodic retrain (every {retrain_freq} days)"

        return result

    def retrain(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        model_factory=None,
        recency_weight: bool = False,
    ) -> Any:
        """Retrain the model on the expanded dataset.

        Args:
            X_train: Feature matrix.
            y_train: Target array.
            model_factory: Callable that creates a fresh model.
            recency_weight: If True, weight recent matches higher.

        Returns:
            The retrained model.
        """
        if model_factory is None:
            from tennis_predictor.models.ensemble import create_default_ensemble
            model_factory = create_default_ensemble

        model = model_factory()

        if recency_weight:
            # Exponential decay: most recent matches get highest weight
            n = len(y_train)
            weights = np.exp(np.linspace(-2, 0, n))
            weights /= weights.sum()
            # Some models support sample_weight
            try:
                model.fit(X_train, y_train, sample_weight=weights)
            except TypeError:
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        self.model = model
        self.last_retrain_date = datetime.now()
        self.retrain_count += 1
        self._save_state()

        return model

    def get_performance_trend(self, window: int = 100) -> dict:
        """Get recent prediction performance statistics."""
        if not self.prediction_log:
            return {"n_predictions": 0}

        recent = self.prediction_log[-window:]
        errors = [p["error"] for p in recent]
        predictions = [p["predicted"] for p in recent]
        actuals = [p["actual"] for p in recent]

        accuracy = np.mean(
            [(1 if p >= 0.5 else 0) == a for p, a in zip(predictions, actuals)]
        )

        return {
            "n_predictions": len(recent),
            "avg_brier_score": float(np.mean(errors)),
            "recent_accuracy": float(accuracy),
            "brier_trend": _compute_trend(errors),
            "total_predictions": len(self.prediction_log),
            "total_retrains": self.retrain_count,
        }

    def _save_state(self) -> None:
        """Persist learner state to disk."""
        state = {
            "prediction_log": self.prediction_log[-10000:],  # Keep last 10K
            "last_retrain_date": (
                self.last_retrain_date.isoformat() if self.last_retrain_date else None
            ),
            "retrain_count": self.retrain_count,
            "drift_detector": self.drift_detector.to_dict(),
        }
        state_file = self.state_dir / "learner_state.json"
        state_file.write_text(json.dumps(state, indent=2))

        # Save model separately (binary)
        if self.model is not None:
            model_file = self.state_dir / "model.pkl"
            with open(model_file, "wb") as f:
                pickle.dump(self.model, f)

        # Save TemporalGuard state
        guard_file = self.state_dir / "guard_state.pkl"
        with open(guard_file, "wb") as f:
            pickle.dump(self.guard.state, f)

    def _load_state(self) -> None:
        """Load persisted state."""
        state_file = self.state_dir / "learner_state.json"
        if state_file.exists():
            state = json.loads(state_file.read_text())
            self.prediction_log = state.get("prediction_log", [])
            lrd = state.get("last_retrain_date")
            self.last_retrain_date = (
                datetime.fromisoformat(lrd) if lrd else None
            )
            self.retrain_count = state.get("retrain_count", 0)
            if "drift_detector" in state:
                self.drift_detector = ADWINDriftDetector.from_dict(
                    state["drift_detector"]
                )

        model_file = self.state_dir / "model.pkl"
        if model_file.exists() and self.model is None:
            with open(model_file, "rb") as f:
                self.model = pickle.load(f)

        guard_file = self.state_dir / "guard_state.pkl"
        if guard_file.exists():
            with open(guard_file, "rb") as f:
                self.guard.state = pickle.load(f)


class ADWINDriftDetector:
    """ADWIN-inspired concept drift detector.

    Monitors a stream of prediction errors and detects when the error
    distribution shifts significantly. This indicates the model's
    assumptions no longer hold (concept drift).

    Simplified implementation: compares rolling windows of errors.
    """

    def __init__(self, confidence: float = 0.002, min_window: int = 30):
        self.confidence = confidence
        self.min_window = min_window
        self.window: list[float] = []
        self.drift_count: int = 0

    def update(self, error: float) -> bool:
        """Add new error and check for drift.

        Returns True if drift is detected.
        """
        self.window.append(error)

        if len(self.window) < self.min_window * 2:
            return False

        # Compare first half vs second half of window
        n = len(self.window)
        mid = n // 2

        first_half = np.array(self.window[:mid])
        second_half = np.array(self.window[mid:])

        mean_diff = abs(np.mean(second_half) - np.mean(first_half))

        # Hoeffding bound
        m = min(len(first_half), len(second_half))
        epsilon = np.sqrt(np.log(2 / self.confidence) / (2 * m))

        if mean_diff > epsilon:
            # Drift detected — shrink window to recent data
            self.window = list(second_half)
            self.drift_count += 1
            return True

        # Keep window bounded
        if len(self.window) > 1000:
            self.window = self.window[-500:]

        return False

    def to_dict(self) -> dict:
        return {
            "confidence": self.confidence,
            "min_window": self.min_window,
            "window": self.window[-500:],
            "drift_count": self.drift_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ADWINDriftDetector:
        detector = cls(
            confidence=data.get("confidence", 0.002),
            min_window=data.get("min_window", 30),
        )
        detector.window = data.get("window", [])
        detector.drift_count = data.get("drift_count", 0)
        return detector


def _compute_trend(values: list[float], window: int = 20) -> str:
    """Compute whether errors are trending up, down, or stable."""
    if len(values) < window * 2:
        return "insufficient_data"

    recent = np.mean(values[-window:])
    older = np.mean(values[-window * 2:-window])

    diff = recent - older
    if diff > 0.01:
        return "degrading"
    elif diff < -0.01:
        return "improving"
    else:
        return "stable"
