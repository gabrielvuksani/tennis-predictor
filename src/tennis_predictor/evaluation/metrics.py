"""Evaluation metrics for tennis prediction.

Prioritizes calibration (Brier score) over accuracy. A well-calibrated model
that says "60% chance" should be right 60% of the time — this is what matters
for betting strategy and decision-making.

Key finding from research: calibration-optimized models achieved +34.69% ROI
while accuracy-optimized models achieved -35.17% ROI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score — the primary metric. Lower is better.

    Measures the mean squared error of probability predictions.
    - Perfect: 0.0
    - Random (0.5): 0.25
    - Bookmaker baseline: ~0.196
    - Good model: < 0.20
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def accuracy(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """Classification accuracy."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return float(np.mean(y_true == y_pred))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-7) -> float:
    """Binary cross-entropy / log loss."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute calibration curve data.

    Groups predictions into bins and computes the actual win rate vs predicted
    probability for each bin. A perfectly calibrated model produces a diagonal.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    actual_rates = []
    bin_counts = []

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:  # Include upper bound in last bin
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        count = mask.sum()
        if count > 0:
            bin_centers.append(float(np.mean(y_prob[mask])))
            actual_rates.append(float(np.mean(y_true[mask])))
            bin_counts.append(int(count))

    return {
        "bin_centers": bin_centers,
        "actual_rates": actual_rates,
        "bin_counts": bin_counts,
        "n_bins": n_bins,
    }


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Weighted average of |accuracy - confidence| per bin.
    Lower is better. 0 = perfectly calibrated.
    """
    cal = calibration_curve(y_true, y_prob, n_bins)
    total = sum(cal["bin_counts"])
    if total == 0:
        return 0.0

    ece = sum(
        count * abs(actual - predicted)
        for actual, predicted, count
        in zip(cal["actual_rates"], cal["bin_centers"], cal["bin_counts"])
    ) / total

    return float(ece)


def upset_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    is_upset: np.ndarray,
) -> dict:
    """Metrics specifically for upset prediction.

    An upset is when the lower-ranked player wins. The model's ability to
    identify upsets is crucial — this is where value exists in betting.

    Args:
        y_true: Actual outcomes.
        y_prob: Predicted probabilities for player 1.
        is_upset: Boolean array where True = the actual outcome was an upset.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    is_upset = np.asarray(is_upset, dtype=bool)

    if not is_upset.any():
        return {"n_upsets": 0}

    upset_mask = is_upset
    non_upset_mask = ~is_upset

    # For upsets: the underdog won, so the model should have assigned
    # a lower probability to the favorite (or higher to the underdog)
    upset_probs = y_prob[upset_mask]
    upset_true = y_true[upset_mask]

    # Accuracy on upsets vs non-upsets
    upset_preds = (upset_probs >= 0.5).astype(int)
    non_upset_preds = (y_prob[non_upset_mask] >= 0.5).astype(int)

    upset_accuracy = float(np.mean(upset_preds == upset_true)) if len(upset_true) > 0 else np.nan
    non_upset_accuracy = (
        float(np.mean(non_upset_preds == y_true[non_upset_mask]))
        if non_upset_mask.any() else np.nan
    )

    # Brier score on upsets vs non-upsets
    upset_brier = brier_score(upset_true, upset_probs)
    non_upset_brier = (
        brier_score(y_true[non_upset_mask], y_prob[non_upset_mask])
        if non_upset_mask.any() else np.nan
    )

    # How often does the model give the upset side > 40% probability?
    # (indicating it sees some chance of an upset)
    upset_detected = np.mean(
        ((upset_true == 1) & (upset_probs >= 0.4)) |
        ((upset_true == 0) & (upset_probs <= 0.6))
    )

    return {
        "n_upsets": int(upset_mask.sum()),
        "n_non_upsets": int(non_upset_mask.sum()),
        "upset_pct": float(upset_mask.mean()),
        "upset_accuracy": upset_accuracy,
        "non_upset_accuracy": non_upset_accuracy,
        "upset_brier": upset_brier,
        "non_upset_brier": non_upset_brier,
        "upset_detection_rate": float(upset_detected),
    }


def roi_simulation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    odds_p1: np.ndarray,
    odds_p2: np.ndarray,
    stake: float = 1.0,
    min_edge: float = 0.05,
    kelly_fraction: float = 0.25,
) -> dict:
    """Simulate betting ROI using the model's predictions.

    Uses a fractional Kelly criterion strategy: only bet when the model
    disagrees with the market by at least min_edge, and size bets using
    a fraction of the Kelly optimal.

    Args:
        y_true: Actual outcomes (1 = p1 won).
        y_prob: Model's predicted probability for p1.
        odds_p1: Decimal odds for p1.
        odds_p2: Decimal odds for p2.
        stake: Base stake per bet.
        min_edge: Minimum edge (model_prob - implied_prob) to trigger a bet.
        kelly_fraction: Fraction of Kelly criterion to use (0.25 = quarter Kelly).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    odds_p1 = np.asarray(odds_p1, dtype=float)
    odds_p2 = np.asarray(odds_p2, dtype=float)

    # Implied probabilities from odds
    implied_p1 = 1.0 / odds_p1
    implied_p2 = 1.0 / odds_p2

    total_staked = 0.0
    total_returned = 0.0
    bets = []

    for i in range(len(y_true)):
        if np.isnan(odds_p1[i]) or np.isnan(odds_p2[i]):
            continue

        model_p1 = y_prob[i]
        model_p2 = 1.0 - model_p1

        # Check for edge on p1
        edge_p1 = model_p1 - implied_p1[i]
        # Check for edge on p2
        edge_p2 = model_p2 - implied_p2[i]

        bet_side = None
        edge = 0.0

        if edge_p1 >= min_edge and edge_p1 >= edge_p2:
            bet_side = "p1"
            edge = edge_p1
            kelly = (model_p1 * odds_p1[i] - 1) / (odds_p1[i] - 1) if odds_p1[i] > 1 else 0
        elif edge_p2 >= min_edge:
            bet_side = "p2"
            edge = edge_p2
            kelly = (model_p2 * odds_p2[i] - 1) / (odds_p2[i] - 1) if odds_p2[i] > 1 else 0
        else:
            continue

        if kelly <= 0:
            continue

        bet_size = stake * kelly * kelly_fraction
        bet_size = min(bet_size, stake * 5)  # Cap at 5x base stake

        if bet_side == "p1":
            won = y_true[i] == 1
            payout = bet_size * odds_p1[i] if won else 0
        else:
            won = y_true[i] == 0
            payout = bet_size * odds_p2[i] if won else 0

        total_staked += bet_size
        total_returned += payout

        bets.append({
            "side": bet_side,
            "edge": edge,
            "kelly": kelly,
            "bet_size": bet_size,
            "won": won,
            "payout": payout,
            "profit": payout - bet_size,
        })

    n_bets = len(bets)
    profit = total_returned - total_staked
    roi = profit / total_staked if total_staked > 0 else 0.0
    win_rate = sum(1 for b in bets if b["won"]) / n_bets if n_bets > 0 else 0.0

    return {
        "n_bets": n_bets,
        "total_staked": total_staked,
        "total_returned": total_returned,
        "profit": profit,
        "roi": roi,
        "roi_pct": roi * 100,
        "win_rate": win_rate,
        "avg_edge": float(np.mean([b["edge"] for b in bets])) if bets else 0.0,
        "avg_kelly": float(np.mean([b["kelly"] for b in bets])) if bets else 0.0,
        "avg_bet_size": float(np.mean([b["bet_size"] for b in bets])) if bets else 0.0,
        "max_drawdown": _compute_max_drawdown(bets),
        "bets": bets,
    }


def _compute_max_drawdown(bets: list[dict]) -> float:
    """Compute maximum drawdown from bet sequence."""
    if not bets:
        return 0.0

    cumulative = np.cumsum([b["profit"] for b in bets])
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0


def full_evaluation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    is_upset: np.ndarray | None = None,
    odds_p1: np.ndarray | None = None,
    odds_p2: np.ndarray | None = None,
    label: str = "Model",
) -> dict:
    """Run all evaluation metrics."""
    results = {
        "label": label,
        "n_matches": len(y_true),
        "accuracy": accuracy(y_true, y_prob),
        "brier_score": brier_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
        "calibration": calibration_curve(y_true, y_prob),
    }

    if is_upset is not None:
        results["upset_metrics"] = upset_metrics(y_true, y_prob, is_upset)

    if odds_p1 is not None and odds_p2 is not None:
        results["roi"] = roi_simulation(y_true, y_prob, odds_p1, odds_p2)

    return results


def compare_models(evaluations: list[dict]) -> pd.DataFrame:
    """Create a comparison table of model evaluations."""
    rows = []
    for ev in evaluations:
        row = {
            "Model": ev["label"],
            "Matches": ev["n_matches"],
            "Accuracy": f"{ev['accuracy']:.3f}",
            "Brier Score": f"{ev['brier_score']:.4f}",
            "Log Loss": f"{ev['log_loss']:.4f}",
            "ECE": f"{ev['ece']:.4f}",
        }
        if "upset_metrics" in ev and ev["upset_metrics"].get("n_upsets", 0) > 0:
            row["Upset Acc"] = f"{ev['upset_metrics']['upset_accuracy']:.3f}"
            row["Upset Det"] = f"{ev['upset_metrics']['upset_detection_rate']:.3f}"
        if "roi" in ev:
            row["ROI %"] = f"{ev['roi']['roi_pct']:.1f}"
            row["Bets"] = ev["roi"]["n_bets"]
        rows.append(row)

    return pd.DataFrame(rows)
