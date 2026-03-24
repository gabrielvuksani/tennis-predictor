"""Selective prediction strategy — only predict when we have an edge.

Research proves profit comes from knowing WHEN you have an edge:
- GNN paper: 3.26% ROI on high-intransitivity matches (-5.49% on all)
- TennisPredictor: 85.7% accuracy when both models agree (63% overall)
- Walsh & Joshi: calibration-selected models got +34.69% ROI

This module identifies high-confidence predictions and computes
expected value for potential betting decisions.
"""

from __future__ import annotations

import numpy as np


def compute_edge_signals(predictions: list[dict]) -> list[dict]:
    """Analyze predictions and compute edge signals.

    Returns predictions enriched with edge analysis:
    - confidence_tier: "high", "medium", "low"
    - model_agreement: how much Elo and point sim agree
    - expected_value: EV if betting at given odds
    - recommendation: "strong_predict", "predict", "skip"
    """
    for pred in predictions:
        prob = max(pred.get("prob_p1", 0.5), pred.get("prob_p2", 0.5))
        confidence = pred.get("confidence", 0)

        # Confidence tier
        if confidence >= 0.6:
            pred["confidence_tier"] = "high"
        elif confidence >= 0.3:
            pred["confidence_tier"] = "medium"
        else:
            pred["confidence_tier"] = "low"

        # Model agreement (if we have both Elo and point sim)
        elo_prob = pred.get("elo_prob", pred.get("prob_p1", 0.5))
        sim_prob = pred.get("sim_prob", pred.get("prob_p1", 0.5))
        agreement = 1.0 - abs(elo_prob - sim_prob)
        pred["model_agreement"] = round(agreement, 3)

        # Recommendation
        if confidence >= 0.6 and agreement >= 0.8:
            pred["recommendation"] = "strong_predict"
        elif confidence >= 0.3:
            pred["recommendation"] = "predict"
        else:
            pred["recommendation"] = "skip"

        # Expected value (if odds are available)
        odds = pred.get("odds_decimal_fav")
        if odds and odds > 1:
            ev = prob * odds - 1
            pred["expected_value"] = round(ev, 3)
        else:
            pred["expected_value"] = None

    return predictions


def filter_high_edge(predictions: list[dict]) -> list[dict]:
    """Filter to only high-edge predictions.

    These are the matches where the model is most likely to be right.
    Research: TennisPredictor gets 85.7% when models agree.
    """
    return [p for p in predictions if p.get("recommendation") in ("strong_predict", "predict")]


def compute_selective_accuracy(
    predictions: list[dict],
    actuals: list[int],
) -> dict:
    """Compute accuracy at different confidence thresholds.

    Shows the accuracy-coverage tradeoff.
    """
    results = {}
    for threshold in [0.0, 0.2, 0.4, 0.6, 0.8]:
        mask = [abs(p.get("prob_p1", 0.5) - 0.5) * 2 >= threshold
                for p in predictions]
        n = sum(mask)
        if n == 0:
            continue

        correct = sum(
            1 for pred, actual, m in zip(predictions, actuals, mask)
            if m and ((pred["prob_p1"] >= 0.5) == (actual == 1))
        )
        results[f"threshold_{threshold:.1f}"] = {
            "coverage": n / len(predictions),
            "accuracy": correct / n if n > 0 else 0,
            "n_predictions": n,
        }

    return results
