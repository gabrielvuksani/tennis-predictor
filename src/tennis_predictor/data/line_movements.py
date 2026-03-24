"""Betting line movement tracker — captures smart money signals.

Line movements tell us where sharp bettors are putting their money.
A line that moves AGAINST the public favorite signals sharp action on the underdog.
This is one of the most predictive signals available.

Source: Bovada API (free, no key) — provides pre-match odds that we snapshot
over time to track movement direction and magnitude.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from tennis_predictor.config import CACHE_DIR


def track_line_movements(upcoming_matches: list[dict]) -> dict[str, dict]:
    """Track odds movements for upcoming matches.

    Compares current odds against our stored snapshots from earlier.
    Returns movement signals: positive = line moving toward player, negative = moving away.
    """
    movement_dir = CACHE_DIR / "line_movements"
    movement_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for match in upcoming_matches:
        p1 = match.get("player1", "")
        p2 = match.get("player2", "")
        odds_p1 = match.get("odds_p1")
        odds_p2 = match.get("odds_p2")

        if not p1 or not p2 or odds_p1 is None or odds_p2 is None:
            continue

        key = _match_key(p1, p2)
        history_file = movement_dir / f"{key}.json"

        # Load history
        history = []
        if history_file.exists():
            history = json.loads(history_file.read_text())

        # Add current snapshot
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "odds_p1": odds_p1,
            "odds_p2": odds_p2,
            "implied_p1": 1.0 / odds_p1 if odds_p1 > 0 else 0.5,
            "implied_p2": 1.0 / odds_p2 if odds_p2 > 0 else 0.5,
        }
        history.append(snapshot)

        # Keep last 20 snapshots
        history = history[-20:]
        history_file.write_text(json.dumps(history, indent=2))

        # Compute movement
        movement = _compute_movement(history)
        results[key] = movement

    return results


def _compute_movement(history: list[dict]) -> dict:
    """Compute line movement metrics from odds history."""
    if len(history) < 2:
        return {
            "direction": 0.0,
            "magnitude": 0.0,
            "sharp_signal": 0.0,
            "n_snapshots": len(history),
        }

    first = history[0]
    last = history[-1]

    # Opening vs current implied probability
    opening_p1 = first.get("implied_p1", 0.5)
    current_p1 = last.get("implied_p1", 0.5)

    # Direction: positive = line moving toward p1 (p1 becoming more favored)
    direction = current_p1 - opening_p1

    # Magnitude: how much the line has moved (in probability points)
    magnitude = abs(direction)

    # Sharp signal: large moves against the public favorite suggest sharp money
    # If the favorite's implied prob DECREASES, sharps may be on the underdog
    sharp_signal = 0.0
    if opening_p1 > 0.5 and direction < -0.02:
        sharp_signal = abs(direction)  # Sharps on underdog (p2)
    elif opening_p1 < 0.5 and direction > 0.02:
        sharp_signal = abs(direction)  # Sharps on underdog (p1)

    return {
        "opening_p1": opening_p1,
        "current_p1": current_p1,
        "direction": direction,
        "magnitude": magnitude,
        "sharp_signal": sharp_signal,
        "n_snapshots": len(history),
    }


def get_line_features(p1_name: str, p2_name: str) -> dict:
    """Get line movement features for a specific match."""
    movement_dir = CACHE_DIR / "line_movements"
    key = _match_key(p1_name, p2_name)
    history_file = movement_dir / f"{key}.json"

    if not history_file.exists():
        return {
            "line_direction": 0.0,
            "line_magnitude": 0.0,
            "sharp_signal": 0.0,
            "opening_implied_p1": np.nan,
            "current_implied_p1": np.nan,
        }

    history = json.loads(history_file.read_text())
    movement = _compute_movement(history)

    return {
        "line_direction": movement["direction"],
        "line_magnitude": movement["magnitude"],
        "sharp_signal": movement["sharp_signal"],
        "opening_implied_p1": movement.get("opening_p1", np.nan),
        "current_implied_p1": movement.get("current_p1", np.nan),
    }


def _match_key(p1: str, p2: str) -> str:
    """Create a stable key for a match."""
    names = sorted([p1.lower().strip(), p2.lower().strip()])
    return "_vs_".join(n.replace(" ", "_").replace(".", "") for n in names)
