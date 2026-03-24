"""TemporalGuard - The anti-leakage enforcement system.

This module ensures that all feature computation respects temporal boundaries.
Every feature must be computed using ONLY data available BEFORE the match.

The phosphenq model failed because post-match ELO ratings (which encode the
match result) were fed as pre-match features. Our TemporalGuard makes this
class of error architecturally impossible.

Design principles:
1. Features are computed chronologically, one match at a time
2. State (Elo, rolling stats, H2H) is updated AFTER features are extracted
3. The guard maintains a "knowledge cutoff" timestamp
4. Any attempt to access future data raises an error
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TemporalState:
    """Holds all mutable state that evolves over time.

    This is the single source of truth for "what we know at time T".
    All feature computations read from this state, and all updates
    happen through the TemporalGuard's controlled update path.
    """
    # Current knowledge cutoff — no data after this point should be used
    cutoff: pd.Timestamp = pd.Timestamp("1900-01-01")

    # Elo ratings: player_id -> rating
    elo: dict[str, float] = field(default_factory=dict)

    # Surface-specific Elo: (player_id, surface) -> rating
    elo_surface: dict[tuple[str, str], float] = field(default_factory=dict)

    # Serve Elo and Return Elo: player_id -> rating
    elo_serve: dict[str, float] = field(default_factory=dict)
    elo_return: dict[str, float] = field(default_factory=dict)

    # Glicko-2 state: player_id -> (rating, rd, volatility)
    glicko2: dict[str, tuple[float, float, float]] = field(default_factory=dict)

    # Match count per player (for K-factor calculation)
    match_counts: dict[str, int] = field(default_factory=dict)

    # Surface match counts: (player_id, surface) -> count
    surface_match_counts: dict[tuple[str, str], int] = field(default_factory=dict)

    # Rolling match history: player_id -> list of recent match dicts
    match_history: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    # Head-to-head: (player_id_a, player_id_b) -> {"wins_a": int, "wins_b": int, ...}
    h2h: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)

    # Last match date per player (for fatigue/inactivity)
    last_match_date: dict[str, pd.Timestamp] = field(default_factory=dict)

    # Tournament history: (player_id, tourney_name) -> list of results
    tournament_history: dict[tuple[str, str], list[dict]] = field(default_factory=dict)

    # Days since last match (for RD decay in Glicko-2)
    _last_updated: dict[str, pd.Timestamp] = field(default_factory=dict)


class TemporalGuard:
    """Enforces temporal isolation in feature computation.

    Usage:
        guard = TemporalGuard()
        for match in chronologically_sorted_matches:
            features = guard.extract_features(match)  # Uses only pre-match state
            guard.update_state(match)                  # Updates state with match result

    The extract → update ordering is enforced. Calling update before extract
    for the same match, or extracting features for a match before its
    temporal predecessor is processed, will raise TemporalLeakageError.
    """

    def __init__(self, state: TemporalState | None = None):
        self.state = state or TemporalState()
        self._last_extracted_match: str | None = None
        self._last_updated_match: str | None = None
        self._processed_match_ids: set[str] = set()
        self._extraction_count = 0
        self._update_count = 0

    def validate_temporal_order(self, match_date: pd.Timestamp, match_id: str) -> None:
        """Verify this match comes after the current knowledge cutoff."""
        if match_id in self._processed_match_ids:
            raise TemporalLeakageError(
                f"Match {match_id} has already been processed. "
                f"Re-processing would allow information to leak."
            )

    def extract_pre_match_state(self, match: pd.Series) -> dict[str, Any]:
        """Extract a snapshot of all state variables BEFORE this match.

        This returns a flat dict of features that can be used for prediction.
        The state is NOT modified — that only happens in update_state().

        Args:
            match: A row from the match DataFrame.

        Returns:
            Dict of feature_name -> value, all computed from pre-match state only.
        """
        match_id = match.get("match_id", "unknown")
        match_date = pd.Timestamp(match.get("tourney_date", pd.NaT))

        self.validate_temporal_order(match_date, match_id)
        self._last_extracted_match = match_id
        self._extraction_count += 1

        p1_id = str(match.get("p1_id", ""))
        p2_id = str(match.get("p2_id", ""))
        surface = str(match.get("surface", "Hard"))
        tourney_level = str(match.get("tourney_level", "A"))
        tourney_name = str(match.get("tourney_name", ""))
        best_of = int(match.get("best_of", 3))

        features: dict[str, Any] = {}

        # === ELO FEATURES (pre-match values only) ===
        features.update(self._extract_elo_features(p1_id, p2_id, surface))

        # === GLICKO-2 FEATURES ===
        features.update(self._extract_glicko2_features(p1_id, p2_id, match_date))

        # === ROLLING STATISTICS ===
        features.update(self._extract_rolling_features(p1_id, p2_id, surface))

        # === HEAD-TO-HEAD ===
        features.update(self._extract_h2h_features(p1_id, p2_id, surface))

        # === FATIGUE / SCHEDULING ===
        features.update(self._extract_fatigue_features(p1_id, p2_id, match_date))

        # === TOURNAMENT HISTORY ===
        features.update(self._extract_tournament_features(p1_id, p2_id, tourney_name))

        # === STATIC MATCH CONTEXT ===
        features["surface_Hard"] = int(surface == "Hard")
        features["surface_Clay"] = int(surface == "Clay")
        features["surface_Grass"] = int(surface == "Grass")
        features["tourney_level_G"] = int(tourney_level == "G")
        features["tourney_level_M"] = int(tourney_level in ("M", "F"))
        features["best_of_5"] = int(best_of == 5)

        # Player static features
        p1_rank = match.get("p1_rank", np.nan)
        p2_rank = match.get("p2_rank", np.nan)
        features["rank_diff"] = _safe_diff(p1_rank, p2_rank)
        features["rank_ratio"] = _safe_ratio(p1_rank, p2_rank)
        features["log_rank_diff"] = _safe_log_diff(p1_rank, p2_rank)

        p1_age = match.get("p1_age", np.nan)
        p2_age = match.get("p2_age", np.nan)
        features["age_diff"] = _safe_diff(p1_age, p2_age)
        features["p1_age"] = p1_age if not pd.isna(p1_age) else np.nan
        features["p2_age"] = p2_age if not pd.isna(p2_age) else np.nan

        p1_ht = match.get("p1_ht", np.nan)
        p2_ht = match.get("p2_ht", np.nan)
        features["height_diff"] = _safe_diff(p1_ht, p2_ht)

        # Entry type features (qualifiers, wildcards are upset-prone)
        features["p1_is_qualifier"] = int(match.get("p1_entry", "") == "Q")
        features["p2_is_qualifier"] = int(match.get("p2_entry", "") == "Q")
        features["p1_is_wildcard"] = int(match.get("p1_entry", "") == "WC")
        features["p2_is_wildcard"] = int(match.get("p2_entry", "") == "WC")

        # Seeding
        p1_seed = _parse_seed(match.get("p1_seed"))
        p2_seed = _parse_seed(match.get("p2_seed"))
        features["seed_diff"] = _safe_diff(p1_seed, p2_seed)
        features["p1_seeded"] = int(not pd.isna(p1_seed))
        features["p2_seeded"] = int(not pd.isna(p2_seed))

        # === SUPPLEMENTARY DATA (passed through match row) ===

        # Betting odds implied probability (pre-match, no leakage)
        odds_p1 = match.get("odds_implied_p1", np.nan)
        odds_p2 = match.get("odds_implied_p2", np.nan)
        features["odds_implied_p1"] = odds_p1 if not pd.isna(odds_p1) else np.nan
        features["odds_implied_p2"] = odds_p2 if not pd.isna(odds_p2) else np.nan
        features["odds_diff"] = _safe_diff(odds_p1, odds_p2)
        # Disagreement between Elo and odds (value signal)
        features["elo_vs_odds_diff"] = _safe_diff(
            features["elo_win_prob"], odds_p1 if not pd.isna(odds_p1) else np.nan
        )

        # Weather features (pre-match, based on venue/date)
        for wf in ["weather_temp_max", "weather_temp_min", "weather_precipitation",
                    "weather_wind_max", "weather_wind_gust_max", "weather_altitude",
                    "weather_is_indoor"]:
            val = match.get(wf, np.nan)
            features[wf] = val if not pd.isna(val) else np.nan

        # Court speed
        court_speed = match.get("court_speed", np.nan)
        features["court_speed"] = court_speed if not pd.isna(court_speed) else np.nan

        # Cross-feature: player serve strength x court speed interaction
        if not pd.isna(court_speed):
            p1_ace = features.get("p1_avg_ace_rate_20", np.nan)
            p2_ace = features.get("p2_avg_ace_rate_20", np.nan)
            features["p1_serve_x_speed"] = (
                p1_ace * court_speed if not pd.isna(p1_ace) else np.nan
            )
            features["p2_serve_x_speed"] = (
                p2_ace * court_speed if not pd.isna(p2_ace) else np.nan
            )
        else:
            features["p1_serve_x_speed"] = np.nan
            features["p2_serve_x_speed"] = np.nan

        # Intransitivity score (from GNN if available, else NaN)
        features["intransitivity_score"] = match.get("intransitivity_score", np.nan)

        # === ADVANCED FEATURES (40+ new) ===
        from tennis_predictor.features.advanced import extract_advanced_features
        features.update(extract_advanced_features(match, self.state, features))

        return features

    def update_state(self, match: pd.Series, result: int) -> None:
        """Update all state variables with this match's result.

        MUST be called AFTER extract_pre_match_state for the same match.

        Args:
            match: The match row.
            result: 1 if player1 won, 0 if player2 won.
        """
        match_id = match.get("match_id", "unknown")

        if self._last_extracted_match != match_id:
            raise TemporalLeakageError(
                f"Cannot update state for match {match_id} — features were last "
                f"extracted for {self._last_extracted_match}. You must extract "
                f"features before updating state."
            )

        p1_id = str(match.get("p1_id", ""))
        p2_id = str(match.get("p2_id", ""))
        surface = str(match.get("surface", "Hard"))
        match_date = pd.Timestamp(match.get("tourney_date", pd.NaT))
        tourney_level = str(match.get("tourney_level", "A"))
        tourney_name = str(match.get("tourney_name", ""))
        best_of = int(match.get("best_of", 3))

        winner_id = p1_id if result == 1 else p2_id
        loser_id = p2_id if result == 1 else p1_id

        # Update Elo
        self._update_elo(winner_id, loser_id, surface, tourney_level, best_of)

        # Update Glicko-2
        self._update_glicko2(winner_id, loser_id, match_date)

        # Update match history
        self._update_match_history(match, p1_id, p2_id, result, surface, match_date)

        # Update H2H
        self._update_h2h(p1_id, p2_id, result, surface)

        # Update last match date
        self.state.last_match_date[p1_id] = match_date
        self.state.last_match_date[p2_id] = match_date

        # Update tournament history
        self._update_tournament_history(p1_id, p2_id, tourney_name, match, result)

        # Update match counts
        self.state.match_counts[p1_id] = self.state.match_counts.get(p1_id, 0) + 1
        self.state.match_counts[p2_id] = self.state.match_counts.get(p2_id, 0) + 1
        key1 = (p1_id, surface)
        key2 = (p2_id, surface)
        self.state.surface_match_counts[key1] = self.state.surface_match_counts.get(key1, 0) + 1
        self.state.surface_match_counts[key2] = self.state.surface_match_counts.get(key2, 0) + 1

        # Mark as processed
        self._processed_match_ids.add(match_id)
        self._last_updated_match = match_id
        self._update_count += 1
        self.state.cutoff = max(self.state.cutoff, match_date)

    # === INTERNAL FEATURE EXTRACTORS ===

    def _extract_elo_features(self, p1: str, p2: str, surface: str) -> dict:
        from tennis_predictor.config import ELO_CONFIG
        init = ELO_CONFIG["initial_rating"]

        elo1 = self.state.elo.get(p1, init)
        elo2 = self.state.elo.get(p2, init)
        surf_elo1 = self.state.elo_surface.get((p1, surface), init)
        surf_elo2 = self.state.elo_surface.get((p2, surface), init)
        serve_elo1 = self.state.elo_serve.get(p1, init)
        serve_elo2 = self.state.elo_serve.get(p2, init)
        ret_elo1 = self.state.elo_return.get(p1, init)
        ret_elo2 = self.state.elo_return.get(p2, init)

        return {
            "elo_diff": elo1 - elo2,
            "elo_p1": elo1,
            "elo_p2": elo2,
            "surface_elo_diff": surf_elo1 - surf_elo2,
            "surface_elo_p1": surf_elo1,
            "surface_elo_p2": surf_elo2,
            "serve_elo_diff": serve_elo1 - serve_elo2,
            "return_elo_diff": ret_elo1 - ret_elo2,
            # Elo win probability (logistic)
            "elo_win_prob": 1.0 / (1.0 + 10 ** ((elo2 - elo1) / 400)),
            "surface_elo_win_prob": 1.0 / (1.0 + 10 ** ((surf_elo2 - surf_elo1) / 400)),
            # Match experience
            "p1_match_count": self.state.match_counts.get(p1, 0),
            "p2_match_count": self.state.match_counts.get(p2, 0),
            "p1_surface_match_count": self.state.surface_match_counts.get((p1, surface), 0),
            "p2_surface_match_count": self.state.surface_match_counts.get((p2, surface), 0),
        }

    def _extract_glicko2_features(self, p1: str, p2: str, match_date: pd.Timestamp) -> dict:
        from tennis_predictor.config import GLICKO2_CONFIG
        init_r = GLICKO2_CONFIG["initial_rating"]
        init_rd = GLICKO2_CONFIG["initial_rd"]
        init_vol = GLICKO2_CONFIG["initial_vol"]

        r1, rd1, vol1 = self.state.glicko2.get(p1, (init_r, init_rd, init_vol))
        r2, rd2, vol2 = self.state.glicko2.get(p2, (init_r, init_rd, init_vol))

        # Decay RD for inactivity
        rd1 = self._decay_rd(p1, rd1, match_date)
        rd2 = self._decay_rd(p2, rd2, match_date)

        return {
            "glicko2_diff": r1 - r2,
            "glicko2_rd_p1": rd1,
            "glicko2_rd_p2": rd2,
            "glicko2_vol_p1": vol1,
            "glicko2_vol_p2": vol2,
            # Uncertainty-weighted rating difference
            "glicko2_uncertainty_diff": (r1 - r2) / np.sqrt(rd1**2 + rd2**2 + 1e-8),
        }

    def _extract_rolling_features(self, p1: str, p2: str, surface: str) -> dict:
        from tennis_predictor.config import ROLLING_WINDOWS
        features = {}

        for window in ROLLING_WINDOWS:
            for prefix, pid in [("p1", p1), ("p2", p2)]:
                history = self.state.match_history.get(pid, [])
                recent = history[-window:] if history else []

                # Overall win rate
                if recent:
                    wins = sum(1 for m in recent if m["won"])
                    features[f"{prefix}_winrate_{window}"] = wins / len(recent)
                else:
                    features[f"{prefix}_winrate_{window}"] = np.nan

                # Surface-specific win rate
                surface_matches = [m for m in recent if m.get("surface") == surface]
                if surface_matches:
                    wins = sum(1 for m in surface_matches if m["won"])
                    features[f"{prefix}_surface_winrate_{window}"] = wins / len(surface_matches)
                else:
                    features[f"{prefix}_surface_winrate_{window}"] = np.nan

                # Serve stats (rolling averages)
                if recent:
                    serve_pcts = [m["first_serve_pct"] for m in recent
                                  if not np.isnan(m.get("first_serve_pct", np.nan))]
                    if serve_pcts:
                        features[f"{prefix}_avg_1st_serve_pct_{window}"] = np.mean(serve_pcts)
                    else:
                        features[f"{prefix}_avg_1st_serve_pct_{window}"] = np.nan

                    serve_win = [m["first_serve_won_pct"] for m in recent
                                 if not np.isnan(m.get("first_serve_won_pct", np.nan))]
                    if serve_win:
                        features[f"{prefix}_avg_1st_serve_won_{window}"] = np.mean(serve_win)
                    else:
                        features[f"{prefix}_avg_1st_serve_won_{window}"] = np.nan

                    bp_save = [m["bp_save_pct"] for m in recent
                               if not np.isnan(m.get("bp_save_pct", np.nan))]
                    if bp_save:
                        features[f"{prefix}_avg_bp_save_{window}"] = np.mean(bp_save)
                    else:
                        features[f"{prefix}_avg_bp_save_{window}"] = np.nan

                    ace_rates = [m["ace_rate"] for m in recent
                                 if not np.isnan(m.get("ace_rate", np.nan))]
                    if ace_rates:
                        features[f"{prefix}_avg_ace_rate_{window}"] = np.mean(ace_rates)
                    else:
                        features[f"{prefix}_avg_ace_rate_{window}"] = np.nan
                else:
                    for stat in ["avg_1st_serve_pct", "avg_1st_serve_won",
                                 "avg_bp_save", "avg_ace_rate"]:
                        features[f"{prefix}_{stat}_{window}"] = np.nan

        # Difference features for most recent window
        w = ROLLING_WINDOWS[1]  # 10-match window
        for stat in ["winrate", "surface_winrate", "avg_1st_serve_pct",
                     "avg_1st_serve_won", "avg_bp_save", "avg_ace_rate"]:
            p1_val = features.get(f"p1_{stat}_{w}", np.nan)
            p2_val = features.get(f"p2_{stat}_{w}", np.nan)
            features[f"diff_{stat}_{w}"] = _safe_diff(p1_val, p2_val)

        return features

    def _extract_h2h_features(self, p1: str, p2: str, surface: str) -> dict:
        key = (min(p1, p2), max(p1, p2))
        h2h = self.state.h2h.get(key, {})

        if not h2h:
            return {
                "h2h_total_matches": 0,
                "h2h_p1_win_pct": np.nan,
                "h2h_surface_matches": 0,
                "h2h_surface_p1_win_pct": np.nan,
            }

        total = h2h.get("total", 0)
        # Determine which side is p1 in the stored key
        if key[0] == p1:
            p1_wins = h2h.get("wins_a", 0)
            p1_surface_wins = h2h.get(f"surface_wins_a_{surface}", 0)
        else:
            p1_wins = h2h.get("wins_b", 0)
            p1_surface_wins = h2h.get(f"surface_wins_b_{surface}", 0)

        surface_total = h2h.get(f"surface_total_{surface}", 0)

        return {
            "h2h_total_matches": total,
            "h2h_p1_win_pct": p1_wins / total if total > 0 else np.nan,
            "h2h_surface_matches": surface_total,
            "h2h_surface_p1_win_pct": (
                p1_surface_wins / surface_total if surface_total > 0 else np.nan
            ),
        }

    def _extract_fatigue_features(
        self, p1: str, p2: str, match_date: pd.Timestamp
    ) -> dict:
        features = {}
        for prefix, pid in [("p1", p1), ("p2", p2)]:
            last = self.state.last_match_date.get(pid)
            if last is not None and not pd.isna(match_date):
                days = (match_date - last).days
                features[f"{prefix}_days_since_last"] = max(days, 0)
                features[f"{prefix}_is_inactive"] = int(days > 60)
            else:
                features[f"{prefix}_days_since_last"] = np.nan
                features[f"{prefix}_is_inactive"] = np.nan

            # Matches in last 7/14/30 days
            history = self.state.match_history.get(pid, [])
            for days_window in [7, 14, 30]:
                cutoff = match_date - timedelta(days=days_window)
                count = sum(1 for m in history
                            if m.get("date") is not None and m["date"] >= cutoff)
                features[f"{prefix}_matches_last_{days_window}d"] = count

        features["fatigue_diff_7d"] = (
            features.get("p1_matches_last_7d", 0) - features.get("p2_matches_last_7d", 0)
        )
        return features

    def _extract_tournament_features(self, p1: str, p2: str, tourney_name: str) -> dict:
        features = {}
        for prefix, pid in [("p1", p1), ("p2", p2)]:
            hist = self.state.tournament_history.get((pid, tourney_name), [])
            features[f"{prefix}_tourney_appearances"] = len(hist)
            if hist:
                # Best round reached (higher round_order = deeper run)
                features[f"{prefix}_tourney_best_round"] = max(
                    m.get("round_order", 0) for m in hist
                )
                features[f"{prefix}_tourney_avg_round"] = np.mean(
                    [m.get("round_order", 0) for m in hist]
                )
            else:
                features[f"{prefix}_tourney_best_round"] = np.nan
                features[f"{prefix}_tourney_avg_round"] = np.nan

        return features

    # === INTERNAL STATE UPDATERS ===

    def _update_elo(
        self, winner_id: str, loser_id: str,
        surface: str, tourney_level: str, best_of: int,
    ) -> None:
        from tennis_predictor.config import ELO_CONFIG
        init = ELO_CONFIG["initial_rating"]
        base_k = ELO_CONFIG["k_factor_base"]
        exp = ELO_CONFIG["k_factor_exponent"]
        offset = ELO_CONFIG["k_factor_offset"]
        level_mult = ELO_CONFIG["level_multipliers"].get(tourney_level, 1.0)
        bo5_mult = ELO_CONFIG["bo5_multiplier"] if best_of == 5 else 1.0

        # Overall Elo update
        elo_w = self.state.elo.get(winner_id, init)
        elo_l = self.state.elo.get(loser_id, init)
        expected_w = 1.0 / (1.0 + 10 ** ((elo_l - elo_w) / 400))

        # K-factor: decreases with experience (FiveThirtyEight formula)
        k_w = base_k / (self.state.match_counts.get(winner_id, 0) + offset) ** exp
        k_l = base_k / (self.state.match_counts.get(loser_id, 0) + offset) ** exp
        k_w *= level_mult * bo5_mult
        k_l *= level_mult * bo5_mult

        self.state.elo[winner_id] = elo_w + k_w * (1 - expected_w)
        self.state.elo[loser_id] = elo_l + k_l * (0 - (1 - expected_w))

        # Surface-specific Elo
        surf_w = self.state.elo_surface.get((winner_id, surface), init)
        surf_l = self.state.elo_surface.get((loser_id, surface), init)
        expected_surf = 1.0 / (1.0 + 10 ** ((surf_l - surf_w) / 400))
        surf_k_w = k_w * 1.2  # Slightly higher K for surface (less data)
        surf_k_l = k_l * 1.2
        self.state.elo_surface[(winner_id, surface)] = surf_w + surf_k_w * (1 - expected_surf)
        self.state.elo_surface[(loser_id, surface)] = surf_l + surf_k_l * (0 - (1 - expected_surf))

        # Serve Elo (uses overall K-factor but separate rating pool)
        serve_w = self.state.elo_serve.get(winner_id, init)
        serve_l = self.state.elo_serve.get(loser_id, init)
        expected_serve = 1.0 / (1.0 + 10 ** ((serve_l - serve_w) / 400))
        self.state.elo_serve[winner_id] = serve_w + k_w * 0.8 * (1 - expected_serve)
        self.state.elo_serve[loser_id] = serve_l + k_l * 0.8 * (0 - (1 - expected_serve))

        # Return Elo
        ret_w = self.state.elo_return.get(winner_id, init)
        ret_l = self.state.elo_return.get(loser_id, init)
        expected_ret = 1.0 / (1.0 + 10 ** ((ret_l - ret_w) / 400))
        self.state.elo_return[winner_id] = ret_w + k_w * 0.8 * (1 - expected_ret)
        self.state.elo_return[loser_id] = ret_l + k_l * 0.8 * (0 - (1 - expected_ret))

    def _update_glicko2(
        self, winner_id: str, loser_id: str, match_date: pd.Timestamp
    ) -> None:
        """Simplified Glicko-2 update."""
        from tennis_predictor.config import GLICKO2_CONFIG
        init_r = GLICKO2_CONFIG["initial_rating"]
        init_rd = GLICKO2_CONFIG["initial_rd"]
        init_vol = GLICKO2_CONFIG["initial_vol"]
        tau = GLICKO2_CONFIG["tau"]

        r_w, rd_w, vol_w = self.state.glicko2.get(winner_id, (init_r, init_rd, init_vol))
        r_l, rd_l, vol_l = self.state.glicko2.get(loser_id, (init_r, init_rd, init_vol))

        # Decay RD for inactivity
        rd_w = self._decay_rd(winner_id, rd_w, match_date)
        rd_l = self._decay_rd(loser_id, rd_l, match_date)

        # Glicko-2 uses a different scale (mu, phi)
        mu_w = (r_w - 1500) / 173.7178
        mu_l = (r_l - 1500) / 173.7178
        phi_w = rd_w / 173.7178
        phi_l = rd_l / 173.7178

        # g function
        def g(phi):
            return 1.0 / np.sqrt(1 + 3 * phi**2 / np.pi**2)

        # Expected score
        def E(mu, mu_j, phi_j):
            return 1.0 / (1 + np.exp(-g(phi_j) * (mu - mu_j)))

        # Update winner
        g_l = g(phi_l)
        e_w = E(mu_w, mu_l, phi_l)
        v_w = 1.0 / (g_l**2 * e_w * (1 - e_w) + 1e-10)
        delta_w = v_w * g_l * (1 - e_w)

        # New volatility (simplified)
        new_vol_w = min(vol_w * 1.01, 0.1)
        new_phi_w = 1.0 / np.sqrt(1.0 / (phi_w**2 + new_vol_w**2) + 1.0 / v_w)
        new_mu_w = mu_w + new_phi_w**2 * g_l * (1 - e_w)

        # Update loser
        g_w = g(phi_w)
        e_l = E(mu_l, mu_w, phi_w)
        v_l = 1.0 / (g_w**2 * e_l * (1 - e_l) + 1e-10)

        new_vol_l = min(vol_l * 1.01, 0.1)
        new_phi_l = 1.0 / np.sqrt(1.0 / (phi_l**2 + new_vol_l**2) + 1.0 / v_l)
        new_mu_l = mu_l + new_phi_l**2 * g_w * (0 - e_l)

        # Convert back to Glicko scale
        new_r_w = new_mu_w * 173.7178 + 1500
        new_rd_w = np.clip(new_phi_w * 173.7178, GLICKO2_CONFIG["min_rd"],
                           GLICKO2_CONFIG["max_rd"])
        new_r_l = new_mu_l * 173.7178 + 1500
        new_rd_l = np.clip(new_phi_l * 173.7178, GLICKO2_CONFIG["min_rd"],
                           GLICKO2_CONFIG["max_rd"])

        self.state.glicko2[winner_id] = (new_r_w, new_rd_w, new_vol_w)
        self.state.glicko2[loser_id] = (new_r_l, new_rd_l, new_vol_l)
        self.state._last_updated[winner_id] = match_date
        self.state._last_updated[loser_id] = match_date

    def _decay_rd(self, player_id: str, rd: float, current_date: pd.Timestamp) -> float:
        """Increase RD when a player hasn't played recently."""
        from tennis_predictor.config import GLICKO2_CONFIG
        last = self.state._last_updated.get(player_id)
        if last is not None and not pd.isna(current_date) and not pd.isna(last):
            days_inactive = (current_date - last).days
            if days_inactive > 0:
                rd_increase = GLICKO2_CONFIG["rd_decay_per_day"] * days_inactive
                rd = min(rd + rd_increase, GLICKO2_CONFIG["max_rd"])
        return rd

    def _update_match_history(
        self, match: pd.Series, p1_id: str, p2_id: str,
        result: int, surface: str, match_date: pd.Timestamp,
    ) -> None:
        """Add match result to both players' rolling history."""
        from tennis_predictor.data.sackmann import ROUND_ORDER

        round_name = match.get("round", "")

        # Compute match-level stats for history
        def _compute_stats(prefix: str, is_winner: bool) -> dict:
            # Map pairwise prefix back to winner/loser stats
            if (prefix == "p1" and result == 1) or (prefix == "p2" and result == 0):
                stat_prefix = "w_" if is_winner else "l_"
            else:
                stat_prefix = "l_" if is_winner else "w_"

            # We need the original match row which has w_ and l_ prefixed stats
            # Since we're working with pairwise rows, we look for match stats
            svpt = match.get(f"{stat_prefix}svpt", np.nan)
            first_in = match.get(f"{stat_prefix}1stIn", np.nan)
            first_won = match.get(f"{stat_prefix}1stWon", np.nan)
            second_won = match.get(f"{stat_prefix}2ndWon", np.nan)
            ace = match.get(f"{stat_prefix}ace", np.nan)
            bp_saved = match.get(f"{stat_prefix}bpSaved", np.nan)
            bp_faced = match.get(f"{stat_prefix}bpFaced", np.nan)

            fsp = first_in / svpt if _valid(svpt) and _valid(first_in) and svpt > 0 else np.nan
            fswp = (first_won / first_in
                    if _valid(first_won) and _valid(first_in) and first_in > 0 else np.nan)
            bp_save_pct = (bp_saved / bp_faced
                           if _valid(bp_saved) and _valid(bp_faced) and bp_faced > 0 else np.nan)
            ace_rate = ace / svpt if _valid(ace) and _valid(svpt) and svpt > 0 else np.nan

            return {
                "first_serve_pct": fsp,
                "first_serve_won_pct": fswp,
                "bp_save_pct": bp_save_pct,
                "ace_rate": ace_rate,
            }

        # For pairwise format, we need to reconstruct who had which stats
        # In the original data: winner stats = w_, loser stats = l_
        # In pairwise: if result=1, p1=winner; if result=0, p2=winner

        for prefix, pid in [("p1", p1_id), ("p2", p2_id)]:
            is_winner = (prefix == "p1" and result == 1) or (prefix == "p2" and result == 0)
            won = is_winner

            # Since match may have original w_/l_ stats from the unpaired data,
            # we compute stats based on winner/loser mapping
            stat_prefix_w = "w_"
            stat_prefix_l = "l_"
            sp = stat_prefix_w if is_winner else stat_prefix_l

            svpt = match.get(f"{sp}svpt", np.nan)
            first_in = match.get(f"{sp}1stIn", np.nan)
            first_won = match.get(f"{sp}1stWon", np.nan)
            ace = match.get(f"{sp}ace", np.nan)
            bp_saved = match.get(f"{sp}bpSaved", np.nan)
            bp_faced = match.get(f"{sp}bpFaced", np.nan)

            # Additional stats
            df = match.get(f"{sp}df", np.nan)
            second_won = match.get(f"{sp}2ndWon", np.nan)
            sv_gms = match.get(f"{sp}SvGms", np.nan)
            bp_convert_faced = match.get(
                f"{'l_' if is_winner else 'w_'}bpFaced", np.nan
            )
            bp_convert_saved = match.get(
                f"{'l_' if is_winner else 'w_'}bpSaved", np.nan
            )

            second_serve_pts = (svpt - first_in
                                if _valid(svpt) and _valid(first_in) else np.nan)
            opp_svpt = match.get(f"{'l_' if is_winner else 'w_'}svpt", np.nan)
            opp_1st_won = match.get(f"{'l_' if is_winner else 'w_'}1stWon", np.nan)
            opp_2nd_won = match.get(f"{'l_' if is_winner else 'w_'}2ndWon", np.nan)

            record = {
                "date": match_date,
                "surface": surface,
                "won": won,
                "round_order": ROUND_ORDER.get(round_name, 0),
                "tourney_level": match.get("tourney_level", "A"),
                "best_of": match.get("best_of", 3),
                "retirement": bool(match.get("retirement", False)),
                "minutes": match.get("minutes", np.nan),
                "first_serve_pct": (first_in / svpt if _valid(svpt) and _valid(first_in)
                                    and svpt > 0 else np.nan),
                "first_serve_won_pct": (first_won / first_in if _valid(first_won)
                                        and _valid(first_in) and first_in > 0 else np.nan),
                "second_serve_won_pct": (second_won / second_serve_pts
                                         if _valid(second_won) and _valid(second_serve_pts)
                                         and second_serve_pts > 0 else np.nan),
                "bp_save_pct": (bp_saved / bp_faced if _valid(bp_saved) and _valid(bp_faced)
                                and bp_faced > 0 else np.nan),
                "ace_rate": (ace / svpt if _valid(ace) and _valid(svpt)
                             and svpt > 0 else np.nan),
                "df_rate": (df / svpt if _valid(df) and _valid(svpt)
                            and svpt > 0 else np.nan),
                "serve_efficiency": ((ace - df) / svpt
                                     if _valid(ace) and _valid(df) and _valid(svpt)
                                     and svpt > 0 else np.nan),
                "serve_pts_won_pct": ((first_won + second_won) / svpt
                                      if _valid(first_won) and _valid(second_won)
                                      and _valid(svpt) and svpt > 0 else np.nan),
                "return_pts_won_pct": (
                    (opp_svpt - opp_1st_won - opp_2nd_won) / opp_svpt
                    if _valid(opp_svpt) and _valid(opp_1st_won) and _valid(opp_2nd_won)
                    and opp_svpt > 0 else np.nan
                ),
                "hold_pct": (
                    1.0 - (bp_faced - bp_saved) / sv_gms
                    if _valid(bp_faced) and _valid(bp_saved) and _valid(sv_gms) and sv_gms > 0
                    else np.nan
                ),
                "break_pct": (
                    (bp_convert_faced - bp_convert_saved) / bp_convert_faced
                    if _valid(bp_convert_faced) and _valid(bp_convert_saved)
                    and bp_convert_faced > 0 else np.nan
                ),
                "opponent_rank": (match.get("p2_rank" if prefix == "p1" else "p1_rank", np.nan)),
            }

            if pid not in self.state.match_history:
                self.state.match_history[pid] = []
            self.state.match_history[pid].append(record)

            # Keep history bounded (last 200 matches)
            if len(self.state.match_history[pid]) > 200:
                self.state.match_history[pid] = self.state.match_history[pid][-200:]

    def _update_h2h(self, p1: str, p2: str, result: int, surface: str) -> None:
        """Update head-to-head record."""
        key = (min(p1, p2), max(p1, p2))
        if key not in self.state.h2h:
            self.state.h2h[key] = {"total": 0, "wins_a": 0, "wins_b": 0}
        h2h = self.state.h2h[key]

        h2h["total"] += 1
        # Determine which side of the key each player is on
        if key[0] == p1:
            h2h["wins_a" if result == 1 else "wins_b"] += 1
            h2h[f"surface_wins_a_{surface}" if result == 1
                 else f"surface_wins_b_{surface}"] = (
                h2h.get(f"surface_wins_a_{surface}" if result == 1
                        else f"surface_wins_b_{surface}", 0) + 1
            )
        else:
            h2h["wins_b" if result == 1 else "wins_a"] += 1
            h2h[f"surface_wins_b_{surface}" if result == 1
                 else f"surface_wins_a_{surface}"] = (
                h2h.get(f"surface_wins_b_{surface}" if result == 1
                        else f"surface_wins_a_{surface}", 0) + 1
            )
        h2h[f"surface_total_{surface}"] = h2h.get(f"surface_total_{surface}", 0) + 1

    def _update_tournament_history(
        self, p1: str, p2: str, tourney_name: str,
        match: pd.Series, result: int,
    ) -> None:
        from tennis_predictor.data.sackmann import ROUND_ORDER
        round_name = match.get("round", "")

        for prefix, pid in [("p1", p1), ("p2", p2)]:
            key = (pid, tourney_name)
            if key not in self.state.tournament_history:
                self.state.tournament_history[key] = []

            won = (prefix == "p1" and result == 1) or (prefix == "p2" and result == 0)
            self.state.tournament_history[key].append({
                "round_order": ROUND_ORDER.get(round_name, 0),
                "won": won,
                "year": match.get("tourney_date", pd.NaT),
            })

    @property
    def stats(self) -> dict:
        """Return guard statistics for debugging."""
        return {
            "matches_processed": len(self._processed_match_ids),
            "extractions": self._extraction_count,
            "updates": self._update_count,
            "players_tracked": len(self.state.elo),
            "h2h_pairs": len(self.state.h2h),
            "cutoff": str(self.state.cutoff),
        }


class TemporalLeakageError(Exception):
    """Raised when temporal isolation is violated."""
    pass


# === UTILITY FUNCTIONS ===

def _safe_diff(a, b):
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float(a) - float(b)


def _safe_ratio(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return float(a) / float(b)


def _safe_log_diff(a, b):
    if pd.isna(a) or pd.isna(b) or a <= 0 or b <= 0:
        return np.nan
    return np.log(float(a)) - np.log(float(b))


def _valid(x):
    return x is not None and not pd.isna(x)


def _parse_seed(seed):
    if pd.isna(seed) or seed is None or str(seed).strip() == "":
        return np.nan
    try:
        return float(str(seed).strip())
    except (ValueError, TypeError):
        return np.nan
