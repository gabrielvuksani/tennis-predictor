"""Tests for TemporalGuard — ensures zero data leakage.

This is the most important test file in the project. The phosphenq model
failed because ELO leakage inflated accuracy. These tests ensure that
can never happen in our system.
"""

import numpy as np
import pandas as pd
import pytest

from tennis_predictor.temporal.guard import TemporalGuard, TemporalLeakageError


def _make_match(
    match_id: str,
    p1_id: str = "100",
    p2_id: str = "200",
    date: str = "2020-01-01",
    surface: str = "Hard",
    y: int = 1,
    **kwargs,
) -> pd.Series:
    """Create a fake match row for testing."""
    data = {
        "match_id": match_id,
        "p1_id": p1_id,
        "p2_id": p2_id,
        "tourney_date": pd.Timestamp(date),
        "tourney_name": "Test Open",
        "tourney_level": "A",
        "surface": surface,
        "round": "R32",
        "best_of": 3,
        "draw_size": 32,
        "p1_rank": 10,
        "p2_rank": 50,
        "p1_age": 25.0,
        "p2_age": 28.0,
        "p1_ht": 185,
        "p2_ht": 180,
        "p1_entry": "",
        "p2_entry": "",
        "p1_seed": "1",
        "p2_seed": "",
        "p1_hand": "R",
        "p2_hand": "R",
        "p1_ioc": "USA",
        "p2_ioc": "GBR",
        "minutes": 90,
        "retirement": False,
        "n_sets": 3,
        "y": y,
        "w_ace": 10,
        "w_df": 3,
        "w_svpt": 80,
        "w_1stIn": 50,
        "w_1stWon": 35,
        "w_2ndWon": 15,
        "w_SvGms": 12,
        "w_bpSaved": 4,
        "w_bpFaced": 6,
        "l_ace": 5,
        "l_df": 5,
        "l_svpt": 75,
        "l_1stIn": 45,
        "l_1stWon": 28,
        "l_2ndWon": 12,
        "l_SvGms": 12,
        "l_bpSaved": 2,
        "l_bpFaced": 5,
    }
    data.update(kwargs)
    return pd.Series(data)


class TestTemporalGuardOrdering:
    """Test that extract-then-update ordering is enforced."""

    def test_extract_then_update_works(self):
        guard = TemporalGuard()
        match = _make_match("m1")
        features = guard.extract_pre_match_state(match)
        guard.update_state(match, 1)
        assert "elo_diff" in features

    def test_update_without_extract_raises(self):
        guard = TemporalGuard()
        match = _make_match("m1")
        with pytest.raises(TemporalLeakageError):
            guard.update_state(match, 1)

    def test_duplicate_match_raises(self):
        guard = TemporalGuard()
        match = _make_match("m1")
        guard.extract_pre_match_state(match)
        guard.update_state(match, 1)
        with pytest.raises(TemporalLeakageError):
            guard.extract_pre_match_state(match)


class TestEloLeakagePrevention:
    """Test that Elo values used for features are ALWAYS pre-match."""

    def test_elo_not_updated_before_extraction(self):
        """The critical test: Elo must reflect PRE-match state.

        This is the exact bug that killed the phosphenq model.
        """
        guard = TemporalGuard()

        # Match 1: Player 100 beats Player 200
        m1 = _make_match("m1", p1_id="100", p2_id="200", date="2020-01-01")
        f1 = guard.extract_pre_match_state(m1)
        guard.update_state(m1, 1)  # Player 100 wins

        # After match 1, Player 100's Elo should have gone UP
        elo_100_after_m1 = guard.state.elo["100"]
        elo_200_after_m1 = guard.state.elo["200"]
        assert elo_100_after_m1 > 1500  # Won, so Elo went up
        assert elo_200_after_m1 < 1500  # Lost, so Elo went down

        # Match 2: Player 100 vs Player 300
        m2 = _make_match("m2", p1_id="100", p2_id="300", date="2020-01-02")
        f2 = guard.extract_pre_match_state(m2)

        # Player 100's Elo in features should be the POST-m1 value
        # (which is the PRE-m2 value — this is correct)
        assert f2["elo_p1"] == elo_100_after_m1

        # Player 300 is new, should have initial Elo
        assert f2["elo_p2"] == 1500.0

    def test_surface_elo_respects_temporal_boundary(self):
        guard = TemporalGuard()

        # Match on Clay
        m1 = _make_match("m1", surface="Clay", date="2020-01-01")
        f1 = guard.extract_pre_match_state(m1)
        # Before any matches, surface Elo should be initial
        assert f1["surface_elo_p1"] == 1500.0
        guard.update_state(m1, 1)

        # Next match on Hard - surface Elo should still be initial for Hard
        m2 = _make_match("m2", surface="Hard", date="2020-01-02")
        f2 = guard.extract_pre_match_state(m2)
        assert f2["surface_elo_p1"] == 1500.0  # No Hard matches yet
        guard.update_state(m2, 1)

        # Another Clay match - should reflect updated Clay Elo
        m3 = _make_match("m3", surface="Clay", date="2020-01-03")
        f3 = guard.extract_pre_match_state(m3)
        assert f3["surface_elo_p1"] > 1500.0  # Won on Clay before


class TestRollingStatsIntegrity:
    """Test that rolling statistics use only pre-match data."""

    def test_rolling_winrate_excludes_current_match(self):
        guard = TemporalGuard()

        # Player 100 wins 3 matches
        for i in range(3):
            m = _make_match(f"m{i}", date=f"2020-01-0{i+1}")
            guard.extract_pre_match_state(m)
            guard.update_state(m, 1)

        # 4th match: features should show 3 wins in history
        m4 = _make_match("m4", date="2020-01-05")
        f4 = guard.extract_pre_match_state(m4)

        # Win rate over 5 matches should be 1.0 (3 wins out of 3)
        assert f4["p1_winrate_5"] == 1.0

    def test_h2h_excludes_current_match(self):
        guard = TemporalGuard()

        # First meeting
        m1 = _make_match("m1", p1_id="100", p2_id="200", date="2020-01-01")
        f1 = guard.extract_pre_match_state(m1)
        assert f1["h2h_total_matches"] == 0  # No prior meetings
        guard.update_state(m1, 1)  # 100 wins

        # Second meeting
        m2 = _make_match("m2", p1_id="100", p2_id="200", date="2020-02-01")
        f2 = guard.extract_pre_match_state(m2)
        assert f2["h2h_total_matches"] == 1  # One prior meeting
        assert f2["h2h_p1_win_pct"] == 1.0   # 100 won the previous one


class TestFeatureCompleteness:
    """Test that all expected features are generated."""

    def test_all_feature_categories_present(self):
        guard = TemporalGuard()
        match = _make_match("m1")
        features = guard.extract_pre_match_state(match)

        # Elo features
        assert "elo_diff" in features
        assert "surface_elo_diff" in features
        assert "elo_win_prob" in features
        assert "serve_elo_diff" in features
        assert "return_elo_diff" in features

        # Glicko-2 features
        assert "glicko2_diff" in features
        assert "glicko2_rd_p1" in features

        # Rolling features
        assert "p1_winrate_5" in features
        assert "p1_winrate_10" in features

        # H2H features
        assert "h2h_total_matches" in features
        assert "h2h_p1_win_pct" in features

        # Fatigue features
        assert "p1_days_since_last" in features
        assert "p1_matches_last_7d" in features

        # Tournament features
        assert "p1_tourney_appearances" in features

        # Context features
        assert "surface_Hard" in features
        assert "rank_diff" in features
        assert "age_diff" in features
        assert "best_of_5" in features

    def test_feature_count(self):
        guard = TemporalGuard()
        match = _make_match("m1")
        features = guard.extract_pre_match_state(match)
        # Should have a substantial number of features
        assert len(features) >= 80, f"Only {len(features)} features generated"


class TestGuardStats:
    """Test guard tracking and statistics."""

    def test_stats_tracking(self):
        guard = TemporalGuard()

        for i in range(5):
            m = _make_match(f"m{i}", date=f"2020-01-0{i+1}")
            guard.extract_pre_match_state(m)
            guard.update_state(m, 1)

        stats = guard.stats
        assert stats["matches_processed"] == 5
        assert stats["extractions"] == 5
        assert stats["updates"] == 5
