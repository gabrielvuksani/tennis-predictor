"""Tests for feature engineering.

Tests the pairwise row creation from Sackmann data and verifies that
advanced feature extraction produces the expected number of feature columns.
"""

import numpy as np
import pandas as pd
import pytest


def _make_matches_df(n: int = 20) -> pd.DataFrame:
    """Create a small synthetic matches DataFrame in winner/loser format."""
    rng = np.random.RandomState(42)
    rows = []
    for i in range(n):
        rows.append({
            "match_id": f"T{i // 4}_{i}",
            "tourney_id": f"T{i // 4}",
            "tourney_name": "Test Open",
            "tourney_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
            "tourney_level": "A",
            "surface": rng.choice(["Hard", "Clay", "Grass"]),
            "round": rng.choice(["R32", "R16", "QF", "SF", "F"]),
            "best_of": 3,
            "draw_size": 32,
            "minutes": rng.randint(60, 180),
            "retirement": False,
            "n_sets": rng.choice([2, 3]),
            "winner_id": str(rng.choice([100, 101, 102, 103, 104])),
            "winner_name": "Player A",
            "winner_hand": "R",
            "winner_ht": 185,
            "winner_ioc": "USA",
            "winner_age": 25.0,
            "winner_rank": rng.randint(1, 50),
            "winner_rank_points": rng.randint(500, 5000),
            "winner_seed": "",
            "winner_entry": "",
            "loser_id": str(rng.choice([200, 201, 202, 203, 204])),
            "loser_name": "Player B",
            "loser_hand": "R",
            "loser_ht": 180,
            "loser_ioc": "GBR",
            "loser_age": 28.0,
            "loser_rank": rng.randint(50, 200),
            "loser_rank_points": rng.randint(100, 1000),
            "loser_seed": "",
            "loser_entry": "",
            "score": "6-4 6-3",
        })
    return pd.DataFrame(rows)


class TestCreatePairwiseRows:
    """Test the winner/loser to p1/p2 conversion."""

    def test_output_shape(self):
        from tennis_predictor.data.sackmann import create_pairwise_rows

        matches = _make_matches_df(20)
        pairwise = create_pairwise_rows(matches)

        assert len(pairwise) == len(matches), "Should produce one row per match"

    def test_target_values(self):
        from tennis_predictor.data.sackmann import create_pairwise_rows

        matches = _make_matches_df(50)
        pairwise = create_pairwise_rows(matches)

        # Target should only be 0 or 1
        assert set(pairwise["y"].unique()).issubset({0, 1})

    def test_deterministic_with_seed(self):
        """create_pairwise_rows uses a fixed seed, so results must be identical across calls."""
        from tennis_predictor.data.sackmann import create_pairwise_rows

        matches = _make_matches_df(30)
        pw1 = create_pairwise_rows(matches)
        pw2 = create_pairwise_rows(matches)

        pd.testing.assert_frame_equal(pw1, pw2)

    def test_p1_p2_ids_assigned(self):
        from tennis_predictor.data.sackmann import create_pairwise_rows

        matches = _make_matches_df(10)
        pairwise = create_pairwise_rows(matches)

        assert "p1_id" in pairwise.columns
        assert "p2_id" in pairwise.columns
        # Every row should have p1 and p2 IDs
        assert pairwise["p1_id"].notna().all()
        assert pairwise["p2_id"].notna().all()

    def test_swap_consistency(self):
        """When y=1, p1 should be the winner; when y=0, p1 should be the loser."""
        from tennis_predictor.data.sackmann import create_pairwise_rows

        matches = _make_matches_df(50)
        pairwise = create_pairwise_rows(matches)

        for idx in pairwise.index:
            pw_row = pairwise.loc[idx]
            orig_row = matches.loc[idx]
            if pw_row["y"] == 1:
                assert str(pw_row["p1_id"]) == str(orig_row["winner_id"])
                assert str(pw_row["p2_id"]) == str(orig_row["loser_id"])
            else:
                assert str(pw_row["p1_id"]) == str(orig_row["loser_id"])
                assert str(pw_row["p2_id"]) == str(orig_row["winner_id"])

    def test_roughly_balanced_target(self):
        """With random swapping, target should be roughly 50/50 over many samples."""
        from tennis_predictor.data.sackmann import create_pairwise_rows

        matches = _make_matches_df(200)
        pairwise = create_pairwise_rows(matches)

        y_mean = pairwise["y"].mean()
        assert 0.3 < y_mean < 0.7, f"Target mean {y_mean} is too skewed"

    def test_metadata_preserved(self):
        from tennis_predictor.data.sackmann import create_pairwise_rows

        matches = _make_matches_df(10)
        pairwise = create_pairwise_rows(matches)

        assert "surface" in pairwise.columns
        assert "tourney_date" in pairwise.columns
        assert "round" in pairwise.columns
        assert "best_of" in pairwise.columns


class TestAdvancedFeatures:
    """Test that advanced feature extraction produces features correctly."""

    def test_handedness_features(self):
        from tennis_predictor.features.advanced import _handedness_features

        match = pd.Series({
            "p1_hand": "L",
            "p2_hand": "R",
        })
        features = _handedness_features(match)

        assert features["p1_is_lefty"] == 1
        assert features["p2_is_lefty"] == 0
        assert features["lefty_vs_righty"] == 1
        assert features["both_lefty"] == 0

    def test_handedness_both_right(self):
        from tennis_predictor.features.advanced import _handedness_features

        match = pd.Series({
            "p1_hand": "R",
            "p2_hand": "R",
        })
        features = _handedness_features(match)

        assert features["p1_is_lefty"] == 0
        assert features["p2_is_lefty"] == 0
        assert features["lefty_vs_righty"] == 0
        assert features["both_lefty"] == 0

    def test_home_advantage_us_open(self):
        from tennis_predictor.features.advanced import _home_advantage_features

        match = pd.Series({
            "tourney_name": "US Open",
            "p1_ioc": "USA",
            "p2_ioc": "GBR",
        })
        features = _home_advantage_features(match)

        assert features["p1_home_advantage"] == 1
        assert features["p2_home_advantage"] == 0
        assert features["home_diff"] == 1

    def test_home_advantage_neutral(self):
        from tennis_predictor.features.advanced import _home_advantage_features

        match = pd.Series({
            "tourney_name": "Dubai Open",
            "p1_ioc": "USA",
            "p2_ioc": "GBR",
        })
        features = _home_advantage_features(match)

        assert features["p1_home_advantage"] == 0
        assert features["p2_home_advantage"] == 0
        assert features["home_diff"] == 0

    def test_age_trajectory_features(self):
        from tennis_predictor.features.advanced import _age_trajectory_features

        match = pd.Series({
            "p1_age": 23.0,
            "p2_age": 30.0,
        })
        features = _age_trajectory_features(match)

        assert features["p1_pre_peak"] == 1    # 23 < 25.5
        assert features["p1_post_peak"] == 0   # 23 < 28
        assert features["p2_pre_peak"] == 0    # 30 > 25.5
        assert features["p2_post_peak"] == 1   # 30 > 28
        assert features["p1_dist_from_peak"] == pytest.approx(2.5)
        assert features["p2_dist_from_peak"] == pytest.approx(4.5)

    def test_age_trajectory_missing(self):
        from tennis_predictor.features.advanced import _age_trajectory_features

        match = pd.Series({
            "p1_age": np.nan,
            "p2_age": 25.0,
        })
        features = _age_trajectory_features(match)

        assert np.isnan(features["p1_dist_from_peak"])
        assert features["p2_dist_from_peak"] == pytest.approx(0.5)
