"""Definitive odds merge — combines multiple matching strategies.

Strategy 1: Exact last-name + initial match + date window (highest precision)
Strategy 2: Last-name-only + surface + date window (medium precision)
Strategy 3: Rank proximity + surface + date window (fallback)

This replaces the broken _merge_odds() in pipeline.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tennis_predictor.data.odds import download_tennis_data_odds


def merge_odds_with_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Merge betting odds into match data using multi-strategy matching.

    Returns matches DataFrame with added odds columns.
    """
    min_year = int(matches["tourney_date"].dt.year.min())
    max_year = int(matches["tourney_date"].dt.year.max())
    odds_start = max(min_year, 2001)

    odds_df = download_tennis_data_odds(start_year=odds_start, end_year=max_year)

    if len(odds_df) == 0:
        print("Warning: No odds data available.")
        for col in ["odds_implied_w", "odds_implied_l", "odds_pinnacle_w", "odds_pinnacle_l"]:
            matches[col] = np.nan
        return matches

    # Prepare odds data
    odds_df = _prepare_odds(odds_df)

    # Initialize result columns
    matches["odds_implied_w"] = np.nan
    matches["odds_implied_l"] = np.nan
    matches["odds_pinnacle_w"] = np.nan
    matches["odds_pinnacle_l"] = np.nan

    # Pre-compute match-side keys once (used across strategies)
    matches["_m_w_last"] = matches["winner_name"].map(_sackmann_lastname)
    matches["_m_l_last"] = matches["loser_name"].map(_sackmann_lastname)
    matches["_m_surf"] = matches["surface"].astype(str).str.lower().str.strip()
    matches["_m_date_ord"] = matches["tourney_date"].values.astype("datetime64[D]").astype(np.int64)

    # Pre-compute odds-side date ordinal for vectorized date-window checks
    odds_df["_date_ord"] = odds_df["_date"].values.astype("datetime64[D]").astype(np.int64)

    # Strategy 1: Last name + initial matching (most precise)
    matched_1 = _match_by_name(matches, odds_df)
    print(f"Odds merge pass 1 (name+initial): {matched_1:,} matched")

    # Strategy 2: Last name only + surface (medium precision)
    matched_2 = _match_by_lastname(matches, odds_df)
    print(f"Odds merge pass 2 (lastname+surface): {matched_2:,} matched")

    # Strategy 3: Rank proximity (fallback)
    matched_3 = _match_by_rank(matches, odds_df)
    print(f"Odds merge pass 3 (rank proximity): {matched_3:,} matched")

    total = matches["odds_implied_w"].notna().sum()
    print(f"Total matches with odds: {total:,} / {len(matches):,} ({total/len(matches):.1%})")

    # Clean up temporary columns
    matches.drop(columns=["_m_w_last", "_m_l_last", "_m_surf", "_m_date_ord"], inplace=True)

    return matches


def _prepare_odds(odds_df: pd.DataFrame) -> pd.DataFrame:
    """Parse and compute odds fields."""
    if "Date" in odds_df.columns:
        odds_df["_date"] = pd.to_datetime(odds_df["Date"], dayfirst=True, errors="coerce")
    else:
        odds_df["_date"] = pd.NaT

    # Use implied probabilities — may already be computed by download_tennis_data_odds
    odds_df["_ip_w"] = np.nan
    odds_df["_ip_l"] = np.nan
    odds_df["_ow"] = np.nan
    odds_df["_ol"] = np.nan

    # Try pre-computed columns first (from _clean_odds in odds.py)
    if "implied_prob_w" in odds_df.columns:
        odds_df["_ip_w"] = pd.to_numeric(odds_df["implied_prob_w"], errors="coerce")
        odds_df["_ip_l"] = pd.to_numeric(odds_df["implied_prob_l"], errors="coerce")
    if "odds_pinnacle_w" in odds_df.columns:
        odds_df["_ow"] = pd.to_numeric(odds_df["odds_pinnacle_w"], errors="coerce")
        odds_df["_ol"] = pd.to_numeric(odds_df["odds_pinnacle_l"], errors="coerce")
    elif "odds_b365_w" in odds_df.columns:
        odds_df["_ow"] = pd.to_numeric(odds_df["odds_b365_w"], errors="coerce")
        odds_df["_ol"] = pd.to_numeric(odds_df["odds_b365_l"], errors="coerce")
    elif "odds_avg_w" in odds_df.columns:
        odds_df["_ow"] = pd.to_numeric(odds_df["odds_avg_w"], errors="coerce")
        odds_df["_ol"] = pd.to_numeric(odds_df["odds_avg_l"], errors="coerce")

    # Fallback: compute from raw columns if not pre-computed
    if odds_df["_ip_w"].isna().all():
        for wc, lc in [("PSW", "PSL"), ("B365W", "B365L"), ("AvgW", "AvgL")]:
            if wc in odds_df.columns:
                ow = pd.to_numeric(odds_df[wc], errors="coerce")
                ol = pd.to_numeric(odds_df[lc], errors="coerce")
                total = 1.0 / ow + 1.0 / ol
                mask = odds_df["_ip_w"].isna() & ow.notna()
                odds_df.loc[mask, "_ip_w"] = (1.0 / ow / total)[mask]
                odds_df.loc[mask, "_ip_l"] = (1.0 / ol / total)[mask]
                odds_df.loc[mask, "_ow"] = ow[mask]
                odds_df.loc[mask, "_ol"] = ol[mask]

    # Parse names and ranks
    if "Winner" in odds_df.columns:
        odds_df["_w_last"] = odds_df["Winner"].map(_extract_lastname)
        odds_df["_l_last"] = odds_df["Loser"].map(_extract_lastname)
    if "WRank" in odds_df.columns:
        odds_df["_wr"] = pd.to_numeric(odds_df["WRank"], errors="coerce")
        odds_df["_lr"] = pd.to_numeric(odds_df["LRank"], errors="coerce")
    if "Surface" in odds_df.columns:
        odds_df["_surf"] = odds_df["Surface"].str.lower().str.strip()

    return odds_df.dropna(subset=["_date", "_ip_w"])


def _extract_lastname(name: str) -> str:
    """Extract last name from any format."""
    if pd.isna(name) or not name:
        return ""
    name = str(name).strip().lower()
    # "Last F." format -> just take first word(s) before the initial
    parts = name.split()
    if parts and parts[-1].endswith(".") and len(parts[-1]) <= 3:
        return " ".join(parts[:-1])
    # "First Last" -> take last word
    if len(parts) >= 2:
        return parts[-1]
    return name


def _sackmann_lastname(name) -> str:
    """Extract last name from Sackmann 'First Last' format."""
    if pd.isna(name) or not name:
        return ""
    parts = str(name).strip().lower().split()
    return parts[-1] if parts else ""


def _assign_odds_vectorized(
    matches: pd.DataFrame, match_idx: np.ndarray, odds_df: pd.DataFrame, odds_idx: np.ndarray
) -> None:
    """Bulk-assign odds from matched odds rows into matches DataFrame."""
    if len(match_idx) == 0:
        return
    matches.loc[match_idx, "odds_implied_w"] = odds_df.loc[odds_idx, "_ip_w"].values
    matches.loc[match_idx, "odds_implied_l"] = odds_df.loc[odds_idx, "_ip_l"].values
    matches.loc[match_idx, "odds_pinnacle_w"] = odds_df.loc[odds_idx, "_ow"].values
    matches.loc[match_idx, "odds_pinnacle_l"] = odds_df.loc[odds_idx, "_ol"].values


def _match_by_name(matches: pd.DataFrame, odds_df: pd.DataFrame) -> int:
    """Strategy 1: Match by last name + date window (vectorized).

    Uses pd.merge on (w_last, l_last) then filters by date window [-1, +16] days.
    First match (closest odds date) wins for each match row.
    """
    if "_w_last" not in odds_df.columns:
        return 0

    unmatched_mask = matches["odds_implied_w"].isna()
    m_sub = matches.loc[unmatched_mask].copy()

    # Filter out rows with empty names or missing dates
    valid = (m_sub["_m_w_last"] != "") & (m_sub["_m_l_last"] != "") & m_sub["tourney_date"].notna()
    m_sub = m_sub.loc[valid]
    if len(m_sub) == 0:
        return 0

    # Merge on exact (w_last, l_last), preserving original indices for assignment
    m_sub_keyed = m_sub[["_m_w_last", "_m_l_last", "_m_date_ord"]].copy()
    m_sub_keyed["_midx"] = m_sub_keyed.index

    odds_keyed = odds_df[["_w_last", "_l_last", "_date_ord", "_ip_w", "_ip_l", "_ow", "_ol"]].copy()
    odds_keyed.rename(columns={"_w_last": "_m_w_last", "_l_last": "_m_l_last"}, inplace=True)
    odds_keyed["_oidx"] = odds_keyed.index

    merged = m_sub_keyed.merge(odds_keyed, on=["_m_w_last", "_m_l_last"], how="inner")
    if len(merged) == 0:
        return 0

    diff = merged["_date_ord"] - merged["_m_date_ord"]
    merged = merged.loc[(diff >= -1) & (diff <= 16)].copy()
    if len(merged) == 0:
        return 0

    merged["_abs_diff"] = (merged["_date_ord"] - merged["_m_date_ord"]).abs()
    merged.sort_values("_abs_diff", inplace=True)
    best = merged.drop_duplicates(subset=["_midx"], keep="first")

    _assign_odds_vectorized(matches, best["_midx"].values, odds_df, best["_oidx"].values)
    return len(best)


def _match_by_lastname(matches: pd.DataFrame, odds_df: pd.DataFrame) -> int:
    """Strategy 2: Last name substring + surface + date window (vectorized).

    The original strategy checked `w_last in odds_w_last` (substring containment).
    To replicate this vectorized: we merge on surface + date bucket to limit the
    cross-product size, filter by exact date window, then check substring containment.
    """
    if "_surf" not in odds_df.columns:
        return 0

    unmatched_mask = matches["odds_implied_w"].isna()
    m_sub = matches.loc[unmatched_mask].copy()

    valid = (m_sub["_m_w_last"] != "") & (m_sub["_m_l_last"] != "") & m_sub["tourney_date"].notna()
    m_sub = m_sub.loc[valid]
    if len(m_sub) == 0:
        return 0

    # Use 21-day date buckets to limit cross-product size.
    # A match with date D can match odds in [D, D+16], so we need to join
    # across at most 2 adjacent buckets. We do this by duplicating matches
    # into their bucket and the next bucket.
    bucket_days = 21

    m_keyed = m_sub[["_m_w_last", "_m_l_last", "_m_surf", "_m_date_ord"]].copy()
    m_keyed["_midx"] = m_keyed.index
    m_keyed["_bucket"] = m_keyed["_m_date_ord"] // bucket_days

    # Duplicate match rows into the next bucket so matches near bucket boundaries
    # can find odds in the adjacent bucket
    m_keyed2 = m_keyed.copy()
    m_keyed2["_bucket"] = m_keyed2["_bucket"] + 1
    m_keyed_all = pd.concat([m_keyed, m_keyed2], ignore_index=True)

    o_keyed = odds_df[["_w_last", "_l_last", "_surf", "_date_ord", "_ip_w", "_ip_l", "_ow", "_ol"]].copy()
    o_keyed.rename(columns={"_surf": "_m_surf"}, inplace=True)
    o_keyed["_oidx"] = o_keyed.index
    o_keyed["_bucket"] = o_keyed["_date_ord"] // bucket_days

    # Merge on surface + date bucket
    merged = m_keyed_all.merge(o_keyed, on=["_m_surf", "_bucket"], how="inner")
    if len(merged) == 0:
        return 0

    # Date window [0, +16] days (original used range(0, 17))
    diff = merged["_date_ord"] - merged["_m_date_ord"]
    merged = merged.loc[(diff >= 0) & (diff <= 16)].copy()
    if len(merged) == 0:
        return 0

    # Substring containment: match_w_last in odds_w_last AND match_l_last in odds_l_last
    w_contains = np.array([
        mw in ow
        for mw, ow in zip(merged["_m_w_last"].values, merged["_w_last"].values)
    ])
    l_contains = np.array([
        ml in ol
        for ml, ol in zip(merged["_m_l_last"].values, merged["_l_last"].values)
    ])
    merged = merged.loc[w_contains & l_contains].copy()
    if len(merged) == 0:
        return 0

    # Pick closest date match per match row
    merged["_abs_diff"] = (merged["_date_ord"] - merged["_m_date_ord"]).abs()
    merged.sort_values("_abs_diff", inplace=True)
    best = merged.drop_duplicates(subset=["_midx"], keep="first")

    _assign_odds_vectorized(matches, best["_midx"].values, odds_df, best["_oidx"].values)
    return len(best)


def _match_by_rank(matches: pd.DataFrame, odds_df: pd.DataFrame, tolerance: int = 5) -> int:
    """Strategy 3: Rank proximity + surface + date window (vectorized).

    Merges on surface + date bucket, filters by date window and rank proximity.
    """
    if "_wr" not in odds_df.columns or "_surf" not in odds_df.columns:
        return 0

    unmatched_mask = matches["odds_implied_w"].isna()
    m_sub = matches.loc[unmatched_mask].copy()

    valid = (
        m_sub["winner_rank"].notna()
        & m_sub["loser_rank"].notna()
        & m_sub["tourney_date"].notna()
    )
    m_sub = m_sub.loc[valid]
    if len(m_sub) == 0:
        return 0

    bucket_days = 21

    m_keyed = m_sub[["_m_surf", "_m_date_ord"]].copy()
    m_keyed["_m_wr"] = m_sub["winner_rank"]
    m_keyed["_m_lr"] = m_sub["loser_rank"]
    m_keyed["_midx"] = m_keyed.index
    m_keyed["_bucket"] = m_keyed["_m_date_ord"] // bucket_days

    # Duplicate into adjacent bucket for boundary coverage
    m_keyed2 = m_keyed.copy()
    m_keyed2["_bucket"] = m_keyed2["_bucket"] + 1
    m_keyed_all = pd.concat([m_keyed, m_keyed2], ignore_index=True)

    o_keyed = odds_df[["_surf", "_date_ord", "_wr", "_lr", "_ip_w", "_ip_l", "_ow", "_ol"]].copy()
    o_keyed.rename(columns={"_surf": "_m_surf"}, inplace=True)
    o_keyed = o_keyed.dropna(subset=["_wr", "_lr"])
    o_keyed["_oidx"] = o_keyed.index
    o_keyed["_bucket"] = o_keyed["_date_ord"] // bucket_days

    if len(o_keyed) == 0:
        return 0

    # Merge on surface + date bucket
    merged = m_keyed_all.merge(o_keyed, on=["_m_surf", "_bucket"], how="inner")
    if len(merged) == 0:
        return 0

    # Date window [0, +16]
    diff = merged["_date_ord"] - merged["_m_date_ord"]
    date_ok = (diff >= 0) & (diff <= 16)

    # Rank proximity
    wr_ok = (merged["_wr"] - merged["_m_wr"]).abs() <= tolerance
    lr_ok = (merged["_lr"] - merged["_m_lr"]).abs() <= tolerance

    merged = merged.loc[date_ok & wr_ok & lr_ok].copy()
    if len(merged) == 0:
        return 0

    # Pick closest date per match
    merged["_abs_diff"] = (merged["_date_ord"] - merged["_m_date_ord"]).abs()
    merged.sort_values("_abs_diff", inplace=True)
    best = merged.drop_duplicates(subset=["_midx"], keep="first")

    _assign_odds_vectorized(matches, best["_midx"].values, odds_df, best["_oidx"].values)
    return len(best)
