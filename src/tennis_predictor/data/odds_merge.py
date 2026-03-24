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
    min_year = matches["tourney_date"].dt.year.min()
    max_year = matches["tourney_date"].dt.year.max()
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
        odds_df["_w_last"] = odds_df["Winner"].apply(_extract_lastname)
        odds_df["_l_last"] = odds_df["Loser"].apply(_extract_lastname)
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
    # "Last F." format → just take first word(s) before the initial
    parts = name.split()
    if parts and parts[-1].endswith(".") and len(parts[-1]) <= 3:
        return " ".join(parts[:-1])
    # "First Last" → take last word
    if len(parts) >= 2:
        return parts[-1]
    return name


def _sackmann_lastname(name: str) -> str:
    """Extract last name from Sackmann 'First Last' format."""
    if pd.isna(name) or not name:
        return ""
    parts = str(name).strip().lower().split()
    return parts[-1] if parts else ""


def _apply_odds(matches: pd.DataFrame, idx: int, odds_row) -> None:
    """Write odds data to matches DataFrame."""
    matches.at[idx, "odds_implied_w"] = odds_row["_ip_w"]
    matches.at[idx, "odds_implied_l"] = odds_row["_ip_l"]
    matches.at[idx, "odds_pinnacle_w"] = odds_row["_ow"]
    matches.at[idx, "odds_pinnacle_l"] = odds_row["_ol"]


def _match_by_name(matches: pd.DataFrame, odds_df: pd.DataFrame) -> int:
    """Strategy 1: Match by last name + first initial + date window."""
    if "_w_last" not in odds_df.columns:
        return 0

    # Build lookup: (w_lastname, l_lastname) -> list of odds rows
    from collections import defaultdict
    lookup = defaultdict(list)
    for _, row in odds_df.iterrows():
        key = (row["_w_last"], row["_l_last"])
        lookup[key].append(row)

    matched = 0
    for idx in matches.index:
        if not pd.isna(matches.at[idx, "odds_implied_w"]):
            continue  # Already matched

        w_last = _sackmann_lastname(matches.at[idx, "winner_name"])
        l_last = _sackmann_lastname(matches.at[idx, "loser_name"])
        td = matches.at[idx, "tourney_date"]

        if not w_last or not l_last or pd.isna(td):
            continue

        key = (w_last, l_last)
        if key not in lookup:
            continue

        for orow in lookup[key]:
            diff = (orow["_date"] - td).days
            if -1 <= diff <= 16:
                _apply_odds(matches, idx, orow)
                matched += 1
                break

    return matched


def _match_by_lastname(matches: pd.DataFrame, odds_df: pd.DataFrame) -> int:
    """Strategy 2: Last name + surface + date window (for name format mismatches)."""
    matched = 0

    # Build date-indexed lookup
    from collections import defaultdict
    date_lookup = defaultdict(list)
    for _, row in odds_df.iterrows():
        d = str(row["_date"].date())
        date_lookup[d].append(row)

    for idx in matches.index:
        if not pd.isna(matches.at[idx, "odds_implied_w"]):
            continue

        w_last = _sackmann_lastname(matches.at[idx, "winner_name"])
        l_last = _sackmann_lastname(matches.at[idx, "loser_name"])
        surf = str(matches.at[idx, "surface"]).lower().strip()
        td = matches.at[idx, "tourney_date"]

        if not w_last or not l_last or pd.isna(td):
            continue

        # Search date window
        for day_offset in range(0, 17):
            check_date = str((td + pd.Timedelta(days=day_offset)).date())
            if check_date not in date_lookup:
                continue
            for orow in date_lookup[check_date]:
                if (orow.get("_surf", "") == surf and
                    w_last in orow.get("_w_last", "") and
                    l_last in orow.get("_l_last", "")):
                    _apply_odds(matches, idx, orow)
                    matched += 1
                    break
            if not pd.isna(matches.at[idx, "odds_implied_w"]):
                break

    return matched


def _match_by_rank(matches: pd.DataFrame, odds_df: pd.DataFrame, tolerance: int = 5) -> int:
    """Strategy 3: Rank proximity + surface + date window."""
    if "_wr" not in odds_df.columns:
        return 0

    matched = 0
    from collections import defaultdict
    date_lookup = defaultdict(list)
    for _, row in odds_df.iterrows():
        d = str(row["_date"].date())
        date_lookup[d].append(row)

    for idx in matches.index:
        if not pd.isna(matches.at[idx, "odds_implied_w"]):
            continue

        wr = matches.at[idx, "winner_rank"]
        lr = matches.at[idx, "loser_rank"]
        surf = str(matches.at[idx, "surface"]).lower().strip()
        td = matches.at[idx, "tourney_date"]

        if pd.isna(wr) or pd.isna(lr) or pd.isna(td):
            continue

        for day_offset in range(0, 17):
            check_date = str((td + pd.Timedelta(days=day_offset)).date())
            if check_date not in date_lookup:
                continue
            for orow in date_lookup[check_date]:
                if (orow.get("_surf", "") == surf and
                    abs(orow.get("_wr", 0) - wr) <= tolerance and
                    abs(orow.get("_lr", 0) - lr) <= tolerance):
                    _apply_odds(matches, idx, orow)
                    matched += 1
                    break
            if not pd.isna(matches.at[idx, "odds_implied_w"]):
                break

    return matched
