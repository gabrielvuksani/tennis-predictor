"""Betting odds data loader.

Loads historical closing odds from tennis-data.co.uk (free, 2000-present).
Also supports The Odds API free tier for current/live odds collection.
"""

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from tennis_predictor.config import ODDS_DIR, CACHE_DIR

# tennis-data.co.uk URL patterns
TENNIS_DATA_BASE = "http://www.tennis-data.co.uk"
TENNIS_DATA_YEARS = {
    # year: path on tennis-data.co.uk
    2001: "{year}/{year}.zip",
    2002: "{year}/{year}.zip",
    2003: "{year}/{year}.zip",
    2004: "{year}/{year}.zip",
    2005: "{year}/{year}.zip",
    2006: "{year}/{year}.zip",
    2007: "{year}/{year}.zip",
    2008: "{year}/{year}.zip",
    2009: "{year}/{year}.zip",
    2010: "{year}/{year}.zip",
    2011: "{year}/{year}.zip",
    2012: "{year}/{year}.zip",
    2013: "{year}/{year}.zip",
    2014: "{year}/{year}.zip",
    2015: "{year}/{year}.zip",
    2016: "{year}/{year}.zip",
    2017: "{year}/{year}.zip",
    2018: "{year}/{year}.zip",
    2019: "{year}/{year}.zip",
    2020: "{year}/{year}.zip",
    2021: "{year}/{year}.zip",
    2022: "{year}/{year}.zip",
    2023: "{year}/{year}.zip",
    2024: "{year}/{year}.zip",
    2025: "{year}/{year}.zip",
}

# Key odds columns we want
ODDS_COLUMNS = {
    "B365W": "odds_b365_w", "B365L": "odds_b365_l",
    "PSW": "odds_pinnacle_w", "PSL": "odds_pinnacle_l",
    "MaxW": "odds_max_w", "MaxL": "odds_max_l",
    "AvgW": "odds_avg_w", "AvgL": "odds_avg_l",
}

# Match key columns for joining with Sackmann data
MATCH_KEY_COLS = ["Date", "Winner", "Loser", "WRank", "LRank", "Surface", "Round", "Best of"]


def _download_with_retry(url: str, max_retries: int = 3) -> bytes | None:
    """Download a URL with retries."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30, headers={
                "User-Agent": "TennisPredictor/0.1 (research project)"
            })
            if response.status_code == 200:
                return response.content
            elif response.status_code == 404:
                return None
        except requests.RequestException:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def download_tennis_data_odds(
    start_year: int = 2001,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Download historical odds from tennis-data.co.uk.

    Falls back to cached files if download fails.
    """
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    frames = []

    for year in tqdm(range(int(start_year), int(end_year) + 1), desc="Downloading odds data"):
        cache_path = ODDS_DIR / f"tennis_data_{year}.csv"

        # Use cache if available
        if cache_path.exists():
            df = pd.read_csv(cache_path)
            frames.append(df)
            continue

        # Try to download
        url = f"{TENNIS_DATA_BASE}/{year}/{year}.zip"
        data = _download_with_retry(url)
        if data is None:
            # Try Excel format
            url = f"{TENNIS_DATA_BASE}/{year}/{year}.xlsx"
            data = _download_with_retry(url)

        if data is not None:
            try:
                if url.endswith(".zip"):
                    import zipfile
                    with zipfile.ZipFile(io.BytesIO(data)) as z:
                        for name in z.namelist():
                            if name.endswith((".csv", ".xls", ".xlsx")):
                                with z.open(name) as f:
                                    if name.endswith(".csv"):
                                        df = pd.read_csv(f, encoding="latin-1")
                                    else:
                                        df = pd.read_excel(io.BytesIO(f.read()))
                                    df.to_csv(cache_path, index=False)
                                    frames.append(df)
                                    break
                elif url.endswith(".xlsx"):
                    df = pd.read_excel(io.BytesIO(data))
                    df.to_csv(cache_path, index=False)
                    frames.append(df)
            except Exception as e:
                print(f"Warning: Could not parse {year} odds data: {e}")

    if not frames:
        print("Warning: No odds data loaded. tennis-data.co.uk may be unavailable.")
        return pd.DataFrame()

    odds = pd.concat(frames, ignore_index=True)
    return _clean_odds(odds)


def _clean_odds(odds: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize odds data."""
    # Parse date
    if "Date" in odds.columns:
        odds["match_date"] = pd.to_datetime(odds["Date"], dayfirst=True, errors="coerce")
    elif "date" in odds.columns:
        odds["match_date"] = pd.to_datetime(odds["date"], dayfirst=True, errors="coerce")
    else:
        odds["match_date"] = pd.NaT

    # Standardize player names
    for col in ["Winner", "Loser"]:
        if col in odds.columns:
            odds[col] = odds[col].str.strip()

    # Rename odds columns
    rename_map = {k: v for k, v in ODDS_COLUMNS.items() if k in odds.columns}
    odds = odds.rename(columns=rename_map)

    # Convert odds to numeric
    for col in rename_map.values():
        if col in odds.columns:
            odds[col] = pd.to_numeric(odds[col], errors="coerce")

    # Compute implied probabilities from Pinnacle odds (sharpest bookmaker)
    if "odds_pinnacle_w" in odds.columns and "odds_pinnacle_l" in odds.columns:
        raw_p_w = 1.0 / odds["odds_pinnacle_w"]
        raw_p_l = 1.0 / odds["odds_pinnacle_l"]
        total = raw_p_w + raw_p_l
        # Remove overround (vig) via normalization
        odds["implied_prob_w"] = raw_p_w / total
        odds["implied_prob_l"] = raw_p_l / total
    elif "odds_avg_w" in odds.columns and "odds_avg_l" in odds.columns:
        raw_p_w = 1.0 / odds["odds_avg_w"]
        raw_p_l = 1.0 / odds["odds_avg_l"]
        total = raw_p_w + raw_p_l
        odds["implied_prob_w"] = raw_p_w / total
        odds["implied_prob_l"] = raw_p_l / total

    return odds


def compute_implied_probabilities(
    winner_odds: float | np.ndarray,
    loser_odds: float | np.ndarray,
    method: str = "normalized",
) -> tuple:
    """Convert decimal odds to implied probabilities with vig removal.

    Args:
        winner_odds: Decimal odds for the winner.
        loser_odds: Decimal odds for the loser.
        method: 'normalized' (basic), 'power' (Shin method), or 'raw' (no vig removal).

    Returns:
        Tuple of (prob_player1, prob_player2) after vig removal.
    """
    raw_p1 = 1.0 / np.asarray(winner_odds, dtype=float)
    raw_p2 = 1.0 / np.asarray(loser_odds, dtype=float)

    if method == "raw":
        return raw_p1, raw_p2
    elif method == "normalized":
        total = raw_p1 + raw_p2
        return raw_p1 / total, raw_p2 / total
    elif method == "power":
        # Shin's method for devigging (more accurate for favorites/underdogs)
        total = raw_p1 + raw_p2
        z = (total - 1) / (total * (total - 1))
        # Iterative Shin solver (simplified)
        for _ in range(10):
            z_new = ((raw_p1 - z) * (raw_p2 - z) / ((1 - z) ** 2))
            if abs(z_new - z) < 1e-8:
                break
            z = z_new
        p1 = (np.sqrt(z ** 2 + 4 * (1 - z) * raw_p1 ** 2 / total) - z) / (2 * (1 - z))
        p2 = 1 - p1
        return p1, p2
    else:
        raise ValueError(f"Unknown method: {method}")


# === The Odds API (free tier, 500 credits/month) ===

ODDS_API_BASE = "https://api.the-odds-api.com/v4"


def fetch_live_odds(
    api_key: str,
    sport: str = "tennis_atp",
    markets: str = "h2h",
    regions: str = "us,eu,uk",
) -> pd.DataFrame:
    """Fetch current odds from The Odds API (1 credit per market per region).

    Args:
        api_key: Your The Odds API key.
        sport: Sport key (tennis_atp, tennis_wta, etc.).
        markets: Comma-separated market types.
        regions: Comma-separated regions.
    """
    url = f"{ODDS_API_BASE}/sports/{sport}/odds"
    params = {
        "apiKey": api_key,
        "markets": markets,
        "regions": regions,
        "oddsFormat": "decimal",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    rows = []
    for event in data:
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] == "h2h":
                    outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                    if len(outcomes) == 2:
                        players = list(outcomes.keys())
                        rows.append({
                            "event_id": event["id"],
                            "sport": event["sport_key"],
                            "commence_time": event["commence_time"],
                            "home_team": event.get("home_team", players[0]),
                            "away_team": event.get("away_team", players[1]),
                            "bookmaker": bookmaker["key"],
                            "player1": players[0],
                            "player2": players[1],
                            "odds_p1": outcomes[players[0]],
                            "odds_p2": outcomes[players[1]],
                            "last_update": bookmaker["last_update"],
                        })

    return pd.DataFrame(rows)
