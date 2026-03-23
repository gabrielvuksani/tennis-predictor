"""Court speed data integration.

Court speed is a critical feature — it affects playing styles differently.
Big servers dominate fast courts; defensive players thrive on slow courts.

Sources:
- CourtSpeed.com (CC BY 4.0, downloadable CSV, CPI data 2012-2026)
- Tennis Abstract surface speed ratings (ace-rate based, 1991-2026)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from tennis_predictor.config import CACHE_DIR, COURT_SPEED_DIR


def scrape_tennis_abstract_speed(year: int) -> pd.DataFrame | None:
    """Scrape surface speed ratings from Tennis Abstract.

    The speed rating is ace-rate adjusted for server/returner quality.
    Tour average = 1.0. Range ~0.43 (slowest clay) to ~1.46 (fastest hard).
    """
    cache_file = CACHE_DIR / "court_speed" / f"ta_speed_{year}.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        return pd.DataFrame(json.loads(cache_file.read_text()))

    url = f"https://www.tennisabstract.com/cgi-bin/surface-speed.cgi?year={year}"

    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "TennisPredictor/0.1 (research)"
        })
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        # The data table is the last table with proper th/td rows
        tables = soup.find_all("table")
        table = None
        for t in tables:
            header_row = t.find("tr")
            if header_row:
                ths = header_row.find_all("th")
                if any("Tournament" in (th.get_text(strip=True)) for th in ths):
                    table = t
                    break
        if table is None:
            return None

        rows = []
        for tr in table.find_all("tr")[1:]:  # Skip header
            tds = tr.find_all("td")
            if len(tds) >= 4:
                try:
                    speed_str = tds[-1].get_text(strip=True)
                    rows.append({
                        "tournament": tds[1].get_text(strip=True),
                        "surface": tds[2].get_text(strip=True),
                        "speed_rating": float(speed_str),
                        "year": year,
                    })
                except (ValueError, IndexError):
                    continue

        if rows:
            cache_file.write_text(json.dumps(rows, indent=2))
            return pd.DataFrame(rows)
        return None

    except (requests.RequestException, ValueError):
        return None


def load_court_speed_history(start_year: int = 2010, end_year: int = 2026) -> pd.DataFrame:
    """Load court speed data across years.

    Returns a DataFrame with tournament, surface, year, and speed_rating.
    """
    frames = []
    for year in range(start_year, end_year + 1):
        df = scrape_tennis_abstract_speed(year)
        if df is not None and len(df) > 0:
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["tournament", "surface", "year", "speed_rating"])

    return pd.concat(frames, ignore_index=True)


def get_tournament_speed(
    tourney_name: str,
    year: int,
    speed_data: pd.DataFrame | None = None,
) -> float:
    """Get the speed rating for a tournament in a given year.

    Falls back to the tournament's average across years if the specific
    year is not available, then to the surface average.
    """
    if speed_data is None or len(speed_data) == 0:
        return np.nan

    # Normalize names for matching
    name_lower = tourney_name.lower().strip()

    # Try exact year match
    year_data = speed_data[speed_data["year"] == year]
    for _, row in year_data.iterrows():
        if _fuzzy_match(row["tournament"], name_lower):
            return row["speed_rating"]

    # Fall back to tournament average across all years
    all_matches = speed_data[
        speed_data["tournament"].apply(lambda x: _fuzzy_match(x, name_lower))
    ]
    if len(all_matches) > 0:
        return float(all_matches["speed_rating"].mean())

    return np.nan


def _fuzzy_match(tournament_name: str, query: str) -> bool:
    """Simple fuzzy matching for tournament names."""
    tn = tournament_name.lower().strip()
    # Exact match
    if tn == query or query in tn or tn in query:
        return True
    # Handle common variations
    variations = {
        "australian open": ["ao", "melbourne"],
        "roland garros": ["french open", "paris"],
        "wimbledon": ["wimbledon"],
        "us open": ["us open", "flushing"],
        "indian wells": ["bnp paribas", "indian wells"],
        "miami": ["miami open", "miami"],
        "monte carlo": ["monte carlo", "monte-carlo"],
    }
    for canonical, aliases in variations.items():
        if any(a in query for a in aliases) and any(a in tn for a in aliases):
            return True
    return False
