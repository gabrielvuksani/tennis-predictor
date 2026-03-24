"""TennisMyLife (TML) Database loader — 2025-2026 match data.

TML provides daily-updated ATP match results in the exact same CSV format
as JeffSackmann, filling the gap since his repo stopped updating in Dec 2024.

Source: https://github.com/Tennismylife/TML-Database
License: MIT
API: https://stats.tennismylife.org/api/data-files
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from tennis_predictor.config import RAW_DIR

TML_API = "https://stats.tennismylife.org/api/data-files"
TML_DATA_BASE = "https://stats.tennismylife.org/data"


def download_tml_data(years: list[int] | None = None, include_challengers: bool = True) -> list[Path]:
    """Download TML CSV files for specified years.

    Args:
        years: Which years to download. Default: [2025, 2026].
        include_challengers: Also download challenger tour data.

    Returns:
        List of downloaded file paths.
    """
    if years is None:
        years = [2025, 2026]

    tml_dir = RAW_DIR / "tml"
    tml_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for year in years:
        files_to_get = [f"{year}.csv"]
        if include_challengers:
            files_to_get.append(f"{year}_challenger.csv")

        for filename in files_to_get:
            url = f"{TML_DATA_BASE}/{filename}"
            dest = tml_dir / filename

            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200 and len(resp.text) > 100:
                    dest.write_text(resp.text)
                    downloaded.append(dest)
                    # Count matches
                    lines = resp.text.count("\n") - 1
                    print(f"  Downloaded {filename}: {lines} matches")
            except requests.RequestException as e:
                print(f"  Failed to download {filename}: {e}")

    return downloaded


def load_tml_matches(years: list[int] | None = None) -> pd.DataFrame:
    """Load TML match data, downloading if needed.

    Returns a DataFrame in the same format as Sackmann data.
    """
    if years is None:
        years = [2025, 2026]

    tml_dir = RAW_DIR / "tml"
    frames = []

    for year in years:
        for suffix in ["", "_challenger"]:
            path = tml_dir / f"{year}{suffix}.csv"

            # Download if not cached
            if not path.exists():
                download_tml_data([year])

            if path.exists():
                try:
                    df = pd.read_csv(path, dtype={"winner_id": str, "loser_id": str}, low_memory=False)
                    frames.append(df)
                except Exception as e:
                    print(f"  Error loading {path.name}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Parse dates (same format as Sackmann: YYYYMMDD)
    combined["tourney_date"] = pd.to_datetime(
        combined["tourney_date"], format="%Y%m%d", errors="coerce"
    )

    print(f"  TML data: {len(combined)} matches ({combined['tourney_date'].min().date()} to {combined['tourney_date'].max().date()})")
    return combined
