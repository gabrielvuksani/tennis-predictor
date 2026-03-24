"""Live ATP rankings from ESPN's free hidden API.

No API key needed. Returns top 150 players with current rank, points,
previous rank, player name, age, and country.

Primary: ESPN API (JSON, 150 players, updated weekly)
Fallback: TennisExplorer scraping (500+ players, HTML)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import requests

from tennis_predictor.config import CACHE_DIR

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/tennis/atp/rankings"
TENNIS_EXPLORER_URL = "https://www.tennisexplorer.com/ranking/atp-men/"


def fetch_live_rankings(use_cache: bool = True) -> dict[str, dict]:
    """Fetch current ATP rankings. Returns name_lower → rank info dict.

    Cached for 24 hours. Uses ESPN API (top 150), falls back to
    TennisExplorer scraping for broader coverage.
    """
    cache_dir = CACHE_DIR / "rankings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "live_rankings.json"

    # Check cache (24 hour validity)
    if use_cache and cache_file.exists():
        cache_data = json.loads(cache_file.read_text())
        cached_at = cache_data.get("cached_at", "")
        if cached_at:
            try:
                age_hours = (datetime.now() - datetime.fromisoformat(cached_at)).total_seconds() / 3600
                if age_hours < 24:
                    return cache_data.get("rankings", {})
            except (ValueError, TypeError):
                pass

    # Fetch from ESPN
    rankings = _fetch_espn()

    if not rankings:
        print("ESPN unavailable, trying TennisExplorer...")
        rankings = _fetch_tennis_explorer()

    if rankings:
        # Save cache
        cache_data = {
            "cached_at": datetime.now().isoformat(),
            "source": "espn" if rankings else "tennis_explorer",
            "count": len(rankings),
            "rankings": rankings,
        }
        cache_file.write_text(json.dumps(cache_data, indent=2))
        print(f"  Live rankings: {len(rankings)} players (source: ESPN)")

    return rankings


def _fetch_espn() -> dict[str, dict]:
    """Fetch from ESPN hidden API — free, no key, JSON."""
    try:
        resp = requests.get(ESPN_URL, timeout=10, headers={
            "User-Agent": "Mozilla/5.0",
        })
        if resp.status_code != 200:
            return {}

        data = resp.json()
        ranks_list = data.get("rankings", [{}])[0].get("ranks", [])

        rankings = {}
        for entry in ranks_list:
            athlete = entry.get("athlete", {})
            name = athlete.get("displayName", "")
            if not name:
                continue

            rank_info = {
                "rank": entry.get("current"),
                "previous_rank": entry.get("previous"),
                "points": entry.get("points"),
                "name": name,
                "first_name": athlete.get("firstName", ""),
                "last_name": athlete.get("lastName", ""),
                "age": athlete.get("age"),
                "country": athlete.get("citizenshipCountry", ""),
            }

            # Index by lowercase name for matching
            rankings[name.lower()] = rank_info

            # Also index by "Last F." format for Flashscore matching
            parts = name.split()
            if len(parts) >= 2:
                last = " ".join(parts[1:])
                initial = parts[0][0]
                flash_key = f"{last} {initial}.".lower()
                rankings[flash_key] = rank_info

        return rankings
    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
        print(f"  ESPN error: {e}")
        return {}


def _fetch_tennis_explorer(max_pages: int = 4) -> dict[str, dict]:
    """Fallback: scrape TennisExplorer for rankings (200 players)."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return {}

    rankings = {}

    for page in range(1, max_pages + 1):
        try:
            resp = requests.get(
                f"{TENNIS_EXPLORER_URL}?page={page}",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
            )
            if resp.status_code != 200:
                break

            soup = BeautifulSoup(resp.text, "html.parser")

            for row in soup.select("tbody tr"):
                cells = row.find_all("td")
                if len(cells) < 4:
                    continue

                rank_cell = cells[0].get_text(strip=True)
                name_cell = cells[1]
                name_link = name_cell.find("a")
                name = name_link.get_text(strip=True) if name_link else ""

                if not name or not rank_cell.isdigit():
                    continue

                rank_info = {
                    "rank": int(rank_cell),
                    "name": name,
                    "points": None,
                }
                rankings[name.lower()] = rank_info

        except requests.RequestException:
            break

    return rankings


def get_player_rank(player_name: str, rankings: dict[str, dict] | None = None) -> int | None:
    """Get a player's current ATP ranking.

    Tries multiple name formats for matching.
    """
    if rankings is None:
        rankings = fetch_live_rankings()

    if not rankings or not player_name:
        return None

    name_lower = player_name.strip().lower()

    # Direct lookup
    if name_lower in rankings:
        return rankings[name_lower].get("rank")

    # Try partial matching (last name)
    parts = name_lower.split()
    if parts:
        last = parts[0].rstrip(".")
        candidates = [
            v for k, v in rankings.items()
            if k.split()[-1] == last or (len(k.split()) > 1 and k.split()[-1] == last)
        ]
        if len(candidates) == 1:
            return candidates[0].get("rank")

    return None
