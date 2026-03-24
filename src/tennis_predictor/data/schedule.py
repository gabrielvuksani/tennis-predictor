"""Upcoming match schedule scraper — no API key required.

Primary source: Flashscore.ninja internal API (365+ matches/day, all levels)
Backup: Bovada API (includes odds)

Returns upcoming ATP singles matches with player names, rankings, tournament,
surface, and start times.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone

import requests

FLASHSCORE_BASE = "https://local-global.flashscore.ninja/2/x/feed"
FLASHSCORE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://www.flashscore.com/",
    "x-fsign": "SW9D1eZo",
    "Origin": "https://www.flashscore.com",
    "Accept": "*/*",
}

BOVADA_URL = (
    "https://www.bovada.lv/services/sports/event/coupon/events/A/description/tennis"
    "?lang=en&eventsLimit=100&preMatchOnly=true&marketFilterId=def"
)


def fetch_upcoming_matches(day_offset: int = 0) -> list[dict]:
    """Fetch upcoming tennis matches from Flashscore.

    Args:
        day_offset: 0=today, 1=tomorrow, -1=yesterday

    Returns:
        List of match dicts with player names, tournament, surface, etc.
    """
    matches = _fetch_flashscore(day_offset)
    if not matches:
        print("Flashscore unavailable, trying Bovada...")
        matches = _fetch_bovada()
    return matches


def _fetch_flashscore(day_offset: int = 0) -> list[dict]:
    """Scrape Flashscore.ninja internal API."""
    url = f"{FLASHSCORE_BASE}/f_2_{day_offset}_3_en_1"

    try:
        resp = requests.get(url, headers=FLASHSCORE_HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"Flashscore returned {resp.status_code}")
            return []

        return _parse_flashscore_response(resp.text)
    except requests.RequestException as e:
        print(f"Flashscore error: {e}")
        return []


def _parse_flashscore_response(text: str) -> list[dict]:
    """Parse Flashscore's custom text format into match dicts.

    Format: records separated by ¬~, fields within records separated by ¬,
    key-value pairs separated by ÷.
    """
    matches = []
    current_tournament = ""
    current_surface = "Hard"

    # Split by record separator
    records = text.split("¬~")

    for rec in records:
        fields = {}
        for part in rec.split("¬"):
            if "÷" in part:
                key, _, val = part.partition("÷")
                fields[key] = val

        # Tournament header record
        if "ZA" in fields:
            header = fields["ZA"]
            header_upper = header.upper()

            # Only ATP singles (skip doubles, WTA, challengers for main predictions)
            if "ATP" in header_upper and "SINGLE" in header_upper and "DOUBLE" not in header_upper:
                current_tournament = header
                header_lower = header.lower()
                if "hard" in header_lower:
                    current_surface = "Hard"
                elif "clay" in header_lower:
                    current_surface = "Clay"
                elif "grass" in header_lower:
                    current_surface = "Grass"
                else:
                    current_surface = "Hard"
            else:
                current_tournament = ""
            continue

        # Skip if not in an ATP tournament
        if not current_tournament:
            continue

        # Match data: AE=player1, AF=player2, AD=timestamp, AB=status
        if "AE" in fields and "AF" in fields:
            p1 = fields.get("AE", "").strip()
            p2 = fields.get("AF", "").strip()

            if not p1 or not p2:
                continue

            # Parse timestamp
            timestamp = fields.get("AD", "")
            start_time = ""
            if timestamp:
                try:
                    ts = int(timestamp)
                    start_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                except (ValueError, OSError):
                    pass

            # Status: 1=not started, 2=live, 3=finished
            status = fields.get("AB", "1")

            # Rankings
            p1_rank = _parse_int(fields.get("CA", ""))
            p2_rank = _parse_int(fields.get("CB", ""))

            # Clean tournament name
            tourney_clean = current_tournament
            # Extract just the tournament name from "ATP - SINGLES: Miami (USA), hard"
            if ":" in tourney_clean:
                tourney_clean = tourney_clean.split(":", 1)[1].strip()
            if "," in tourney_clean:
                tourney_clean = tourney_clean.split(",")[0].strip()

            match = {
                "player1": p1,
                "player2": p2,
                "tournament": tourney_clean,
                "surface": current_surface,
                "start_time": start_time,
                "status": "upcoming" if status == "1" else "live" if status == "2" else "finished",
                "p1_rank": p1_rank,
                "p2_rank": p2_rank,
                "source": "flashscore",
                "match_id": fields.get("AA", ""),
            }

            # Only include upcoming and live matches for predictions
            if status in ("1", "2"):
                matches.append(match)

    return matches


def _fetch_bovada() -> list[dict]:
    """Backup: fetch from Bovada API (includes odds)."""
    try:
        resp = requests.get(
            BOVADA_URL,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        if resp.status_code != 200:
            return []

        data = resp.json()
        matches = []

        for event_group in data:
            path = event_group.get("path", [])
            # Find ATP events
            is_atp = any("atp" in p.get("description", "").lower() for p in path)
            if not is_atp:
                continue

            tournament = path[-1].get("description", "") if path else ""

            for event in event_group.get("events", []):
                competitors = event.get("competitors", [])
                if len(competitors) != 2:
                    continue

                p1 = competitors[0].get("name", "")
                p2 = competitors[1].get("name", "")
                start_ms = event.get("startTime", 0)

                # Extract odds if available
                odds_p1 = None
                odds_p2 = None
                for market in event.get("displayGroups", [{}])[0].get("markets", []):
                    if market.get("description") == "Moneyline":
                        outcomes = market.get("outcomes", [])
                        if len(outcomes) == 2:
                            odds_p1 = _american_to_decimal(outcomes[0].get("price", {}).get("american", ""))
                            odds_p2 = _american_to_decimal(outcomes[1].get("price", {}).get("american", ""))

                matches.append({
                    "player1": p1,
                    "player2": p2,
                    "tournament": tournament,
                    "surface": "Hard",  # Bovada doesn't provide surface
                    "start_time": datetime.fromtimestamp(
                        start_ms / 1000, tz=timezone.utc
                    ).isoformat() if start_ms else "",
                    "status": "upcoming",
                    "p1_rank": None,
                    "p2_rank": None,
                    "odds_p1": odds_p1,
                    "odds_p2": odds_p2,
                    "source": "bovada",
                })

        return matches
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"Bovada error: {e}")
        return []


def _parse_int(s: str) -> int | None:
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def _american_to_decimal(american: str) -> float | None:
    """Convert American odds to decimal odds."""
    try:
        odds = int(american)
        if odds > 0:
            return 1 + odds / 100
        elif odds < 0:
            return 1 + 100 / abs(odds)
    except (ValueError, TypeError):
        pass
    return None
