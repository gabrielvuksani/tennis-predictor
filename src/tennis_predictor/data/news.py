"""News and injury tracking via RSS feeds and free APIs.

Primary source: ATP Tour official RSS feed (free, unlimited, no API key).
Supplementary: GNews free tier (100 requests/day), Reddit via PRAW.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

# ATP Official RSS Feed
ATP_RSS_URL = "https://www.atptour.com/en/media/rss-feed/xml-feed"

# Free news API
GNEWS_BASE = "https://gnews.io/api/v4"

# Injury-related keywords
INJURY_KEYWORDS = [
    "injury", "injured", "withdrawal", "withdrew", "retired", "retirement",
    "surgery", "operation", "ankle", "knee", "shoulder", "back", "wrist",
    "hip", "hamstring", "calf", "thigh", "foot", "elbow", "abdominal",
    "fitness", "fitness doubt", "doubt", "questionable", "uncertain",
    "pulled out", "sidelined", "out of", "ruled out", "miss",
]


def fetch_atp_rss() -> list[dict]:
    """Fetch latest news from ATP Tour RSS feed."""
    try:
        import feedparser
    except ImportError:
        return []

    try:
        feed = feedparser.parse(ATP_RSS_URL)
        articles = []
        for entry in feed.entries:
            articles.append({
                "title": entry.get("title", ""),
                "summary": entry.get("summary", ""),
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
                "source": "atp_official",
            })
        return articles
    except Exception:
        return []


def search_news(
    query: str,
    api_key: str | None = None,
    max_results: int = 10,
) -> list[dict]:
    """Search for tennis news using GNews free tier.

    Free tier: 100 requests/day, 10 articles per request, 12h delay.
    """
    if api_key is None:
        return []

    try:
        resp = requests.get(
            f"{GNEWS_BASE}/search",
            params={
                "q": query,
                "lang": "en",
                "max": max_results,
                "apikey": api_key,
            },
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            return [
                {
                    "title": a.get("title", ""),
                    "description": a.get("description", ""),
                    "url": a.get("url", ""),
                    "published_at": a.get("publishedAt", ""),
                    "source": a.get("source", {}).get("name", "gnews"),
                }
                for a in data.get("articles", [])
            ]
        return []
    except requests.RequestException:
        return []


def detect_injury_signals(articles: list[dict], player_name: str) -> dict:
    """Analyze news articles for injury/withdrawal signals about a player.

    Returns a dict with injury probability estimate and details.
    """
    name_lower = player_name.lower()
    name_parts = name_lower.split()

    relevant_articles = []
    for article in articles:
        text = (article.get("title", "") + " " + article.get("summary", "") +
                " " + article.get("description", "")).lower()

        # Check if article mentions the player
        if any(part in text for part in name_parts if len(part) > 2):
            # Check for injury keywords
            injury_matches = [kw for kw in INJURY_KEYWORDS if kw in text]
            if injury_matches:
                relevant_articles.append({
                    **article,
                    "injury_keywords": injury_matches,
                })

    if not relevant_articles:
        return {
            "injury_signal": 0.0,
            "n_articles": 0,
            "keywords": [],
        }

    # Simple scoring: more articles and more severe keywords = higher signal
    severity_weights = {
        "surgery": 1.0, "operation": 1.0, "ruled out": 0.9,
        "withdrew": 0.8, "withdrawal": 0.8, "sidelined": 0.8,
        "injured": 0.6, "injury": 0.5, "retired": 0.5,
        "doubt": 0.3, "questionable": 0.3, "uncertain": 0.3,
        "fitness": 0.2,
    }

    max_severity = 0.0
    all_keywords = set()
    for article in relevant_articles:
        for kw in article["injury_keywords"]:
            all_keywords.add(kw)
            weight = severity_weights.get(kw, 0.3)
            max_severity = max(max_severity, weight)

    return {
        "injury_signal": min(max_severity, 1.0),
        "n_articles": len(relevant_articles),
        "keywords": list(all_keywords),
    }


def infer_retirement_history(matches: pd.DataFrame, player_id: str) -> dict:
    """Infer injury history from past match retirements.

    Uses Jeff Sackmann data where retirements are encoded in the score field.
    """
    # Find matches where this player retired (lost with retirement)
    player_losses = matches[
        ((matches.get("loser_id") == player_id) if "loser_id" in matches.columns
         else (matches.get("p2_id") == player_id)) &
        (matches.get("retirement", False) == True)
    ]

    if len(player_losses) == 0:
        return {
            "total_retirements": 0,
            "recent_retirements_12m": 0,
            "retirement_rate": 0.0,
        }

    total_matches = len(matches[
        (matches.get("winner_id", matches.get("p1_id")) == player_id) |
        (matches.get("loser_id", matches.get("p2_id")) == player_id)
    ])

    # Count recent retirements
    if "tourney_date" in matches.columns:
        recent_cutoff = matches["tourney_date"].max() - timedelta(days=365)
        recent = player_losses[player_losses["tourney_date"] >= recent_cutoff]
    else:
        recent = pd.DataFrame()

    return {
        "total_retirements": len(player_losses),
        "recent_retirements_12m": len(recent),
        "retirement_rate": len(player_losses) / max(total_matches, 1),
    }
