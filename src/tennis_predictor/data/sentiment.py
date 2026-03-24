"""Reddit sentiment analysis for tennis players.

Monitors r/tennis for player mentions, injury chatter, momentum signals,
and community sentiment. Uses PRAW (official Reddit API, free, 100 req/min).

Sentiment is a supplementary signal — it captures things stats miss:
- Injury rumors before official announcements
- Player confidence/motivation from press conferences
- Public money direction (line movement correlation)
- Insider observations from practice sessions
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from tennis_predictor.config import CACHE_DIR

# Sentiment keywords and their weights
POSITIVE_SIGNALS = {
    "looking great": 0.8, "dominant": 0.7, "peak form": 0.9,
    "confident": 0.6, "unstoppable": 0.8, "incredible": 0.7,
    "on fire": 0.8, "crushing it": 0.7, "best tennis": 0.8,
    "motivated": 0.6, "healthy": 0.5, "fit": 0.4,
    "great form": 0.7, "playing well": 0.6, "strong favorite": 0.5,
}

NEGATIVE_SIGNALS = {
    "injured": -0.8, "injury": -0.7, "withdrew": -0.9,
    "struggling": -0.6, "tired": -0.5, "fatigued": -0.5,
    "doubt": -0.4, "questionable": -0.5, "limping": -0.8,
    "out of form": -0.7, "lost confidence": -0.7, "mental": -0.4,
    "retirement": -0.6, "pulled out": -0.9, "medical timeout": -0.7,
    "surgery": -0.9, "not 100%": -0.6, "fitness concern": -0.7,
}


def get_player_sentiment(
    player_name: str,
    days_back: int = 3,
    use_cache: bool = True,
) -> dict:
    """Get sentiment score for a player from Reddit r/tennis.

    Returns a dict with sentiment score (-1 to +1), confidence, and details.
    Uses cache to avoid hitting Reddit on every prediction.
    """
    cache_dir = CACHE_DIR / "sentiment"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check cache (valid for 6 hours)
    cache_key = re.sub(r"[^a-z0-9]", "_", player_name.lower())
    cache_file = cache_dir / f"{cache_key}.json"

    if use_cache and cache_file.exists():
        cache_data = json.loads(cache_file.read_text())
        cache_time = datetime.fromisoformat(cache_data.get("cached_at", "2000-01-01"))
        if (datetime.now() - cache_time).total_seconds() < 6 * 3600:
            return cache_data

    # Fetch from Reddit
    posts = _fetch_reddit_mentions(player_name, days_back)

    if not posts:
        result = {
            "player": player_name,
            "sentiment_score": 0.0,
            "confidence": 0.0,
            "n_mentions": 0,
            "injury_signal": 0.0,
            "momentum_signal": 0.0,
            "cached_at": datetime.now().isoformat(),
        }
        cache_file.write_text(json.dumps(result))
        return result

    # Analyze sentiment
    sentiment_scores = []
    injury_signals = []
    momentum_signals = []

    name_parts = player_name.lower().split()
    last_name = name_parts[-1] if name_parts else player_name.lower()

    for post in posts:
        text = (post.get("title", "") + " " + post.get("body", "")).lower()

        # Check if post actually mentions this player
        if last_name not in text:
            continue

        # Score positive signals
        for phrase, weight in POSITIVE_SIGNALS.items():
            if phrase in text:
                sentiment_scores.append(weight)
                momentum_signals.append(weight)

        # Score negative signals
        for phrase, weight in NEGATIVE_SIGNALS.items():
            if phrase in text:
                sentiment_scores.append(weight)
                if "injur" in phrase or "withdrew" in phrase or "surgery" in phrase:
                    injury_signals.append(abs(weight))

    n_mentions = len([p for p in posts if last_name in
                      (p.get("title", "") + " " + p.get("body", "")).lower()])

    result = {
        "player": player_name,
        "sentiment_score": float(np.mean(sentiment_scores)) if sentiment_scores else 0.0,
        "confidence": min(1.0, n_mentions / 10),  # More mentions = more confident
        "n_mentions": n_mentions,
        "injury_signal": float(np.mean(injury_signals)) if injury_signals else 0.0,
        "momentum_signal": float(np.mean(momentum_signals)) if momentum_signals else 0.0,
        "cached_at": datetime.now().isoformat(),
    }

    cache_file.write_text(json.dumps(result))
    return result


def _fetch_reddit_mentions(player_name: str, days_back: int = 3) -> list[dict]:
    """Fetch recent Reddit posts mentioning a player from r/tennis."""
    try:
        import praw
    except ImportError:
        return []

    client_id = os.environ.get("REDDIT_CLIENT_ID", "")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        # Fall back to unauthenticated read (limited but works)
        return _fetch_reddit_unauthenticated(player_name, days_back)

    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent="TennisPredictor/0.3 (research)",
        )
        reddit.read_only = True

        subreddit = reddit.subreddit("tennis")
        posts = []

        # Search for player mentions
        last_name = player_name.split()[-1] if player_name.split() else player_name
        for submission in subreddit.search(last_name, time_filter="week", limit=25):
            posts.append({
                "title": submission.title,
                "body": submission.selftext[:500],
                "score": submission.score,
                "created": submission.created_utc,
            })

        return posts
    except Exception:
        return []


def _fetch_reddit_unauthenticated(player_name: str, days_back: int = 3) -> list[dict]:
    """Fallback: fetch Reddit posts without authentication using JSON endpoint."""
    import requests

    last_name = player_name.split()[-1] if player_name.split() else player_name

    try:
        url = f"https://www.reddit.com/r/tennis/search.json"
        resp = requests.get(
            url,
            params={"q": last_name, "restrict_sr": "on", "t": "week", "limit": 15},
            headers={"User-Agent": "TennisPredictor/0.3"},
            timeout=10,
        )
        if resp.status_code != 200:
            return []

        data = resp.json()
        posts = []
        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})
            posts.append({
                "title": post.get("title", ""),
                "body": post.get("selftext", "")[:500],
                "score": post.get("score", 0),
                "created": post.get("created_utc", 0),
            })
        return posts
    except Exception:
        return []


def batch_sentiment(player_names: list[str]) -> dict[str, dict]:
    """Get sentiment for multiple players efficiently."""
    results = {}
    for name in player_names:
        results[name] = get_player_sentiment(name)
    return results
