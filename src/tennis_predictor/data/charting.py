"""Jeff Sackmann Match Charting Project data loader.

Loads point-by-point enrichment data from the Match Charting Project (5000+ matches).
Provides player-level rolling stats: rally length, net approach rate, winner rate,
unforced error rate, and first-strike aggression.

Source: https://github.com/JeffSackmann/tennis_MatchChartingProject (CC BY-NC-SA 4.0)

IMPORTANT: This is supplementary data. Not every player has charting data.
Callers must handle None returns gracefully.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from tennis_predictor.config import CACHE_DIR, RAW_DIR, SACKMANN_DIR

logger = logging.getLogger(__name__)

CHARTING_REPO_URL = "https://github.com/JeffSackmann/tennis_MatchChartingProject.git"
CHARTING_DIR = RAW_DIR / "tennis_charting"
CHARTING_CACHE_FILE = CACHE_DIR / "charting" / "player_charting_stats.json"

# Minimum fuzzy match score (0-100) to consider two player names the same
_FUZZY_MATCH_THRESHOLD = 88

# Module-level caches so we only parse once per process
_player_stats_cache: dict[str, dict[str, float]] | None = None
_name_to_id_cache: dict[str, str] | None = None


# ---------------------------------------------------------------------------
# Repository management
# ---------------------------------------------------------------------------

def clone_or_update_charting_repo() -> Path:
    """Clone the Match Charting Project repo, or pull if already cloned.

    Returns the repo directory path. Raises no exception on network failure;
    returns the directory path regardless (it may not exist if clone failed).
    """
    try:
        if CHARTING_DIR.exists() and (CHARTING_DIR / ".git").exists():
            logger.info("Updating tennis_MatchChartingProject repository...")
            subprocess.run(
                ["git", "pull", "--quiet"],
                cwd=CHARTING_DIR,
                check=True,
                capture_output=True,
                timeout=120,
            )
        else:
            logger.info("Cloning tennis_MatchChartingProject repository...")
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", CHARTING_REPO_URL, str(CHARTING_DIR)],
                check=True,
                capture_output=True,
                timeout=300,
            )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Could not clone/update charting repo: %s", exc)
    return CHARTING_DIR


# ---------------------------------------------------------------------------
# Name-to-ID mapping via the main Sackmann player list
# ---------------------------------------------------------------------------

def _build_name_to_id_map() -> dict[str, str]:
    """Build a mapping from normalized player name -> Sackmann player_id.

    Uses the atp_players.csv from the main tennis_atp repo.
    """
    global _name_to_id_cache
    if _name_to_id_cache is not None:
        return _name_to_id_cache

    players_path = SACKMANN_DIR / "atp_players.csv"
    if not players_path.exists():
        logger.warning("atp_players.csv not found; player ID mapping unavailable")
        _name_to_id_cache = {}
        return _name_to_id_cache

    players = pd.read_csv(
        players_path, dtype={"player_id": str, "wikidata_id": str}
    )
    mapping: dict[str, str] = {}

    for _, row in players.iterrows():
        first = str(row.get("name_first", "")).strip()
        last = str(row.get("name_last", "")).strip()
        pid = str(row["player_id"]).strip()

        if not last or last == "nan":
            continue

        # Store multiple forms: "First Last" and "Last First" (charting uses both)
        full_fl = f"{first} {last}".strip()
        full_lf = f"{last} {first}".strip()
        mapping[full_fl.lower()] = pid
        mapping[full_lf.lower()] = pid

    _name_to_id_cache = mapping
    return _name_to_id_cache


def _resolve_player_id(charting_name: str) -> str | None:
    """Resolve a charting-project player name to a Sackmann player_id.

    Tries exact match first, then fuzzy matching with rapidfuzz.
    """
    name_map = _build_name_to_id_map()
    if not name_map:
        return None

    normalized = charting_name.strip().lower()

    # Exact match
    if normalized in name_map:
        return name_map[normalized]

    # The charting project often uses "First_Last" with underscores
    normalized_space = normalized.replace("_", " ")
    if normalized_space in name_map:
        return name_map[normalized_space]

    # Fuzzy match against all known names
    result = process.extractOne(
        normalized_space,
        name_map.keys(),
        scorer=fuzz.token_sort_ratio,
        score_cutoff=_FUZZY_MATCH_THRESHOLD,
    )
    if result is not None:
        matched_name, _score, _idx = result
        return name_map[matched_name]

    return None


# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------

def _read_charting_csv(filename: str) -> pd.DataFrame | None:
    """Read a charting stats CSV, returning None if not found."""
    path = CHARTING_DIR / filename
    if not path.exists():
        logger.debug("Charting file not found: %s", path)
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", filename, exc)
        return None


def load_overview_stats() -> pd.DataFrame | None:
    """Load match-level aggregate stats (serve/return/winners/UE).

    Columns: match_id, player, set, serve_pts, aces, dfs, first_in,
    first_won, second_in, second_won, bk_pts, bp_saved, return_pts,
    return_pts_won, winners, winners_fh, winners_bh, unforced,
    unforced_fh, unforced_bh.
    """
    return _read_charting_csv("charting-m-stats-Overview.csv")


def load_return_outcomes() -> pd.DataFrame | None:
    """Load return outcome breakdown stats."""
    return _read_charting_csv("charting-m-stats-ReturnOutcomes.csv")


def load_shot_types() -> pd.DataFrame | None:
    """Load shot type distribution stats.

    Columns: match_id, player, row, shots, pt_ending, winners,
    induced_forced, unforced, serve_return, shots_in_pts_won,
    shots_in_pts_lost.
    """
    return _read_charting_csv("charting-m-stats-ShotTypes.csv")


def load_rally_stats() -> pd.DataFrame | None:
    """Load rally length stats.

    Columns: match_id, server, returner, row, pts, pl1_won, pl1_winners,
    pl1_forced, pl1_unforced, pl2_won, pl2_winners, pl2_forced,
    pl2_unforced.  The 'row' column contains rally-length buckets:
    "Total", "1-3", "4-6", "7-9", "10+".
    """
    return _read_charting_csv("charting-m-stats-Rally.csv")


def load_net_points() -> pd.DataFrame | None:
    """Load net point stats.

    Columns: match_id, player, row, net_pts, pts_won, net_winner,
    induced_forced, net_unforced, passed_at_net,
    passing_shot_induced_forced, total_shots.
    """
    return _read_charting_csv("charting-m-stats-NetPoints.csv")


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _safe_ratio(numerator: float, denominator: float) -> float:
    """Divide with NaN for zero/missing denominator."""
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def _compute_player_rolling_stats(
    overview: pd.DataFrame | None,
    rally: pd.DataFrame | None,
    shot_types: pd.DataFrame | None,
    net_points: pd.DataFrame | None,
) -> dict[str, dict[str, float]]:
    """Compute per-player rolling averages from charting data.

    Returns a dict keyed by Sackmann player_id with stat dicts as values.
    Players that cannot be mapped to a Sackmann ID are excluded.
    """
    # Per-player per-match stat accumulator: pid -> list of per-match dicts
    player_matches: dict[str, list[dict[str, float]]] = {}

    # ------------------------------------------------------------------
    # 1. Overview: winners, UEs, aces, serve stats  (one row per player
    #    per set; we only use the "Total" rows)
    # ------------------------------------------------------------------
    if overview is not None and "player" in overview.columns:
        totals = overview[overview["set"] == "Total"] if "set" in overview.columns else overview
        for _, row in totals.iterrows():
            pid = _resolve_player_id(str(row["player"]))
            if pid is None:
                continue

            serve_pts = float(row.get("serve_pts", 0) or 0)
            return_pts = float(row.get("return_pts", 0) or 0)
            total_pts = serve_pts + return_pts
            winners = float(row.get("winners", 0) or 0)
            unforced = float(row.get("unforced", 0) or 0)
            aces = float(row.get("aces", 0) or 0)

            # first_won on serve can proxy for service winners
            # (aces are already separated; first_won includes aces +
            # unreturnable serves, so "service winners" ~ first_won - aces
            # is a reasonable proxy when serve_winners isn't explicit)
            first_won = float(row.get("first_won", 0) or 0)
            svc_winners = first_won  # aces + unreturnable first serves

            match_id = row.get("match_id", "")
            entry: dict[str, float] = {
                "_match_id": hash(match_id),
                "winner_rate": _safe_ratio(winners, total_pts),
                "unforced_error_rate": _safe_ratio(unforced, total_pts),
                "first_strike_aggression": _safe_ratio(aces + svc_winners, serve_pts),
            }
            player_matches.setdefault(pid, []).append(entry)

    # ------------------------------------------------------------------
    # 2. Rally stats: average rally length per match
    #    Each row has server/returner; "Total" row has overall counts.
    #    Rally length buckets: "1-3", "4-6", "7-9", "10+"
    # ------------------------------------------------------------------
    if rally is not None and "row" in rally.columns:
        # Compute weighted average rally length from bucket rows per match
        for mid, grp in rally.groupby("match_id"):
            # Get total points from the "Total" row
            total_row = grp[grp["row"] == "Total"]
            if total_row.empty:
                continue
            total_pts = float(total_row.iloc[0].get("pts", 0) or 0)
            if total_pts == 0:
                continue

            # Compute weighted rally length from bucket rows
            bucket_rows = grp[grp["row"].isin(["1-3", "4-6", "7-9", "10+"])]
            weighted_sum = 0.0
            bucket_total = 0.0
            # Midpoints for each bucket
            midpoints = {"1-3": 2.0, "4-6": 5.0, "7-9": 8.0, "10+": 12.0}
            for _, brow in bucket_rows.iterrows():
                bucket_name = brow["row"]
                pts_in_bucket = float(brow.get("pts", 0) or 0)
                if pts_in_bucket > 0 and bucket_name in midpoints:
                    weighted_sum += midpoints[bucket_name] * pts_in_bucket
                    bucket_total += pts_in_bucket

            avg_rally = weighted_sum / bucket_total if bucket_total > 0 else np.nan

            # Assign to both server and returner
            server = str(total_row.iloc[0].get("server", ""))
            returner = str(total_row.iloc[0].get("returner", ""))
            for pname in (server, returner):
                pid = _resolve_player_id(pname)
                if pid is None:
                    continue
                # Try to merge into existing entry for this match
                entries = player_matches.setdefault(pid, [])
                merged = False
                for e in entries:
                    if e.get("_match_id") == hash(mid):
                        e["avg_rally_length"] = avg_rally
                        merged = True
                        break
                if not merged:
                    entries.append({
                        "_match_id": hash(mid),
                        "avg_rally_length": avg_rally,
                    })

    # ------------------------------------------------------------------
    # 3. Net points: net approach rate
    #    Use "NetPoints" rows (not "Approach" sub-rows).
    # ------------------------------------------------------------------
    if net_points is not None and "player" in net_points.columns:
        net_total = net_points[net_points["row"] == "NetPoints"] if "row" in net_points.columns else net_points
        for _, row in net_total.iterrows():
            pid = _resolve_player_id(str(row["player"]))
            if pid is None:
                continue

            net_pts_val = float(row.get("net_pts", 0) or 0)
            total_shots = float(row.get("total_shots", 0) or 0)
            net_rate = _safe_ratio(net_pts_val, total_shots)

            mid = row.get("match_id", "")
            entries = player_matches.setdefault(pid, [])
            merged = False
            for e in entries:
                if e.get("_match_id") == hash(mid):
                    e["net_approach_rate"] = net_rate
                    merged = True
                    break
            if not merged:
                entries.append({
                    "_match_id": hash(mid),
                    "net_approach_rate": net_rate,
                })

    # ------------------------------------------------------------------
    # 4. Aggregate: mean across all charted matches per player
    # ------------------------------------------------------------------
    stat_keys = [
        "avg_rally_length",
        "net_approach_rate",
        "winner_rate",
        "unforced_error_rate",
        "first_strike_aggression",
    ]

    result: dict[str, dict[str, float]] = {}
    for pid, entries in player_matches.items():
        if not entries:
            continue
        agg: dict[str, float] = {}
        for key in stat_keys:
            values = [e[key] for e in entries if key in e and pd.notna(e.get(key))]
            agg[key] = float(np.mean(values)) if values else np.nan
        # Only include players with at least one valid stat
        if any(pd.notna(v) for v in agg.values()):
            result[pid] = agg

    return result


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def _save_cache(stats: dict[str, dict[str, float]]) -> None:
    """Persist computed stats to disk cache."""
    CHARTING_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Convert NaN to None for JSON serialization
    serializable: dict[str, dict[str, Any]] = {}
    for pid, pstats in stats.items():
        serializable[pid] = {
            k: (None if pd.isna(v) else v) for k, v in pstats.items()
        }
    CHARTING_CACHE_FILE.write_text(json.dumps(serializable, indent=2))
    logger.info("Saved charting stats cache for %d players", len(stats))


def _load_cache() -> dict[str, dict[str, float]] | None:
    """Load stats from disk cache if available."""
    if not CHARTING_CACHE_FILE.exists():
        return None
    try:
        raw = json.loads(CHARTING_CACHE_FILE.read_text())
        # Convert None back to NaN
        stats: dict[str, dict[str, float]] = {}
        for pid, pstats in raw.items():
            stats[pid] = {
                k: (np.nan if v is None else float(v)) for k, v in pstats.items()
            }
        logger.info("Loaded charting stats cache (%d players)", len(stats))
        return stats
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Corrupt charting cache, will rebuild: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_charting_stats(force_rebuild: bool = False) -> dict[str, dict[str, float]]:
    """Build or load the player charting stats dictionary.

    This is the main entry point for initializing charting data. Call this
    once during pipeline setup; then use get_player_charting_stats() for
    per-player lookups.

    Args:
        force_rebuild: If True, re-parse CSVs even if a cache exists.

    Returns:
        Dict mapping player_id -> stat dict. May be empty if the charting
        repo is unavailable or contains no parseable data.
    """
    global _player_stats_cache

    # Return in-memory cache if available
    if _player_stats_cache is not None and not force_rebuild:
        return _player_stats_cache

    # Try disk cache
    if not force_rebuild:
        cached = _load_cache()
        if cached is not None:
            _player_stats_cache = cached
            return _player_stats_cache

    # Clone/update and build from CSVs
    clone_or_update_charting_repo()

    if not CHARTING_DIR.exists():
        logger.warning("Charting repo not available; returning empty stats")
        _player_stats_cache = {}
        return _player_stats_cache

    logger.info("Building charting stats from CSV files...")
    overview = load_overview_stats()
    rally = load_rally_stats()
    shot_types = load_shot_types()
    net_points = load_net_points()

    stats = _compute_player_rolling_stats(overview, rally, shot_types, net_points)

    if stats:
        _save_cache(stats)
        logger.info("Built charting stats for %d players", len(stats))
    else:
        logger.warning("No charting stats could be computed (files may be missing or empty)")

    _player_stats_cache = stats
    return _player_stats_cache


def get_player_charting_stats(player_id: str) -> dict[str, float] | None:
    """Return charting-derived stats for a player, or None if not charted.

    Stats returned (all floats, may contain NaN for individual fields):
        - avg_rally_length: average rally count per point
        - net_approach_rate: frequency of net approaches
        - winner_rate: winners per point played
        - unforced_error_rate: unforced errors per point played
        - first_strike_aggression: (aces + service winners) / serve points

    If build_charting_stats() has not been called, this will trigger a
    build (with caching) on first access.
    """
    global _player_stats_cache

    if _player_stats_cache is None:
        build_charting_stats()

    assert _player_stats_cache is not None
    return _player_stats_cache.get(str(player_id))


def invalidate_cache() -> None:
    """Clear both in-memory and on-disk caches, forcing a rebuild on next access."""
    global _player_stats_cache, _name_to_id_cache
    _player_stats_cache = None
    _name_to_id_cache = None
    if CHARTING_CACHE_FILE.exists():
        CHARTING_CACHE_FILE.unlink()
        logger.info("Charting cache invalidated")
