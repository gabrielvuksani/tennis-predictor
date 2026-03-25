"""Jeff Sackmann tennis_atp data loader.

Loads, cleans, and unifies ATP match data from the JeffSackmann/tennis_atp repository.
This is the primary historical data source (~1968-2024, match stats from 1991+).
"""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from tennis_predictor.config import RAW_DIR, SACKMANN_DIR

REPO_URL = "https://github.com/JeffSackmann/tennis_atp.git"

# Canonical column schema for match data
MATCH_COLUMNS = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level",
    "tourney_date", "match_num", "winner_id", "winner_seed", "winner_entry",
    "winner_name", "winner_hand", "winner_ht", "winner_ioc", "winner_age",
    "winner_rank", "winner_rank_points", "loser_id", "loser_seed", "loser_entry",
    "loser_name", "loser_hand", "loser_ht", "loser_ioc", "loser_age",
    "loser_rank", "loser_rank_points", "score", "best_of", "round", "minutes",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms",
    "w_bpSaved", "w_bpFaced", "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon",
    "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced",
]

# Round ordering for within-tournament sort
ROUND_ORDER = {
    "R128": 1, "R64": 2, "R32": 3, "R16": 4,
    "QF": 5, "SF": 6, "F": 7, "RR": 3, "BR": 6,
    "ER": 0, "R12": 1, "R24": 2,
}


def clone_or_update_repo() -> Path:
    """Clone the JeffSackmann tennis_atp repo, or pull if already cloned."""
    if SACKMANN_DIR.exists() and (SACKMANN_DIR / ".git").exists():
        print("Updating tennis_atp repository...")
        subprocess.run(
            ["git", "pull", "--quiet"],
            cwd=SACKMANN_DIR,
            check=True,
            capture_output=True,
        )
    else:
        print("Cloning tennis_atp repository...")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(SACKMANN_DIR)],
            check=True,
            capture_output=True,
        )
    return SACKMANN_DIR


def load_players() -> pd.DataFrame:
    """Load the player metadata file."""
    path = SACKMANN_DIR / "atp_players.csv"
    df = pd.read_csv(path, dtype={"player_id": str})
    df["dob"] = pd.to_datetime(df["dob"], format="%Y%m%d", errors="coerce")
    return df


def load_rankings(start_year: int = 1985) -> pd.DataFrame:
    """Load all ranking files into a unified DataFrame."""
    ranking_files = sorted(SACKMANN_DIR.glob("atp_rankings_*.csv"))
    frames = []
    for f in ranking_files:
        try:
            df = pd.read_csv(f, dtype={"player": str})
            df.columns = ["ranking_date", "rank", "player_id", "points"][:len(df.columns)]
            df["ranking_date"] = pd.to_datetime(df["ranking_date"], format="%Y%m%d", errors="coerce")
            if start_year:
                df = df[df["ranking_date"].dt.year >= start_year]
            frames.append(df)
        except Exception as e:
            print(f"Warning: Could not load {f.name}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("ranking_date").reset_index(drop=True)


def _parse_score(score: str) -> dict:
    """Parse a match score string into structured data."""
    if pd.isna(score) or not score:
        return {
            "n_sets": np.nan, "retirement": False, "walkover": False, "tiebreaks": 0,
            "deciding_set": False, "straight_sets": False,
            "sets_won_winner": np.nan, "sets_won_loser": np.nan, "tiebreak_count": 0,
        }

    score = str(score).strip()
    retirement = any(tag in score.upper() for tag in ["RET", "ABD", "DEF", "UNP"])
    walkover = "W/O" in score.upper() or "WO" in score.upper()

    # Count tiebreaks (indicated by parentheses)
    tiebreaks = score.count("(")

    # Count sets
    sets = [s.strip() for s in score.split() if "-" in s]
    n_sets = len(sets)

    # Parse set-level details: count sets won by each side
    sets_won_winner = 0
    sets_won_loser = 0
    for s in sets:
        # Strip tiebreak score in parentheses, e.g. "7-6(4)" -> "7-6"
        clean = s.split("(")[0]
        parts = clean.split("-")
        if len(parts) == 2:
            try:
                games_a = int(parts[0])
                games_b = int(parts[1])
                if games_a > games_b:
                    sets_won_winner += 1
                elif games_b > games_a:
                    sets_won_loser += 1
            except ValueError:
                pass

    return {
        "n_sets": n_sets if n_sets > 0 else np.nan,
        "retirement": retirement,
        "walkover": walkover,
        "tiebreaks": tiebreaks,
        "deciding_set": bool(
            n_sets > 0 and sets_won_loser > 0
            and sets_won_winner == sets_won_loser + 1
            and not retirement
        ),
        "straight_sets": bool(
            n_sets > 0 and sets_won_loser == 0 and not retirement
        ),
        "sets_won_winner": sets_won_winner if n_sets > 0 else np.nan,
        "sets_won_loser": sets_won_loser if n_sets > 0 else np.nan,
        "tiebreak_count": tiebreaks,
    }


def load_matches(
    start_year: int = 1991,
    end_year: int | None = None,
    include_qual_chall: bool = True,
    include_futures: bool = False,
) -> pd.DataFrame:
    """Load all ATP match data into a unified, temporally ordered DataFrame.

    Args:
        start_year: First year to include (1991+ for match stats).
        end_year: Last year to include (None = all available).
        include_qual_chall: Include qualifier and challenger matches.
        include_futures: Include futures matches.

    Returns:
        DataFrame with all matches sorted by date and round, with a unique match_id.
    """
    if not SACKMANN_DIR.exists():
        clone_or_update_repo()

    # Collect match files
    file_patterns = ["atp_matches_[0-9]*.csv"]
    if include_qual_chall:
        file_patterns.append("atp_matches_qual_chall_[0-9]*.csv")
    if include_futures:
        file_patterns.append("atp_matches_futures_[0-9]*.csv")

    all_files = []
    for pattern in file_patterns:
        all_files.extend(sorted(SACKMANN_DIR.glob(pattern)))

    frames = []
    for f in tqdm(all_files, desc="Loading match files"):
        try:
            year_str = f.stem.split("_")[-1]
            year = int(year_str) if year_str.isdigit() else None
            if year is not None:
                if start_year and year < start_year:
                    continue
                if end_year and year > end_year:
                    continue

            df = pd.read_csv(
                f,
                dtype={"winner_id": str, "loser_id": str, "winner_seed": str, "loser_seed": str},
                low_memory=False,
            )

            # Ensure we only keep known columns (some files have extras)
            valid_cols = [c for c in MATCH_COLUMNS if c in df.columns]
            df = df[valid_cols]
            frames.append(df)
        except Exception as e:
            print(f"Warning: Could not load {f.name}: {e}")

    if not frames:
        raise FileNotFoundError(
            f"No match files found in {SACKMANN_DIR}. Run clone_or_update_repo() first."
        )

    # Supplement with TML Database (2025-2026 data that Sackmann is missing)
    try:
        from tennis_predictor.data.tml import load_tml_matches
        tml = load_tml_matches()
        if len(tml) > 0:
            valid_cols = [c for c in MATCH_COLUMNS if c in tml.columns]
            tml_clean = tml[valid_cols]
            frames.append(tml_clean)
    except Exception as e:
        print(f"Note: TML data unavailable ({e})")

    matches = pd.concat(frames, ignore_index=True)

    # Parse dates
    matches["tourney_date"] = pd.to_datetime(
        matches["tourney_date"], format="%Y%m%d", errors="coerce"
    )

    # Parse scores
    score_info = matches["score"].apply(_parse_score).apply(pd.Series)
    matches = pd.concat([matches, score_info], axis=1)

    # Remove walkovers (no match actually played)
    matches = matches[~matches["walkover"]].copy()

    # Create round ordering for temporal sort
    matches["round_order"] = matches["round"].map(ROUND_ORDER).fillna(0).astype(int)

    # Temporal sort: by tournament date, then by round order within tournament
    matches = matches.sort_values(
        ["tourney_date", "tourney_id", "round_order", "match_num"]
    ).reset_index(drop=True)

    # Create unique match ID
    matches["match_id"] = (
        matches["tourney_id"].astype(str) + "_" + matches.index.astype(str)
    )

    # Numeric conversions
    for col in ["winner_rank", "loser_rank", "winner_rank_points", "loser_rank_points",
                "winner_ht", "loser_ht", "draw_size", "best_of", "minutes"]:
        if col in matches.columns:
            matches[col] = pd.to_numeric(matches[col], errors="coerce")

    stat_cols = [c for c in matches.columns if c.startswith(("w_", "l_"))]
    for col in stat_cols:
        matches[col] = pd.to_numeric(matches[col], errors="coerce")

    print(f"Loaded {len(matches):,} matches ({matches['tourney_date'].min().year}"
          f"-{matches['tourney_date'].max().year})")

    return matches


def create_pairwise_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """Convert winner/loser format to player1/player2 format with target label.

    For each match, creates a row where player1 is randomly assigned (to avoid
    any systematic ordering bias). Target y=1 means player1 won.

    CRITICAL: This randomization must use a fixed seed for reproducibility,
    and must NOT leak information about who won.
    """
    rng = np.random.RandomState(42)
    n = len(matches)

    # Random coin flip: if 1, player1=winner; if 0, player1=loser
    swap = rng.randint(0, 2, size=n).astype(bool)

    rows = pd.DataFrame(index=matches.index)
    rows["match_id"] = matches["match_id"]
    rows["tourney_id"] = matches["tourney_id"]
    rows["tourney_name"] = matches["tourney_name"]
    rows["tourney_date"] = matches["tourney_date"]
    rows["tourney_level"] = matches["tourney_level"]
    rows["surface"] = matches["surface"]
    rows["round"] = matches["round"]
    rows["best_of"] = matches["best_of"]
    rows["draw_size"] = matches["draw_size"]
    rows["minutes"] = matches["minutes"]
    rows["retirement"] = matches["retirement"]
    rows["n_sets"] = matches["n_sets"]

    # Assign player1 and player2 based on random swap
    p1_fields = ["id", "name", "hand", "ht", "ioc", "age", "rank", "rank_points",
                 "seed", "entry"]
    p2_fields = p1_fields.copy()

    for field in p1_fields:
        w_col = f"winner_{field}"
        l_col = f"loser_{field}"
        if w_col in matches.columns and l_col in matches.columns:
            rows[f"p1_{field}"] = np.where(swap, matches[w_col], matches[l_col])
            rows[f"p2_{field}"] = np.where(swap, matches[l_col], matches[w_col])

    # Target: 1 if player1 won, 0 if player2 won
    rows["y"] = swap.astype(int)

    return rows
