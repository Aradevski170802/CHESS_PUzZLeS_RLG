"""
Loads the Lichess puzzle CSV, normalises it, and enriches each puzzle with
structured difficulty tiers and primary theme categories for use throughout
the recommendation pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Theme taxonomy
# ---------------------------------------------------------------------------

# Maps every Lichess theme tag to a human-readable category used by the
# recommender.  A puzzle can belong to multiple categories.
THEME_CATEGORIES: dict[str, list[str]] = {
    # --- Tactical motifs ---
    "fork":               ["Fork"],
    "pin":                ["Pin"],
    "skewer":             ["Skewer"],
    "discoveredAttack":   ["Discovered Attack"],
    "doubleCheck":        ["Discovered Attack"],   # double-check is a discovered-attack variant
    "hangingPiece":       ["Hanging Piece"],
    "trappedPiece":       ["Hanging Piece"],
    "sacrifice":          ["Sacrifice"],
    "deflection":         ["Deflection"],
    "attraction":         ["Attraction"],
    "interference":       ["Interference"],
    "intermezzo":         ["Interference"],
    "clearance":          ["Clearance"],
    "quietMove":          ["Quiet Move"],
    "zugzwang":           ["Zugzwang"],
    "xRayAttack":         ["X-Ray Attack"],
    "capturingDefender":  ["Deflection"],
    "exposedKing":        ["King Safety"],
    "kingsideAttack":     ["King Safety"],
    "queensideAttack":    ["King Safety"],

    # --- Mating patterns ---
    "mate":               ["Mating Pattern"],
    "mateIn1":            ["Mating Pattern"],
    "mateIn2":            ["Mating Pattern"],
    "mateIn3":            ["Mating Pattern"],
    "mateIn4":            ["Mating Pattern"],
    "mateIn5":            ["Mating Pattern"],
    "smotheredMate":      ["Mating Pattern"],
    "bodensMate":         ["Mating Pattern"],
    "arabianMate":        ["Mating Pattern"],
    "anastasiaMate":      ["Mating Pattern"],
    "backRankMate":       ["Mating Pattern"],
    "doubleBishopMate":   ["Mating Pattern"],
    "hookMate":           ["Mating Pattern"],
    "epauletteMate":      ["Mating Pattern"],
    "killBoxMate":        ["Mating Pattern"],
    "dovetailMate":       ["Mating Pattern"],
    "vukovicMate":        ["Mating Pattern"],

    # --- Endgame types ---
    "rookEndgame":        ["Endgame", "Rook Endgame"],
    "queenEndgame":       ["Endgame", "Queen Endgame"],
    "pawnEndgame":        ["Endgame", "Pawn Endgame"],
    "bishopEndgame":      ["Endgame", "Bishop Endgame"],
    "knightEndgame":      ["Endgame", "Knight Endgame"],
    "queenRookEndgame":   ["Endgame", "Queen Endgame", "Rook Endgame"],
    "bishopVsKnight":     ["Endgame"],

    # --- Special moves ---
    "promotion":          ["Promotion"],
    "underPromotion":     ["Promotion"],
    "enPassant":          ["En Passant"],
    "castling":           ["Castling"],

    # --- Game phase (informational, not used for weakness tracking) ---
    "opening":            ["Opening"],
    "middlegame":         ["Middlegame"],
    "endgame":            ["Endgame"],

    # --- Puzzle length / effort (informational) ---
    "oneMove":            [],
    "short":              [],
    "long":               [],
    "veryLong":           [],

    # --- Outcome quality (informational) ---
    "crushing":           [],
    "advantage":          [],
    "equality":           [],
}

# The ordered set of categories exposed to the recommender's weakness tracker.
# Only these are used for bandit arms — the others are metadata.
WEAKNESS_CATEGORIES: list[str] = [
    "Fork",
    "Pin",
    "Skewer",
    "Discovered Attack",
    "Hanging Piece",
    "Sacrifice",
    "Deflection",
    "Attraction",
    "Interference",
    "Clearance",
    "Quiet Move",
    "Zugzwang",
    "X-Ray Attack",
    "King Safety",
    "Mating Pattern",
    "Endgame",
    "Rook Endgame",
    "Queen Endgame",
    "Pawn Endgame",
    "Bishop Endgame",
    "Knight Endgame",
    "Promotion",
    "En Passant",
]

# ---------------------------------------------------------------------------
# Difficulty tiers
# ---------------------------------------------------------------------------

DIFFICULTY_TIERS: list[tuple[int, int, str]] = [
    (0,    999,  "Beginner"),
    (1000, 1199, "Easy"),
    (1200, 1499, "Intermediate"),
    (1500, 1799, "Advanced"),
    (1800, 1999, "Hard"),
    (2000, 2199, "Expert"),
    (2200, 9999, "Master"),
]


def rating_to_tier(rating: int) -> str:
    for lo, hi, label in DIFFICULTY_TIERS:
        if lo <= rating <= hi:
            return label
    return "Master"


# ---------------------------------------------------------------------------
# Puzzle length classification (from Moves column)
# ---------------------------------------------------------------------------

def moves_to_length(moves: str) -> str:
    """Returns oneMove / short / long / veryLong based on move count."""
    n = len(moves.strip().split())
    if n <= 1:
        return "oneMove"
    if n <= 3:
        return "short"
    if n <= 7:
        return "long"
    return "veryLong"


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_puzzles(
    csv_path: str | Path,
    *,
    min_rating: int = 0,
    max_rating: int = 9999,
    min_popularity: int = -100,
    min_plays: int = 0,
    themes: Optional[list[str]] = None,
    difficulty_tiers: Optional[list[str]] = None,
    sample: Optional[int] = None,
    chunksize: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load and enrich the Lichess puzzle CSV.

    Parameters
    ----------
    csv_path         : path to lichess_db_puzzle.csv
    min_rating       : lower rating bound (inclusive)
    max_rating       : upper rating bound (inclusive)
    min_popularity   : minimum popularity score (-100 to 100)
    min_plays        : minimum number of plays
    themes           : filter to puzzles that contain ANY of these WEAKNESS_CATEGORIES
    difficulty_tiers : filter to these difficulty tier labels
    sample           : if set, return a random sample of this many rows
    chunksize        : if set, read in chunks (useful for memory-constrained envs)

    Returns
    -------
    Enriched DataFrame with columns:
        PuzzleId, FEN, Moves, Rating, RatingDeviation, Popularity, NbPlays,
        Themes, GameUrl, OpeningTags,
        ThemeList       — list of raw Lichess tags
        Categories      — list of WEAKNESS_CATEGORY labels
        PrimaryCategory — single most-specific category (or "General")
        DifficultyTier  — Beginner / Easy / … / Master
        MoveLength      — oneMove / short / long / veryLong
    """
    csv_path = Path(csv_path)
    logger.info("Loading puzzles from %s", csv_path)

    dtype = {
        "PuzzleId": str,
        "FEN": str,
        "Moves": str,
        "Rating": "int32",
        "RatingDeviation": "int16",
        "Popularity": "int8",
        "NbPlays": "int32",
        "Themes": str,
        "GameUrl": str,
        "OpeningTags": str,
    }

    if chunksize:
        chunks = []
        for chunk in pd.read_csv(csv_path, dtype=dtype, chunksize=chunksize):
            chunk = _enrich(chunk)
            chunk = _filter(chunk, min_rating, max_rating, min_popularity,
                            min_plays, themes, difficulty_tiers)
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(csv_path, dtype=dtype)
        df = _enrich(df)
        df = _filter(df, min_rating, max_rating, min_popularity,
                     min_plays, themes, difficulty_tiers)

    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)

    logger.info("Loaded %d puzzles after filtering", len(df))
    return df


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse raw theme string into a list
    df["ThemeList"] = df["Themes"].fillna("").str.split()

    # Map to weakness categories (deduplicated, order preserved)
    df["Categories"] = df["ThemeList"].apply(_tags_to_categories)

    # Primary category = first WEAKNESS category found, else "General"
    # Phase tags (Opening, Middlegame) appear in Categories but are excluded here.
    weakness_set = set(WEAKNESS_CATEGORIES)
    df["PrimaryCategory"] = df["Categories"].apply(
        lambda cats: next((c for c in cats if c in weakness_set), "General")
    )

    # Difficulty tier from rating
    df["DifficultyTier"] = df["Rating"].apply(rating_to_tier)

    # Move length
    df["MoveLength"] = df["Moves"].fillna("").apply(moves_to_length)

    return df


def _tags_to_categories(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        for cat in THEME_CATEGORIES.get(tag, []):
            if cat not in seen:
                seen.add(cat)
                result.append(cat)
    return result


def _filter(
    df: pd.DataFrame,
    min_rating: int,
    max_rating: int,
    min_popularity: int,
    min_plays: int,
    themes: Optional[list[str]],
    difficulty_tiers: Optional[list[str]],
) -> pd.DataFrame:
    mask = (
        df["Rating"].between(min_rating, max_rating)
        & (df["Popularity"] >= min_popularity)
        & (df["NbPlays"] >= min_plays)
    )
    if difficulty_tiers:
        mask &= df["DifficultyTier"].isin(difficulty_tiers)
    if themes:
        theme_set = set(themes)
        mask &= df["Categories"].apply(lambda cats: bool(set(cats) & theme_set))
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def save_processed(df: pd.DataFrame, out_path: str | Path) -> None:
    """Save enriched DataFrame to Parquet for fast subsequent loads."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Saved %d puzzles to %s", len(df), out_path)


def load_processed(parquet_path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)


def split_by_category(df: pd.DataFrame, out_dir: str | Path) -> dict[str, int]:
    """
    Save one Parquet file per WEAKNESS_CATEGORY into out_dir.
    Returns a dict of {category: puzzle_count}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {}

    for cat in WEAKNESS_CATEGORIES:
        subset = df[df["Categories"].apply(lambda cats: cat in cats)].copy()
        if subset.empty:
            summary[cat] = 0
            continue
        safe_name = cat.replace(" ", "_").replace("-", "_")
        subset.to_parquet(out_dir / f"{safe_name}.parquet", index=False)
        summary[cat] = len(subset)
        logger.info("  %-25s  %d puzzles", cat, len(subset))

    return summary


def get_puzzles_for_player(
    df: pd.DataFrame,
    player_rating: int,
    target_categories: list[str],
    *,
    rating_window: int = 200,
    per_category: int = 50,
) -> pd.DataFrame:
    """
    Return a ranked puzzle pool for a player targeting specific weak categories.

    Puzzles are filtered to within ±rating_window of player_rating, then
    capped at per_category per category so the pool is balanced.
    """
    lo = max(0, player_rating - rating_window)
    hi = player_rating + rating_window
    pool = df[df["Rating"].between(lo, hi)]

    frames = []
    for cat in target_categories:
        subset = pool[pool["Categories"].apply(lambda cats: cat in cats)]
        if not subset.empty:
            frames.append(subset.head(per_category))

    if not frames:
        return pd.DataFrame()

    return (
        pd.concat(frames)
        .drop_duplicates(subset="PuzzleId")
        .sort_values("Popularity", ascending=False)
        .reset_index(drop=True)
    )
