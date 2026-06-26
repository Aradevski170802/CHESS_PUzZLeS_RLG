"""
Quality-filters the enriched puzzle DataFrame.
Removes duplicates, degenerate puzzles, and outliers before training or serving.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def clean(df: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    before = len(df)

    # Drop rows with missing required fields
    df = df.dropna(subset=["FEN", "Moves", "Rating"])

    # Remove duplicates by PuzzleId
    df = df.drop_duplicates(subset="PuzzleId")

    # Remove puzzles with absurdly high rating deviation (poorly calibrated)
    df = df[df["RatingDeviation"] <= 150]

    # Remove puzzles that have never been played (no quality signal)
    df = df[df["NbPlays"] > 0]

    # Remove puzzles with negative popularity (players disliked them)
    df = df[df["Popularity"] >= 0]

    # Require at least one move in the solution
    df = df[df["Moves"].str.strip().str.len() > 0]

    df = df.reset_index(drop=True)

    if verbose:
        removed = before - len(df)
        logger.info("Cleaned: %d → %d puzzles  (removed %d)", before, len(df), removed)

    return df


def report(df: pd.DataFrame) -> None:
    """Print a summary of the dataset."""
    print(f"Total puzzles : {len(df):,}")
    print(f"Rating range  : {df['Rating'].min()} – {df['Rating'].max()}")
    print()

    print("By difficulty tier:")
    tier_counts = df["DifficultyTier"].value_counts().reindex(
        [label for _, _, label in __import__("src.data.puzzle_loader", fromlist=["DIFFICULTY_TIERS"]).DIFFICULTY_TIERS],
        fill_value=0,
    )
    for tier, count in tier_counts.items():
        pct = count / len(df) * 100
        print(f"  {tier:<15} {count:>8,}  ({pct:.1f}%)")
    print()

    print("By primary category (top 15):")
    cat_counts = df["PrimaryCategory"].value_counts().head(15)
    for cat, count in cat_counts.items():
        pct = count / len(df) * 100
        print(f"  {cat:<25} {count:>8,}  ({pct:.1f}%)")
