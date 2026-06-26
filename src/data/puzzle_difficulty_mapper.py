"""
Maps Lichess puzzle ratings to ELO-bracket difficulty tiers and provides
utilities for aligning puzzle difficulty to a player's current ELO.
"""

from __future__ import annotations

from src.data.puzzle_loader import DIFFICULTY_TIERS, rating_to_tier


def elo_to_recommended_tier(player_elo: int) -> list[str]:
    """
    Returns the 1-2 difficulty tiers most appropriate for a player.

    Puzzles slightly above the player's ELO (up to +200) are proven to be
    the most effective for learning (zone of proximal development).
    """
    target_lo = player_elo
    target_hi = player_elo + 200

    tiers = set()
    for lo, hi, label in DIFFICULTY_TIERS:
        # Include any tier that overlaps [target_lo, target_hi]
        if lo <= target_hi and hi >= target_lo:
            tiers.add(label)

    # Return in order defined in DIFFICULTY_TIERS
    ordered = [label for _, _, label in DIFFICULTY_TIERS if label in tiers]
    return ordered


def tier_rating_range(tier: str) -> tuple[int, int]:
    for lo, hi, label in DIFFICULTY_TIERS:
        if label == tier:
            return lo, hi
    raise ValueError(f"Unknown tier: {tier!r}")


def describe_tiers() -> None:
    print(f"{'Tier':<15} {'Rating range'}")
    print("-" * 30)
    for lo, hi, label in DIFFICULTY_TIERS:
        hi_str = str(hi) if hi < 9999 else "+"
        print(f"{label:<15} {lo}–{hi_str}")
