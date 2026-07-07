"""
Aggregates Stockfish analysis results into a structured player profile.

The profile serves two purposes:
  1. Shown to the player on the UI (opening repertoire, accuracy, error breakdown)
  2. Seeds the Thompson Sampling bandit with informed initial priors
     so it doesn't start from scratch on the first puzzle session.

How priors are set
──────────────────
Every category starts at Beta(1,1) — a flat prior meaning "no information".
After game analysis we adjust based on:
  - Phase of most errors (middlegame errors → tactical weakness)
  - Blunder rate per game (high blunder rate → hanging piece / fork weakness)
  - Endgame error rate → specific endgame category weakness

The "prior confidence" is kept modest (α + β = 10) so the bandit
can update quickly from real puzzle sessions.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from src.classifier.stockfish_analyzer import GameAnalysis, MoveError
from src.data.puzzle_loader import WEAKNESS_CATEGORIES

logger = logging.getLogger(__name__)

# Map broad error phases to specific weakness categories
# (used to initialise bandit priors from game analysis)
_MIDDLEGAME_CATS = [
    "Fork", "Pin", "Skewer", "Discovered Attack",
    "Sacrifice", "Deflection", "Hanging Piece", "King Safety",
]
_ENDGAME_CATS = [
    "Endgame", "Rook Endgame", "Queen Endgame",
    "Pawn Endgame", "Knight Endgame", "Bishop Endgame",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OpeningStat:
    eco:          str
    family:       str
    games_played: int
    wins:         int
    losses:       int
    draws:        int

    @property
    def win_rate(self) -> float:
        return self.wins / self.games_played if self.games_played else 0.0

    @property
    def result_str(self) -> str:
        return f"{self.wins}W / {self.losses}L / {self.draws}D"


@dataclass
class PlayerProfile:
    username:       str
    estimated_elo:  int

    # ── Opening repertoire ────────────────────────────────────────────────
    top_openings_white: list[OpeningStat] = field(default_factory=list)
    top_openings_black: list[OpeningStat] = field(default_factory=list)

    # ── Error breakdown ───────────────────────────────────────────────────
    blunders_total:     int   = 0
    mistakes_total:     int   = 0
    inaccuracies_total: int   = 0
    errors_by_phase:    dict  = field(default_factory=dict)  # phase → count
    avg_cp_loss:        float = 0.0   # average centipawn loss per player move

    # ── Game record ───────────────────────────────────────────────────────
    games_analysed: int = 0
    games_won:      int = 0
    games_lost:     int = 0
    games_drawn:    int = 0

    # ── Weakness scores (→ bandit priors) ─────────────────────────────────
    # 0.0 = very strong in this category, 1.0 = very weak
    weakness_scores: dict[str, float] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return self.games_won / self.games_analysed if self.games_analysed else 0.0

    @property
    def accuracy_estimate(self) -> float:
        """
        Rough accuracy percentage (0–100) based on average centipawn loss.
        Mirrors Chess.com's own accuracy formula (approximate).
        """
        import math
        if self.avg_cp_loss <= 0:
            return 100.0
        accuracy = 103.1668 * math.exp(-0.04354 * self.avg_cp_loss) - 3.1669
        return max(0.0, min(100.0, accuracy))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_profile(
    analyses:     list[GameAnalysis],
    username:     str,
    estimated_elo: int,
) -> PlayerProfile:
    """
    Build a PlayerProfile from a list of GameAnalysis objects.

    Parameters
    ----------
    analyses      : output of stockfish_analyzer.analyze_games_parallel()
    username      : Chess.com username (for display)
    estimated_elo : player's current ELO (used for difficulty tier mapping)

    Returns
    -------
    PlayerProfile ready to display on the UI and seed the bandit.
    """
    profile = PlayerProfile(username=username, estimated_elo=estimated_elo)

    successful = [a for a in analyses if not a.failed]
    if not successful:
        logger.warning("No successful analyses for %s — returning empty profile", username)
        profile.weakness_scores = {cat: 0.5 for cat in WEAKNESS_CATEGORIES}
        return profile

    # ── Game results ──────────────────────────────────────────────────────
    profile.games_analysed = len(successful)
    profile.games_won      = sum(1 for a in successful if a.player_won is True)
    profile.games_lost     = sum(1 for a in successful if a.player_won is False)
    profile.games_drawn    = sum(1 for a in successful if a.player_won is None)

    # ── Errors ────────────────────────────────────────────────────────────
    all_errors: list[MoveError] = [e for a in successful for e in a.errors]

    profile.blunders_total     = sum(1 for e in all_errors if e.severity == "blunder")
    profile.mistakes_total     = sum(1 for e in all_errors if e.severity == "mistake")
    profile.inaccuracies_total = sum(1 for e in all_errors if e.severity == "inaccuracy")
    profile.errors_by_phase    = dict(Counter(e.phase for e in all_errors))

    cp_losses = [a.avg_cp_loss for a in successful if a.avg_cp_loss > 0]
    profile.avg_cp_loss = sum(cp_losses) / len(cp_losses) if cp_losses else 0.0

    # ── Opening repertoire ────────────────────────────────────────────────
    white_stats: dict[str, dict] = defaultdict(
        lambda: {"eco": "", "games": 0, "wins": 0, "losses": 0, "draws": 0}
    )
    black_stats: dict[str, dict] = defaultdict(
        lambda: {"eco": "", "games": 0, "wins": 0, "losses": 0, "draws": 0}
    )

    for analysis in successful:
        family = analysis.opening_family or "Unknown"
        target = white_stats if analysis.player_color == "white" else black_stats
        target[family]["eco"]    = analysis.opening_eco
        target[family]["games"] += 1
        if analysis.player_won is True:
            target[family]["wins"]   += 1
        elif analysis.player_won is False:
            target[family]["losses"] += 1
        else:
            target[family]["draws"]  += 1

    def _to_opening_stats(stats_dict) -> list[OpeningStat]:
        return sorted(
            [
                OpeningStat(
                    eco          = v["eco"],
                    family       = k,
                    games_played = v["games"],
                    wins         = v["wins"],
                    losses       = v["losses"],
                    draws        = v["draws"],
                )
                for k, v in stats_dict.items()
            ],
            key=lambda s: s.games_played,
            reverse=True,
        )[:10]  # top 10 openings

    profile.top_openings_white = _to_opening_stats(white_stats)
    profile.top_openings_black = _to_opening_stats(black_stats)

    # ── Weakness scores → bandit priors ───────────────────────────────────
    profile.weakness_scores = _compute_weakness_scores(profile, all_errors)

    return profile


def profile_to_bandit_priors(profile: PlayerProfile) -> dict[str, tuple[int, int]]:
    """
    Convert weakness scores to Beta distribution (alpha, beta) initial values
    for the Thompson Sampling bandit.

    Logic
    ─────
    weakness_score ∈ [0.0, 1.0]
      0.0 = strong (high expected solve rate)
      1.0 = weak   (low expected solve rate)

    We set α + β = 10 (modest confidence — enough to guide early exploration
    without overriding the real puzzle-session data too aggressively).

      expected_solve_rate = 1 - weakness_score
      alpha = round(expected_solve_rate × 10)    ← expected successes
      beta  = 10 - alpha                          ← expected failures

    Example:
      weakness_score = 0.8 → solve_rate = 0.2 → Beta(2, 8)
        (strong prior: player struggles here, give more puzzles in this category)
      weakness_score = 0.2 → solve_rate = 0.8 → Beta(8, 2)
        (strong prior: player is good here, deprioritise)
      weakness_score = 0.5 → solve_rate = 0.5 → Beta(5, 5)
        (neutral — no information)
    """
    priors: dict[str, tuple[int, int]] = {}
    for cat, weakness in profile.weakness_scores.items():
        solve_rate = 1.0 - weakness
        alpha = max(1, round(solve_rate * 10))
        beta  = max(1, 10 - alpha)
        priors[cat] = (alpha, beta)
    return priors


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_weakness_scores(
    profile: PlayerProfile,
    all_errors: list[MoveError],
) -> dict[str, float]:
    """
    Derive per-category weakness scores (0–1) from the game analysis.
    All categories start at 0.5 (neutral). We push them up or down
    based on observed error patterns.
    """
    scores = {cat: 0.5 for cat in WEAKNESS_CATEGORIES}

    total_errors = max(1, len(all_errors))
    middlegame_n = sum(1 for e in all_errors if e.phase == "middlegame")
    endgame_n    = sum(1 for e in all_errors if e.phase == "endgame")
    blunder_n    = sum(1 for e in all_errors if e.severity == "blunder")

    middlegame_rate = middlegame_n / total_errors
    endgame_rate    = endgame_n    / total_errors
    blunder_per_game = blunder_n / max(1, profile.games_analysed)

    # Many middlegame errors → tactical weaknesses
    if middlegame_rate > 0.5:
        boost = min(0.25, middlegame_rate * 0.3)
        for cat in _MIDDLEGAME_CATS:
            scores[cat] = min(0.9, scores[cat] + boost)

    # Many endgame errors → endgame category weaknesses
    if endgame_rate > 0.25:
        boost = min(0.30, endgame_rate * 0.4)
        for cat in _ENDGAME_CATS:
            scores[cat] = min(0.9, scores[cat] + boost)

    # High blunder rate → very likely to miss hanging pieces / forks
    if blunder_per_game >= 2:
        for cat in ["Hanging Piece", "Fork", "Pin"]:
            scores[cat] = min(0.92, scores[cat] + 0.25)
    elif blunder_per_game >= 1:
        for cat in ["Hanging Piece", "Fork"]:
            scores[cat] = min(0.85, scores[cat] + 0.15)

    # Low overall accuracy → general tactical weakness
    accuracy = profile.accuracy_estimate
    if accuracy < 70:
        for cat in _MIDDLEGAME_CATS[:4]:   # Fork, Pin, Skewer, Discovered Attack
            scores[cat] = min(0.90, scores[cat] + 0.10)

    return scores
