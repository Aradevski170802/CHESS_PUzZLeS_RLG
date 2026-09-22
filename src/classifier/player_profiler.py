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

Opportunity-normalised weakness (preferred when available)
────────────────────────────────────────────────────────────
stockfish_analyzer now records every *critical* position — one clearly-best
move — with the tactic of that move and whether the player found it. That
gives a real per-category hit rate:

    hit rate_k = found_k / opportunities_k

weighted so recent games count more (half-life RECENCY_HALF_LIFE_GAMES) and
bullet games count less (time pressure, not tactical knowledge), then shrunk
toward the player's own overall hit rate (empirical Bayes, strength
SHRINK_STRENGTH) so a category seen twice cannot swing to 0 % or 100 %.
Bandit priors and IRT priors are built from these counts, with confidence
proportional to how much evidence each category actually has.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.classifier.stockfish_analyzer import GameAnalysis, MoveError
from src.data.puzzle_loader import WEAKNESS_CATEGORIES

logger = logging.getLogger(__name__)

NORMS_PATH = Path(__file__).resolve().parents[1] / "data" / "category_norms.json"
_NORMS_CACHE: dict = {}


def load_category_norms(path: Path = NORMS_PATH) -> Optional[dict]:
    """
    Population category norms fitted on the research cohort
    (scripts/research/evaluate_weakness_models.py --write-norms): for each
    category, how much harder or easier its critical positions are than average,
    as a logit offset, plus the cross-validated shrinkage strength. Aggregate
    statistics only — no player data. Returns None if not fitted yet, in which
    case profiling falls back to shrinking toward the player's own overall rate.
    """
    key = str(path)
    if key not in _NORMS_CACHE:
        try:
            _NORMS_CACHE[key] = json.loads(Path(path).read_text("utf-8"))
        except (OSError, ValueError):
            _NORMS_CACHE[key] = None
    return _NORMS_CACHE[key]


RECENCY_HALF_LIFE_GAMES = 25
TIME_CLASS_WEIGHT = {"bullet": 0.5, "blitz": 0.85, "rapid": 1.0, "daily": 1.0, "unknown": 0.85}
SHRINK_STRENGTH = 4.0       # pseudo-opportunities pulling each category to the overall rate
MIN_OPPORTUNITIES = 10      # below this, fall back to the rule-based scores for priors
PRIOR_BUDGET = 10.0         # max pseudo-count a game-derived Beta prior may carry
GAME_EVIDENCE_WEIGHT = 0.5  # one game opportunity = half a puzzle attempt in the IRT prior
IRT_DELTA_SD = 0.6
WP_PER_CP_NEAR_EVEN = 0.092 # slope of the win-% curve at 0 cp, for legacy profiles

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
    errors_by_category: dict  = field(default_factory=dict)  # tactic category → count
    avg_cp_loss:        float = 0.0   # average centipawn loss per player move
    avg_wp_loss:        float = 0.0   # average win-% points lost per player move

    # ── Opportunity-normalised evidence (see module docstring) ────────────
    # category → {"hits": weighted, "n": weighted, "raw_n": int, "rate": shrunk}
    opportunity_stats:  dict  = field(default_factory=dict)
    overall_hit_rate:   float = 0.0
    games_by_time_class: dict = field(default_factory=dict)

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
        Accuracy percentage (0–100) from Lichess's move-accuracy formula,
        103.1668·e^(−0.04354·x) − 3.1669, where x is the WIN-% points lost.

        This used to be fed the average CENTIPAWN loss — a unit error that
        reported ~15 % accuracy for a typical ACPL of 40, so the
        "accuracy < 70" rule below fired for nearly every player. Profiles
        without win-% data fall back to the curve's slope at equality
        (≈ 0.092 win-% per centipawn).
        """
        x = self.avg_wp_loss
        if x <= 0 and self.avg_cp_loss > 0:
            x = self.avg_cp_loss * WP_PER_CP_NEAR_EVEN
        if x <= 0:
            return 100.0
        accuracy = 103.1668 * math.exp(-0.04354 * x) - 3.1669
        return max(0.0, min(100.0, accuracy))

    @property
    def has_opportunity_data(self) -> bool:
        return sum(s.get("raw_n", 0) for s in self.opportunity_stats.values()) >= MIN_OPPORTUNITIES

    def empirical_weakness_scores(self) -> dict[str, float]:
        """1 − shrunk hit rate per category (0 = strong, 1 = weak)."""
        base = self.overall_hit_rate or 0.5
        return {
            cat: round(1.0 - self.opportunity_stats.get(cat, {}).get("rate", base), 4)
            for cat in WEAKNESS_CATEGORIES
        }


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
    profile.errors_by_category = dict(Counter(e.category for e in all_errors))

    cp_losses = [a.avg_cp_loss for a in successful if a.avg_cp_loss > 0]
    profile.avg_cp_loss = sum(cp_losses) / len(cp_losses) if cp_losses else 0.0
    wp_losses = [a.avg_wp_loss for a in successful if getattr(a, "avg_wp_loss", 0) > 0]
    profile.avg_wp_loss = sum(wp_losses) / len(wp_losses) if wp_losses else 0.0
    profile.games_by_time_class = dict(Counter(getattr(a, "time_class", "unknown") for a in successful))

    # ── Opportunity-normalised hit rates ──────────────────────────────────
    profile.opportunity_stats, profile.overall_hit_rate = aggregate_opportunities(
        successful, load_category_norms())

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


def aggregate_opportunities(
    analyses: list[GameAnalysis],
    norms: Optional[dict] = None,
) -> tuple[dict[str, dict], float]:
    """
    Weighted per-category hit counts over critical positions.

    Games are ranked newest-first (by end_time; list order when unknown) and
    weighted 0.5^(rank / RECENCY_HALF_LIFE_GAMES) × TIME_CLASS_WEIGHT.

    Each category's hit rate is then shrunk toward an EXPECTED rate:
      * with population norms: σ(logit(player's overall rate) + offset_k),
        the rate typical for category k at this player's level (hierarchical
        empirical Bayes). Some motifs are harder for everyone; without this,
        "this category is hard" was read as "this player is weak here".
      * without norms: the player's overall rate (SHRINK_STRENGTH pseudo-counts).

    Returns ({category: {"hits", "n", "raw_n", "rate", "expected"}}, overall_hit_rate).
    """
    ordered = sorted(
        enumerate(analyses),
        key=lambda ia: (ia[1].end_time is None, -(ia[1].end_time or 0), ia[0]),
    )
    hits: dict[str, float] = defaultdict(float)
    n: dict[str, float] = defaultdict(float)
    raw_n: dict[str, int] = defaultdict(int)
    for rank, (_, a) in enumerate(ordered):
        w = 0.5 ** (rank / RECENCY_HALF_LIFE_GAMES)
        w *= TIME_CLASS_WEIGHT.get(getattr(a, "time_class", "unknown"), 0.85)
        for opp in getattr(a, "opportunities", []):
            n[opp.category] += w
            hits[opp.category] += w if opp.hit else 0.0
            raw_n[opp.category] += 1

    total_n = sum(n.values())
    overall = sum(hits.values()) / total_n if total_n > 0 else 0.0
    offsets = (norms or {}).get("offsets", {})
    strength = float((norms or {}).get("shrink_strength", SHRINK_STRENGTH))
    stats = {}
    for cat in n:
        expected = overall
        if cat in offsets and 0.0 < overall < 1.0:
            expected = 1.0 / (1.0 + math.exp(-(_logit(overall) + float(offsets[cat]))))
        rate = (hits[cat] + strength * expected) / (n[cat] + strength)
        stats[cat] = {"hits": round(hits[cat], 4), "n": round(n[cat], 4),
                      "raw_n": raw_n[cat], "rate": round(rate, 4),
                      "expected": round(expected, 4)}
    return stats, round(overall, 4)


def profile_to_bandit_priors(profile: PlayerProfile) -> dict[str, tuple[float, float]]:
    """
    Initial Beta(α, β) per category for the Thompson Sampling bandit.

    With opportunity data (preferred)
    ──────────────────────────────────
    The prior mean is the category's shrunk hit rate p̂_k, and its strength
    grows with the evidence actually observed for that category, capped at
    PRIOR_BUDGET so puzzle sessions can still override it:

        C_k = min(PRIOR_BUDGET, n_k + 2)
        α = 1 + p̂_k · C_k,   β = 1 + (1 − p̂_k) · C_k

    A category never seen in the player's games gets C_k = 2 around the
    player's overall rate — a weak prior, honestly reflecting no evidence.
    (This replaces the old round((1 − w)·10) rule, which gave every category
    the same confidence regardless of evidence and quantised to steps of 0.1.)

    Without opportunity data (engine-free fallback profiles)
    ────────────────────────────────────────────────────────
    Legacy mapping from weakness_scores with α + β = 10, without rounding.
    """
    stats = getattr(profile, "opportunity_stats", None) or {}
    has_opps = sum(s.get("raw_n", 0) for s in stats.values()) >= MIN_OPPORTUNITIES
    priors: dict[str, tuple[float, float]] = {}
    if has_opps:
        base = getattr(profile, "overall_hit_rate", 0.5) or 0.5
        for cat in WEAKNESS_CATEGORIES:
            s = stats.get(cat, {})
            rate = s.get("rate", base)
            strength = min(PRIOR_BUDGET, s.get("n", 0.0) + 2.0)
            priors[cat] = (round(1.0 + rate * strength, 4), round(1.0 + (1.0 - rate) * strength, 4))
        return priors

    for cat, weakness in profile.weakness_scores.items():
        solve_rate = min(0.9, max(0.1, 1.0 - weakness))
        priors[cat] = (round(solve_rate * PRIOR_BUDGET, 4), round((1.0 - solve_rate) * PRIOR_BUDGET, 4))
    return priors


def _logit(p: float) -> float:
    p = min(0.98, max(0.02, p))
    return math.log(p / (1.0 - p))


def profile_to_irt_prior(profile: PlayerProfile) -> dict[str, tuple[float, float]]:
    """
    Prior N(mean, sd²) for each category offset δ_k of the IRT learner.

    With opportunity data: δ_k is centred on how much better or worse the
    player converts category-k opportunities than their own average (a
    logit difference, clipped to ±1.5), and its uncertainty shrinks with
    the evidence — one game opportunity counts as GAME_EVIDENCE_WEIGHT of a
    puzzle attempt, because finding a move in a game and solving a flagged
    puzzle are related but not identical tasks.

    Without it: a weak prior from the rule-based weakness scores.
    """
    stats = getattr(profile, "opportunity_stats", None) or {}
    has_opps = sum(s.get("raw_n", 0) for s in stats.values()) >= MIN_OPPORTUNITIES
    prior: dict[str, tuple[float, float]] = {}
    if has_opps:
        base = getattr(profile, "overall_hit_rate", 0.5) or 0.5
        for cat in WEAKNESS_CATEGORIES:
            s = stats.get(cat)
            if not s:
                continue
            rate = s["rate"]
            # δ is a PERSONAL offset (the IRT's puzzle ratings already price in
            # how hard a motif is for everyone), so compare against what is
            # typical for this category at this player's level when norms exist.
            mu = max(-1.5, min(1.5, _logit(rate) - _logit(s.get("expected", base))))
            info = GAME_EVIDENCE_WEIGHT * s["n"] * rate * (1.0 - rate)
            sd = 1.0 / math.sqrt(1.0 / IRT_DELTA_SD ** 2 + info)
            prior[cat] = (round(mu, 4), round(sd, 4))
        return prior

    for cat, weakness in getattr(profile, "weakness_scores", {}).items():
        mu = 0.5 * max(-1.5, min(1.5, _logit(1.0 - weakness)))
        prior[cat] = (round(mu, 4), 0.55)
    return prior


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
