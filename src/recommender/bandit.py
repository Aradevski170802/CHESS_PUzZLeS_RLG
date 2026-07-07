"""
Thompson Sampling multi-armed bandit for adaptive puzzle recommendation.

One arm per WEAKNESS_CATEGORY (23 arms total).
Each arm maintains a Beta(α, β) distribution:
  α = puzzles solved in this category + prior
  β = puzzles failed in this category + prior

Selection rule: sample θ_i ~ Beta(α_i, β_i) for every arm, then pick
the arm with the LOWEST θ_i.  Lowest sampled solve rate = most likely
current weakness = highest training value for the player.

Seeding from game analysis (profile_to_bandit_priors):
  weakness_score 0.8 → Beta(2, 8)  (player struggles → prioritise)
  weakness_score 0.5 → Beta(5, 5)  (neutral, equal prior)
  weakness_score 0.2 → Beta(8, 2)  (player strong → de-prioritise)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from src.data.puzzle_loader import WEAKNESS_CATEGORIES


@dataclass
class ArmState:
    alpha: int = 1
    beta:  int = 1

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def sample(self) -> float:
        return float(np.random.beta(self.alpha, self.beta))

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta}


class ThompsonBandit:
    """
    Thompson Sampling bandit — one arm per weakness category.
    Picks the arm with the lowest sampled solve rate (= biggest weakness).
    """

    def __init__(self, priors: dict[str, tuple[int, int]] | None = None):
        self.arms: dict[str, ArmState] = {c: ArmState() for c in WEAKNESS_CATEGORIES}
        if priors:
            for cat, (a, b) in priors.items():
                if cat in self.arms:
                    self.arms[cat] = ArmState(alpha=max(1, a), beta=max(1, b))
        self.history:     list[dict] = []
        self.streak:      int = 0
        self.best_streak: int = 0

    # ── Core ──────────────────────────────────────────────────────────────

    def select_one(self) -> str:
        """Pick the weakest category by Thompson Sampling."""
        samples = {cat: arm.sample() for cat, arm in self.arms.items()}
        return min(samples, key=samples.get)

    def update(self, category: str, solved: bool) -> None:
        arm = self.arms.get(category)
        if arm is None:
            return
        if solved:
            arm.alpha += 1
            self.streak += 1
            self.best_streak = max(self.streak, self.best_streak)
        else:
            arm.beta += 1
            self.streak = 0
        self.history.append({"cat": category, "solved": solved})

    # ── Statistics ────────────────────────────────────────────────────────

    def solve_rate(self, category: str) -> float:
        """Expected solve rate for a category (lower = weaker)."""
        arm = self.arms.get(category)
        return arm.mean if arm else 0.5

    def weakness_map(self) -> dict[str, float]:
        """Solve rate per category. Lower means the player is weaker there."""
        return {cat: arm.mean for cat, arm in self.arms.items()}

    def top_weaknesses(self, n: int = 5) -> list[tuple[str, float]]:
        """Categories sorted by weakness (lowest solve rate first)."""
        return sorted(self.weakness_map().items(), key=lambda x: x[1])[:n]

    def session_accuracy(self) -> float:
        if not self.history:
            return 0.0
        return sum(1 for h in self.history if h["solved"]) / len(self.history)

    def puzzles_played(self) -> int:
        return len(self.history)

    # ── Persistence ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "arms":        {c: s.to_dict() for c, s in self.arms.items()},
            "history":     self.history,
            "streak":      self.streak,
            "best_streak": self.best_streak,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ThompsonBandit":
        b = cls()
        for cat, v in d.get("arms", {}).items():
            if cat in b.arms:
                b.arms[cat] = ArmState(alpha=v["alpha"], beta=v["beta"])
        b.history     = d.get("history", [])
        b.streak      = d.get("streak", 0)
        b.best_streak = d.get("best_streak", 0)
        return b
