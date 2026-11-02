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

Counts are floats, not ints: evidence-scaled priors from game analysis
(player_profiler.profile_to_bandit_priors) and discounting both produce
fractional pseudo-counts, and np.random.beta accepts any positive reals.

Non-stationarity — discounted Thompson Sampling
────────────────────────────────────────────────
A player who practises improves, so old outcomes should count for less
than recent ones. With discount γ < 1, every update first shrinks every
arm's evidence toward that arm's own prior:

    α ← α₀ + γ(α − α₀),   β ← β₀ + γ(β − β₀)

(the discounted-TS scheme of Raj & Kalyani, 2017). The effective memory is
about 1/(1−γ) attempts — γ = 0.98 remembers roughly the last 50. γ = 1.0
(the default) is the original stationary bandit, unchanged.

Selection probabilities (propensities)
───────────────────────────────────────
selection_probabilities() estimates P(arm i is selected) under the current
posterior by Monte Carlo. Logging it with every served puzzle is what makes
future off-policy evaluation (inverse propensity scoring) possible: without
it, data collected under Thompson Sampling can never be used to fairly score
a different policy.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from src.data.puzzle_loader import WEAKNESS_CATEGORIES


@dataclass
class ArmState:
    alpha: float = 1.0
    beta:  float = 1.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def sample(self, rng: np.random.Generator | None = None) -> float:
        if rng is not None:
            return float(rng.beta(self.alpha, self.beta))
        return float(np.random.beta(self.alpha, self.beta))

    def to_dict(self) -> dict:
        return {"alpha": round(self.alpha, 4), "beta": round(self.beta, 4)}


class ThompsonBandit:
    """
    Thompson Sampling bandit — one arm per weakness category.
    Picks the arm with the lowest sampled solve rate (= biggest weakness).
    """

    def __init__(
        self,
        priors: dict[str, tuple[float, float]] | None = None,
        *,
        discount: float = 1.0,
    ):
        if not 0.0 < discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        self.discount = discount
        self.arms: dict[str, ArmState] = {c: ArmState() for c in WEAKNESS_CATEGORIES}
        if priors:
            for cat, (a, b) in priors.items():
                if cat in self.arms:
                    self.arms[cat] = ArmState(alpha=max(1.0, float(a)), beta=max(1.0, float(b)))
        # The anchor each arm decays back toward under discounting.
        self.prior_anchor: dict[str, tuple[float, float]] = {
            c: (arm.alpha, arm.beta) for c, arm in self.arms.items()
        }
        self.history:     list[dict] = []
        self.streak:      int = 0
        self.best_streak: int = 0

    # ── Core ──────────────────────────────────────────────────────────────

    def select_one(self, rng: np.random.Generator | None = None) -> str:
        """Pick the weakest category by Thompson Sampling."""
        samples = {cat: arm.sample(rng) for cat, arm in self.arms.items()}
        return min(samples, key=samples.get)

    def selection_probabilities(
        self, n_samples: int = 2000, rng: np.random.Generator | None = None,
    ) -> dict[str, float]:
        """Monte Carlo estimate of P(each arm is chosen by select_one())."""
        rng = rng or np.random.default_rng()
        cats = list(self.arms)
        a = np.array([self.arms[c].alpha for c in cats])
        b = np.array([self.arms[c].beta for c in cats])
        draws = rng.beta(a, b, size=(n_samples, len(cats)))
        counts = np.bincount(draws.argmin(axis=1), minlength=len(cats))
        return {c: float(n) / n_samples for c, n in zip(cats, counts)}

    def update(self, category: str, solved: bool) -> None:
        arm = self.arms.get(category)
        if arm is None:
            return
        if self.discount < 1.0:
            self._decay()
        if solved:
            arm.alpha += 1.0
            self.streak += 1
            self.best_streak = max(self.streak, self.best_streak)
        else:
            arm.beta += 1.0
            self.streak = 0
        self.history.append({"cat": category, "solved": solved})

    def _decay(self) -> None:
        g = self.discount
        for cat, arm in self.arms.items():
            a0, b0 = self.prior_anchor.get(cat, (1.0, 1.0))
            arm.alpha = a0 + g * (arm.alpha - a0)
            arm.beta  = b0 + g * (arm.beta - b0)

    # ── Statistics ────────────────────────────────────────────────────────

    def solve_rate(self, category: str) -> float:
        """Expected solve rate for a category (lower = weaker)."""
        arm = self.arms.get(category)
        return arm.mean if arm else 0.5

    def weakness_map(self) -> dict[str, float]:
        """Solve rate per category. Lower means the player is weaker there."""
        return {cat: arm.mean for cat, arm in self.arms.items()}

    def top_weaknesses(self, n: int = 5) -> list[dict]:
        """Categories sorted by weakness (lowest solve rate first)."""
        return [
            {"category": cat, "weakness": round(rate, 4)}
            for cat, rate in sorted(self.weakness_map().items(), key=lambda x: x[1])[:n]
        ]

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
            "priorAnchor": {c: [round(a, 4), round(b, 4)] for c, (a, b) in self.prior_anchor.items()},
            "discount":    self.discount,
            "history":     self.history,
            "streak":      self.streak,
            "best_streak": self.best_streak,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ThompsonBandit":
        b = cls(discount=float(d.get("discount", 1.0)))
        for cat, v in d.get("arms", {}).items():
            if cat in b.arms:
                b.arms[cat] = ArmState(alpha=float(v["alpha"]), beta=float(v["beta"]))
        # State saved before anchors existed decays toward the uniform prior.
        for cat, v in d.get("priorAnchor", {}).items():
            if cat in b.prior_anchor:
                b.prior_anchor[cat] = (float(v[0]), float(v[1]))
        b.history     = d.get("history", [])
        b.streak      = d.get("streak", 0)
        b.best_streak = d.get("best_streak", 0)
        return b
