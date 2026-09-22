"""
Difficulty-aware weakness model: an online Bayesian item-response (IRT) learner.

Why this exists
────────────────
ThompsonBandit (bandit.py) counts raw solves per category, so failing a
2200-rated puzzle and failing a 1000-rated puzzle move the posterior by the
same amount. Categories that happen to be served harder puzzles then look like
weaknesses. It also cannot say how hard the next puzzle should be. And the
Glicko-2 fitter (analysis/difficulty_fitter.py) has the opposite gap: it knows
about difficulty but keeps one global rating per player, so it cannot express
"1600 at forks, 1200 at rook endgames" — the very thing this system is about.

This module joins the two with one model:

    P(solve | player, puzzle) = σ( θ + δ_k − b )

    θ    the player's overall puzzle ability            (what Glicko-2 tracks)
    δ_k  the player's offset in tactical category k      (what the bandit wants)
    b    the puzzle's difficulty, from its rating         (known, with uncertainty)

All three live on the logit scale, which is exactly Glicko's internal scale:
b = (rating − 1500) / 173.7178, so a 400-point rating gap is a 10:1 odds ratio.
This is a Rasch-type model (Elo/IRT family) with a category-specific ability
offset, as used in adaptive learning systems (Pelánek, 2016).

Inference
──────────
The player's state is a Gaussian N(m, S) over w = [θ, δ_1 … δ_23] (24-dim,
full covariance, so evidence about θ and δ_k is shared correctly). Each
attempt is an online Laplace / extended-Kalman step on the logistic
likelihood — in one dimension this reduces exactly to Glicko's update
(the κ factor below is Glicko's g(φ)):

    x      = e_θ + e_k                 (indicator of the terms in the logit)
    z̄      = xᵀm − b,   q = xᵀSx
    κ      = 1 / √(1 + 3 var_b/π²)     discount for the puzzle's own uncertainty
    p      = σ(κ z̄)                    Glicko's E
    h      = κ² p(1 − p)               Glicko's 1/v
    S'     = S − h·(Sx)(Sx)ᵀ / (1 + h q)
    m'     = m + κ(y − p)·Sx / (1 + h q)

With a single parameter this is term-for-term Glicko: S' = 1/(1/φ² + 1/v)
and m' = μ + φ'²·g(φ_j)(s − E). tests/test_irt_model.py checks that
equivalence numerically against difficulty_fitter.glicko2_update().
predict() additionally folds the player's own uncertainty q into κ, which
gives better-calibrated probabilities for new players.

Skill drift (the player improving) is modelled as a Gaussian random walk:
S ← S + diag(q_θ, q_δ, …) before every update, the multivariate analogue of
Glicko inflating RD between rating periods.

Decisions
──────────
  select_category()  Thompson Sampling on δ: draw w ~ N(m, S) and return the
                     category with the LOWEST sampled δ_k — the likeliest
                     weakness *after* adjusting for puzzle difficulty.
  target_rating()    the puzzle rating at which the predicted solve
                     probability equals p_target (default 0.65), so practice
                     stays challenging but not demoralising.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from src.data.puzzle_loader import WEAKNESS_CATEGORIES

LOGIT_SCALE = 173.7178             # rating points per logit — Glicko-2's own constant
DEFAULT_RATING = 1500.0
UNSEEDED_THETA_SD = 350.0 / LOGIT_SCALE   # RD 350: nothing known
SEEDED_THETA_SD = 250.0 / LOGIT_SCALE     # Chess.com Elo known, but game and puzzle
                                          # ratings are not on the same scale
DEFAULT_DELTA_SD = 0.6             # prior spread of category offsets (~100 rating pts)
DEFAULT_PUZZLE_RD = 75.0           # Lichess puzzles: rated from many attempts
MINED_PUZZLE_RD = 300.0            # own-game puzzles: heuristic rating only
DEFAULT_P_TARGET = 0.65
DEFAULT_DRIFT_THETA = 0.02 ** 2    # per-attempt random-walk variance, logits²
DEFAULT_DRIFT_DELTA = 0.04 ** 2

_N = len(WEAKNESS_CATEGORIES)
_IDX = {c: i + 1 for i, c in enumerate(WEAKNESS_CATEGORIES)}   # 0 is θ

# Tactic families for the optional structured prior: categories in a family
# share a family-level effect, so evidence about one (e.g. Pin) moves the
# others (Skewer, X-Ray, Discovered Attack) a little too.
TACTIC_FAMILIES: dict[str, list[str]] = {
    "line":          ["Pin", "Skewer", "X-Ray Attack", "Discovered Attack"],
    "double_attack": ["Fork", "Hanging Piece", "Deflection", "Attraction",
                      "Interference", "Clearance", "Sacrifice"],
    "king_attack":   ["Mating Pattern", "King Safety"],
    "endgame":       ["Endgame", "Rook Endgame", "Queen Endgame", "Pawn Endgame",
                      "Bishop Endgame", "Knight Endgame", "Zugzwang"],
    "pawn":          ["Promotion", "En Passant"],
    "quiet":         ["Quiet Move"],
}
DEFAULT_FAMILY_SHARE = 0.45   # fraction of prior δ variance that is family-level


def family_covariance(sd: float = DEFAULT_DELTA_SD, share: float = DEFAULT_FAMILY_SHARE) -> np.ndarray:
    """
    Prior covariance of the 23 δ's under δ_k = f_family(k) + ε_k:
    Var δ_k = sd², Cov(δ_k, δ_j) = share·sd² when k and j share a family.
    """
    fam_of = {c: f for f, cs in TACTIC_FAMILIES.items() for c in cs}
    cov = np.eye(_N) * sd ** 2
    for i, a in enumerate(WEAKNESS_CATEGORIES):
        for j, b in enumerate(WEAKNESS_CATEGORIES):
            if i != j and fam_of.get(a) is not None and fam_of.get(a) == fam_of.get(b):
                cov[i, j] = share * sd ** 2
    return cov


def rating_to_logit(rating: float) -> float:
    return (float(rating) - DEFAULT_RATING) / LOGIT_SCALE


def logit_to_rating(value: float) -> float:
    return DEFAULT_RATING + LOGIT_SCALE * float(value)


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _kappa(var: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * var / math.pi ** 2)


@dataclass
class IRTLearner:
    """Per-player Gaussian posterior over [θ, δ_1 … δ_23]."""

    mean: np.ndarray = field(default_factory=lambda: np.zeros(_N + 1))
    cov: np.ndarray = field(default_factory=lambda: np.diag(
        [UNSEEDED_THETA_SD ** 2] + [DEFAULT_DELTA_SD ** 2] * _N))
    p_target: float = DEFAULT_P_TARGET
    drift_theta: float = DEFAULT_DRIFT_THETA
    drift_delta: float = DEFAULT_DRIFT_DELTA
    n_updates: int = 0

    # ── Construction ─────────────────────────────────────────────────────

    @classmethod
    def new(
        cls,
        elo: Optional[float] = None,
        delta_prior: Optional[dict[str, tuple[float, float]]] = None,
        *,
        family_prior: bool = False,
        family_share: float = DEFAULT_FAMILY_SHARE,
        **kwargs,
    ) -> "IRTLearner":
        """
        elo           the player's rating if known (seeds θ, with wide uncertainty)
        delta_prior   {category: (mean, sd)} — e.g. from game analysis via
                      player_profiler.profile_to_irt_prior(); others default to N(0, 0.6²)
        family_prior  correlate δ's within a tactic family (TACTIC_FAMILIES), so
                      evidence is shared between related motifs
        """
        mean = np.zeros(_N + 1)
        sds = np.full(_N + 1, DEFAULT_DELTA_SD)
        if elo:
            mean[0] = rating_to_logit(elo)
            sds[0] = SEEDED_THETA_SD
        else:
            sds[0] = UNSEEDED_THETA_SD
        for cat, (mu, sd) in (delta_prior or {}).items():
            i = _IDX.get(cat)
            if i is not None:
                mean[i] = float(mu)
                sds[i] = max(1e-3, float(sd))
        cov = np.diag(sds ** 2)
        if family_prior:
            # Scale the family correlation structure to each category's own sd.
            corr = family_covariance(1.0, family_share)
            d = sds[1:]
            cov[1:, 1:] = corr * np.outer(d, d)
        return cls(mean=mean, cov=cov, **kwargs)

    # ── Prediction ───────────────────────────────────────────────────────

    def _x(self, category: str) -> Optional[np.ndarray]:
        i = _IDX.get(category)
        if i is None:
            return None
        x = np.zeros(_N + 1)
        x[0] = 1.0
        x[i] = 1.0
        return x

    def predict(self, category: str, puzzle_rating: float,
                puzzle_rd: float = DEFAULT_PUZZLE_RD) -> float:
        """Predicted probability that the player solves this puzzle."""
        x = self._x(category)
        if x is None:
            x = np.zeros(_N + 1)
            x[0] = 1.0
        z = float(x @ self.mean) - rating_to_logit(puzzle_rating)
        s2 = float(x @ self.cov @ x) + (puzzle_rd / LOGIT_SCALE) ** 2
        return _sigmoid(_kappa(s2) * z)

    # ── Learning ─────────────────────────────────────────────────────────

    def update(self, category: str, puzzle_rating: float, solved: bool,
               puzzle_rd: float = DEFAULT_PUZZLE_RD) -> float:
        """
        Absorb one attempt. Returns the probability the model predicted
        *before* seeing the outcome, so callers can log it for calibration.
        Unknown categories update θ only.
        """
        self.cov = self.cov + np.diag(
            [self.drift_theta] + [self.drift_delta] * _N)
        x = self._x(category)
        if x is None:
            x = np.zeros(_N + 1)
            x[0] = 1.0
        predicted = self.predict(category, puzzle_rating, puzzle_rd)
        Sx = self.cov @ x
        q = float(x @ Sx)
        z = float(x @ self.mean) - rating_to_logit(puzzle_rating)
        kappa = _kappa((puzzle_rd / LOGIT_SCALE) ** 2)
        p = _sigmoid(kappa * z)
        h = kappa ** 2 * p * (1.0 - p)
        y = 1.0 if solved else 0.0
        denom = 1.0 + h * q
        self.mean = self.mean + (kappa * (y - p) / denom) * Sx
        self.cov = self.cov - (h / denom) * np.outer(Sx, Sx)
        self.cov = 0.5 * (self.cov + self.cov.T)      # keep it exactly symmetric
        self.n_updates += 1
        return predicted

    # ── Decisions ────────────────────────────────────────────────────────

    def _sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        try:
            chol = np.linalg.cholesky(self.cov)
        except np.linalg.LinAlgError:
            chol = np.linalg.cholesky(self.cov + 1e-9 * np.eye(_N + 1))
        z = rng.standard_normal((size, _N + 1))
        return self.mean + z @ chol.T

    def select_category(
        self,
        rng: Optional[np.random.Generator] = None,
        allowed: Optional[Iterable[str]] = None,
    ) -> str:
        """Thompson Sampling on the category offsets: lowest sampled δ wins."""
        rng = rng or np.random.default_rng()
        deltas = self._sample(rng)[0, 1:]
        cats = WEAKNESS_CATEGORIES
        if allowed is not None:
            allowed = set(allowed)
            best = min((c for c in cats if c in allowed),
                       key=lambda c: deltas[_IDX[c] - 1], default=None)
            if best is not None:
                return best
        return cats[int(np.argmin(deltas))]

    def selection_probabilities(
        self, n_samples: int = 2000, rng: Optional[np.random.Generator] = None,
    ) -> dict[str, float]:
        """Monte Carlo estimate of P(each category is selected)."""
        rng = rng or np.random.default_rng()
        deltas = self._sample(rng, n_samples)[:, 1:]
        counts = np.bincount(deltas.argmin(axis=1), minlength=_N)
        return {c: float(n) / n_samples for c, n in zip(WEAKNESS_CATEGORIES, counts)}

    def target_rating(self, category: str, p_target: Optional[float] = None) -> float:
        """Puzzle rating at which the predicted solve probability is p_target."""
        p = self.p_target if p_target is None else p_target
        i = _IDX.get(category)
        ability = self.mean[0] + (self.mean[i] if i is not None else 0.0)
        return logit_to_rating(ability - math.log(p / (1.0 - p)))

    # ── Read-outs ────────────────────────────────────────────────────────

    def category_rating(self, category: str) -> tuple[float, float]:
        """(rating, RD) of the player's ability in one category: θ + δ_k."""
        x = self._x(category)
        if x is None:
            return logit_to_rating(self.mean[0]), LOGIT_SCALE * math.sqrt(self.cov[0, 0])
        return (logit_to_rating(float(x @ self.mean)),
                LOGIT_SCALE * math.sqrt(float(x @ self.cov @ x)))

    def weakness_map(self) -> dict[str, float]:
        """
        Predicted solve probability for a puzzle pitched exactly at the
        player's overall level, per category. 0.5 = average for this player;
        lower = weaker. Same shape as ThompsonBandit.weakness_map().
        """
        out = {}
        for c in WEAKNESS_CATEGORIES:
            i = _IDX[c]
            out[c] = _sigmoid(_kappa(self.cov[i, i]) * self.mean[i])
        return out

    def top_weaknesses(self, n: int = 5) -> list[dict]:
        return [
            {"category": c, "weakness": round(v, 4)}
            for c, v in sorted(self.weakness_map().items(), key=lambda kv: kv[1])[:n]
        ]

    # ── Persistence ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "categories": list(WEAKNESS_CATEGORIES),
            "mean": [round(float(v), 6) for v in self.mean],
            "cov": [[round(float(v), 8) for v in row] for row in self.cov],
            "pTarget": self.p_target,
            "driftTheta": self.drift_theta,
            "driftDelta": self.drift_delta,
            "nUpdates": self.n_updates,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IRTLearner":
        learner = cls(
            p_target=float(d.get("pTarget", DEFAULT_P_TARGET)),
            drift_theta=float(d.get("driftTheta", DEFAULT_DRIFT_THETA)),
            drift_delta=float(d.get("driftDelta", DEFAULT_DRIFT_DELTA)),
            n_updates=int(d.get("nUpdates", 0)),
        )
        saved_cats = d.get("categories", [])
        mean = np.asarray(d.get("mean", []), dtype=float)
        cov = np.asarray(d.get("cov", []), dtype=float)
        if saved_cats == list(WEAKNESS_CATEGORIES) and mean.shape == (_N + 1,) \
                and cov.shape == (_N + 1, _N + 1):
            learner.mean, learner.cov = mean, cov
        return learner

    @classmethod
    def from_history(
        cls,
        history: Iterable[dict],
        elo: Optional[float] = None,
        delta_prior: Optional[dict[str, tuple[float, float]]] = None,
    ) -> "IRTLearner":
        """
        Rebuild a learner by replaying a session history
        ({category, rating, solved} entries, oldest first). This is how
        players who trained before this model existed get a posterior
        from their real attempts instead of starting cold.
        """
        learner = cls.new(elo=elo, delta_prior=delta_prior)
        for h in history:
            rating = h.get("rating")
            if not rating or h.get("solved") is None:
                continue
            learner.update(h.get("category", ""), float(rating), bool(h["solved"]))
        return learner
