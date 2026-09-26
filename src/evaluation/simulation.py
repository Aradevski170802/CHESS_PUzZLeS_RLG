"""
Simulation environment for comparing puzzle-recommendation policies.

Why simulate
─────────────
A player's true per-category weakness is never observable — only noisy
solve/fail outcomes are. So the only way to check that a policy converges to
the RIGHT answer, and to compare policies fairly, is a controlled environment
with known ground truth. This is the standard way bandit algorithms are
validated before live deployment (Chapelle & Li, 2011).

What is simulated
──────────────────
* A synthetic player with overall ability θ and per-category offsets δ_k, who
  solves a puzzle of difficulty b in category k with probability
  σ(θ + δ_k − b) — the same Rasch-type form as irt_model.py, so the IRT
  policy is not handed an unfair advantage beyond using the right family.
  A few categories are genuinely weak (δ well below 0).
* Puzzle difficulties drawn from the REAL Lichess rating distribution of each
  category (semi-synthetic), and a Chess.com-style Elo that is only a noisy
  guide to puzzle ability.
* Game evidence: critical positions per category, found or missed, and then
  MISLABELLED with the confusion matrix measured for a real tactic labeller
  (validate_labeller.py). Priors are built from that evidence by the real
  production functions (profile_to_bandit_priors / profile_to_irt_prior).
* A learning model — how the player improves from practice — is an explicit,
  switchable ASSUMPTION ("none", "zpd", "error", "changepoint"). Any claim
  about learning gains is conditional on it, so results are reported under
  several.

Policies use the production classes (ThompsonBandit, IRTLearner) directly.
Common random numbers: for a given seed, every policy faces the identical
player and identical game evidence, so comparisons are paired.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from src.classifier.player_profiler import (
    SHRINK_STRENGTH,
    PlayerProfile,
    profile_to_bandit_priors,
    profile_to_irt_prior,
)
from src.data.puzzle_loader import WEAKNESS_CATEGORIES
from src.recommender.bandit import ThompsonBandit
from src.recommender.irt_model import IRTLearner, LOGIT_SCALE, TACTIC_FAMILIES, logit_to_rating

K = len(WEAKNESS_CATEGORIES)
CAT_INDEX = {c: i for i, c in enumerate(WEAKNESS_CATEGORIES)}
BAND_HALF = 300 / LOGIT_SCALE        # the app's Elo ± 300 band, in logits
ADAPTIVE_HALF = 150 / LOGIT_SCALE    # the IRT policy's ± 150 window


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PopulationConfig:
    theta_sd: float = 0.6            # spread of overall ability (logits)
    delta_sd: float = 0.35           # background spread of category offsets
    n_weak: int = 3
    weak_delta_mean: float = -1.2    # ≈ −210 rating points in the weak categories
    weak_delta_sd: float = 0.25
    elo_noise_sd: float = 0.5        # Chess.com Elo vs puzzle ability mismatch
    game_base_hit: float = 0.55      # typical hit rate in critical game positions
    game_base_sd: float = 0.3
    opportunities_mean: float = 80.0 # critical positions in ~50 analysed games
    category_weights: Optional[np.ndarray] = None   # how often each motif arises
    family_sd: float = 0.0           # >0: δ gets a shared per-family component, so
                                     # weaknesses cluster within tactic families
    weak_family: bool = False        # weak categories drawn from ONE family


@dataclass
class LearnerConfig:
    model: str = "none"              # none | zpd | error | changepoint
    eta: float = 0.06                # learning rate (logits per attempt, scaled)
    ceiling: float = 1.5             # max total improvement per category
    changepoint_t: int = 75
    changepoint_gain: float = 1.5


class SyntheticPlayer:
    def __init__(self, rng: np.random.Generator, pop: PopulationConfig, learn: LearnerConfig):
        self.pop, self.learn = pop, learn
        self.theta = float(rng.normal(0.0, pop.theta_sd))
        self.delta = rng.normal(0.0, pop.delta_sd, size=K)
        if pop.family_sd > 0:
            for cats in TACTIC_FAMILIES.values():
                idx = [CAT_INDEX[c] for c in cats if c in CAT_INDEX]
                self.delta[idx] += rng.normal(0.0, pop.family_sd)
        if pop.weak_family:
            fams = [[CAT_INDEX[c] for c in cats] for cats in TACTIC_FAMILIES.values() if len(cats) >= 3]
            fam = fams[rng.integers(len(fams))]
            self.weak = rng.choice(fam, size=min(pop.n_weak, len(fam)), replace=False)
        else:
            self.weak = rng.choice(K, size=pop.n_weak, replace=False)
        self.delta[self.weak] = rng.normal(pop.weak_delta_mean, pop.weak_delta_sd, size=len(self.weak))
        self.delta0 = self.delta.copy()
        self.elo_logit = self.theta + float(rng.normal(0.0, pop.elo_noise_sd))

    def p_solve(self, k: int, b: float) -> float:
        return float(sigmoid(self.theta + self.delta[k] - b))

    def attempt(self, k: int, b: float, rng: np.random.Generator) -> tuple[bool, float]:
        p = self.p_solve(k, b)
        solved = bool(rng.random() < p)
        m = self.learn.model
        gain = 0.0
        if m == "zpd":          # most learning at the edge of ability
            gain = self.learn.eta * 4.0 * p * (1.0 - p)
        elif m == "error":      # learning from seeing the solution after a miss
            gain = self.learn.eta * (0.0 if solved else 1.0)
        if gain:
            self.delta[k] = min(self.delta0[k] + self.learn.ceiling, self.delta[k] + gain)
        return solved, p

    def tick(self, t: int) -> None:
        if self.learn.model == "changepoint" and t == self.learn.changepoint_t:
            # The player fixes their worst weakness away from the app.
            worst = int(np.argmin(self.delta))
            self.delta[worst] += self.learn.changepoint_gain

    def weakest(self, n: int = 3) -> np.ndarray:
        return np.argsort(self.delta)[:n]


class PuzzlePool:
    """Per-category sorted difficulties (logits) from real Lichess ratings."""

    def __init__(self, ratings_by_category: dict[str, np.ndarray]):
        self.b = []
        for c in WEAKNESS_CATEGORIES:
            r = np.asarray(ratings_by_category.get(c, []), dtype=float)
            if r.size == 0:
                r = np.array([1500.0])
            self.b.append(np.sort((r - 1500.0) / LOGIT_SCALE))

    def draw(self, k: int, lo: float, hi: float, rng: np.random.Generator) -> float:
        arr = self.b[k]
        i, j = np.searchsorted(arr, lo), np.searchsorted(arr, hi, side="right")
        if j > i:
            return float(arr[rng.integers(i, j)])
        centre = 0.5 * (lo + hi)
        return float(arr[min(len(arr) - 1, np.searchsorted(arr, centre))])


def game_evidence(
    player: SyntheticPlayer,
    rng: np.random.Generator,
    confusion: Optional[np.ndarray],
    labels: list[str],
) -> PlayerProfile:
    """
    Critical positions per true category, found or missed, then relabelled
    with P(observed | true) from `confusion` (rows/cols over `labels`, which
    may include "General"; None = perfect labels). Returns a PlayerProfile
    carrying opportunity_stats exactly as player_profiler builds them.
    """
    pop = player.pop
    w = pop.category_weights if pop.category_weights is not None else np.ones(K)
    w = w / w.sum()
    n_true = rng.poisson(pop.opportunities_mean * w)
    g0 = logit(pop.game_base_hit) + rng.normal(0.0, pop.game_base_sd)
    hits_obs = np.zeros(len(labels))
    n_obs = np.zeros(len(labels))
    lab_index = {c: i for i, c in enumerate(labels)}
    for k in range(K):
        n = int(n_true[k])
        if n == 0:
            continue
        hits = rng.random(n) < sigmoid(g0 + player.delta[k])
        row = lab_index[WEAKNESS_CATEGORIES[k]]
        if confusion is None:
            obs = np.full(n, row)
        else:
            p = np.asarray(confusion[row], dtype=float)
            p = p / p.sum() if p.sum() > 0 else np.eye(len(labels))[row]
            obs = rng.choice(len(labels), size=n, p=p)
        np.add.at(n_obs, obs, 1.0)
        np.add.at(hits_obs, obs, hits.astype(float))

    stats: dict[str, dict] = {}
    keep = [i for i, c in enumerate(labels) if c in CAT_INDEX and n_obs[i] > 0]
    total_n = sum(n_obs[i] for i in keep)
    overall = sum(hits_obs[i] for i in keep) / total_n if total_n else 0.0
    for i in keep:
        rate = (hits_obs[i] + SHRINK_STRENGTH * overall) / (n_obs[i] + SHRINK_STRENGTH)
        stats[labels[i]] = {"hits": float(hits_obs[i]), "n": float(n_obs[i]),
                            "raw_n": int(n_obs[i]), "rate": float(rate)}
    return PlayerProfile(
        username="sim", estimated_elo=int(logit_to_rating(player.elo_logit)),
        opportunity_stats=stats, overall_hit_rate=float(overall),
        weakness_scores={c: 0.5 for c in WEAKNESS_CATEGORIES},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Policies
# ─────────────────────────────────────────────────────────────────────────────

class Policy:
    """choose() -> (category index, lo, hi) difficulty window in logits."""

    name = "policy"
    uses_band = True

    def reset(self, profile: PlayerProfile, elo_logit: float, rng: np.random.Generator) -> None:
        self.rng = rng
        self.elo_logit = elo_logit
        priors = profile_to_bandit_priors(profile)
        # Every policy keeps a shadow Beta posterior with the same priors, so
        # estimation quality is compared on equal terms even for policies that
        # do not use it to decide.
        self.shadow = ThompsonBandit(priors=priors)
        self.prior_means = np.array([priors[c][0] / sum(priors[c]) for c in WEAKNESS_CATEGORIES])

    def band(self) -> tuple[float, float]:
        return self.elo_logit - BAND_HALF, self.elo_logit + BAND_HALF

    def choose(self, t: int) -> tuple[int, float, float]:
        raise NotImplementedError

    def observe(self, k: int, b: float, solved: bool) -> None:
        self.shadow.update(WEAKNESS_CATEGORIES[k], solved)

    def weakness_estimate(self) -> np.ndarray:
        """Higher = believed weaker."""
        return -np.array([self.shadow.arms[c].mean for c in WEAKNESS_CATEGORIES])


class RandomPolicy(Policy):
    name = "Random"

    def choose(self, t):
        return int(self.rng.integers(K)), *self.band()


class RoundRobinPolicy(Policy):
    name = "Round-robin"

    def choose(self, t):
        return t % K, *self.band()


class StaticProfilePolicy(Policy):
    """Simple heuristic: rotate through the 3 weakest categories of the game profile."""
    name = "Static profile (top-3)"

    def reset(self, profile, elo_logit, rng):
        super().reset(profile, elo_logit, rng)
        self.targets = np.argsort(self.prior_means)[:3]

    def choose(self, t):
        return int(self.targets[t % 3]), *self.band()


class GreedyBetaPolicy(Policy):
    """Always the lowest posterior mean — no exploration."""
    name = "Greedy (Beta mean)"

    def choose(self, t):
        means = np.array([self.shadow.arms[c].mean for c in WEAKNESS_CATEGORIES])
        best = np.flatnonzero(means == means.min())
        return int(self.rng.choice(best)), *self.band()


class BetaTSPolicy(Policy):
    """The original production policy: Beta Thompson Sampling, Elo ± 300 band."""

    def __init__(self, discount: float = 1.0, prior_scale: float = 1.0):
        self.discount, self.prior_scale = discount, prior_scale
        self.name = "Beta-TS" + (f" (γ={discount})" if discount < 1 else "") + \
            (f" (prior×{prior_scale:g})" if prior_scale != 1 else "")

    def reset(self, profile, elo_logit, rng):
        super().reset(profile, elo_logit, rng)
        priors = profile_to_bandit_priors(profile)
        if self.prior_scale != 1.0:
            priors = {c: (1 + (a - 1) * self.prior_scale, 1 + (b - 1) * self.prior_scale)
                      for c, (a, b) in priors.items()}
        self.bandit = ThompsonBandit(priors=priors, discount=self.discount)

    def choose(self, t):
        return CAT_INDEX[self.bandit.select_one(self.rng)], *self.band()

    def observe(self, k, b, solved):
        super().observe(k, b, solved)
        self.bandit.update(WEAKNESS_CATEGORIES[k], solved)

    def weakness_estimate(self):
        return -np.array([self.bandit.arms[c].mean for c in WEAKNESS_CATEGORIES])


class IRTTSPolicy(Policy):
    """The new policy: difficulty-aware IRT learner (+ adaptive difficulty)."""

    def __init__(self, adaptive: bool = True, p_target: float = 0.65, use_prior: bool = True,
                 family_prior: bool = False):
        self.adaptive, self.p_target, self.use_prior = adaptive, p_target, use_prior
        self.family_prior = family_prior
        self.uses_band = not adaptive
        self.name = "IRT-TS" + ("" if adaptive else " (Elo band)") + \
            (f" (p*={p_target:g})" if p_target != 0.65 else "") + ("" if use_prior else " (no prior)") + \
            (" (family prior)" if family_prior else "")

    def reset(self, profile, elo_logit, rng):
        super().reset(profile, elo_logit, rng)
        prior = profile_to_irt_prior(profile) if self.use_prior else None
        self.learner = IRTLearner.new(elo=logit_to_rating(elo_logit), delta_prior=prior,
                                      p_target=self.p_target, family_prior=self.family_prior)

    def choose(self, t):
        cat = self.learner.select_category(self.rng)
        k = CAT_INDEX[cat]
        if not self.adaptive:
            return k, *self.band()
        centre = (self.learner.target_rating(cat) - 1500.0) / LOGIT_SCALE
        return k, centre - ADAPTIVE_HALF, centre + ADAPTIVE_HALF

    def observe(self, k, b, solved):
        super().observe(k, b, solved)
        self.learner.update(WEAKNESS_CATEGORIES[k], 1500.0 + LOGIT_SCALE * b, solved)

    def weakness_estimate(self):
        return -self.learner.mean[1:]


class OraclePolicy(Policy):
    """Knows the true current δ: always the weakest category, pitched at p = 0.65."""
    name = "Oracle"
    uses_band = False

    def attach(self, player: SyntheticPlayer) -> None:
        self.player = player

    def choose(self, t):
        k = int(np.argmin(self.player.delta))
        centre = self.player.theta + self.player.delta[k] - logit(0.65)
        return k, centre - ADAPTIVE_HALF, centre + ADAPTIVE_HALF

    def weakness_estimate(self):
        return -self.player.delta.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────────────

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = math.sqrt(float((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom else 0.0


@dataclass
class EpisodeResult:
    weak_hit: np.ndarray       # (T,) chosen category among the current true bottom-3
    regret: np.ndarray         # (T,) δ_chosen − min δ (logits, ≥ 0)
    p_served: np.ndarray       # (T,) true solve probability of the served puzzle
    solved: np.ndarray         # (T,)
    switched: np.ndarray       # (T,) chosen category differs from the previous one
    spearman: np.ndarray       # (T,) rank correlation of estimated vs true weakness
    top3_recall: np.ndarray    # (T,) overlap of estimated and true bottom-3, / 3
    gain_weak: float = 0.0     # mean δ improvement over the initially weak categories
    gain_all: float = 0.0
    mae: Optional[np.ndarray] = None   # (T,) fixed-rate protocol only: |p̂ − p| over arms


def run_episode(
    make_policy: Callable[[], Policy],
    seed: int,
    pool: PuzzlePool,
    pop: PopulationConfig,
    learn: LearnerConfig,
    confusion: Optional[np.ndarray],
    labels: list[str],
    T: int = 150,
) -> EpisodeResult:
    # Common random numbers: the player and the game evidence depend only on
    # the seed, so every policy meets the same player.
    world = np.random.default_rng(seed)
    player = SyntheticPlayer(world, pop, learn)
    profile = game_evidence(player, world, confusion, labels)
    policy = make_policy()
    policy.reset(profile, player.elo_logit, np.random.default_rng(seed + 7919))
    if isinstance(policy, OraclePolicy):
        policy.attach(player)
    outcomes = np.random.default_rng(seed + 104729)

    res = EpisodeResult(*(np.zeros(T) for _ in range(7)))
    prev = -1
    initially_weak = player.weak.copy()
    for t in range(T):
        player.tick(t)
        k, lo, hi = policy.choose(t)
        truth_weak = set(player.weakest(3).tolist())
        res.weak_hit[t] = k in truth_weak
        res.regret[t] = player.delta[k] - player.delta.min()
        res.switched[t] = k != prev
        prev = k
        b = pool.draw(k, lo, hi, outcomes)
        solved, p = player.attempt(k, b, outcomes)
        res.p_served[t], res.solved[t] = p, solved
        policy.observe(k, b, solved)
        est = policy.weakness_estimate()
        res.spearman[t] = _spearman(est, -player.delta)
        res.top3_recall[t] = len(set(np.argsort(-est)[:3].tolist()) & truth_weak) / 3.0
    res.gain_weak = float((player.delta[initially_weak] - player.delta0[initially_weak]).mean())
    res.gain_all = float((player.delta - player.delta0).mean())
    return res


def fixed_rate_episode(make_policy, seed: int, T: int = 150,
                       weak_p: float = 0.30, strong_p: float = 0.80) -> EpisodeResult:
    """
    The pre-registered protocol (EVALUATION_METHODOLOGY): fixed solve rates,
    3 categories at 0.30 and the rest at 0.80, no difficulty, no learning,
    uniform Beta(1,1) priors.
    """
    world = np.random.default_rng(seed)
    weak = world.choice(K, size=3, replace=False)
    true_p = np.full(K, strong_p)
    true_p[weak] = weak_p
    flat = PlayerProfile(username="sim", estimated_elo=1500,
                         weakness_scores={c: 0.5 for c in WEAKNESS_CATEGORIES})
    policy = make_policy()
    policy.reset(flat, 0.0, np.random.default_rng(seed + 7919))
    if hasattr(policy, "bandit"):
        policy.bandit = ThompsonBandit(discount=getattr(policy, "discount", 1.0))
    policy.shadow = ThompsonBandit()
    outcomes = np.random.default_rng(seed + 104729)
    res = EpisodeResult(*(np.zeros(T) for _ in range(7)), mae=np.zeros(T))
    weak_set, prev = set(weak.tolist()), -1
    for t in range(T):
        k, _, _ = policy.choose(t)
        res.weak_hit[t] = k in weak_set
        res.regret[t] = true_p[k] - true_p.min()
        res.switched[t] = k != prev
        prev = k
        solved = bool(outcomes.random() < true_p[k])
        res.p_served[t], res.solved[t] = true_p[k], solved
        policy.observe(k, 0.0, solved)
        est_rate = -policy.weakness_estimate()          # estimated solve rate per arm
        res.mae[t] = float(np.mean(np.abs(est_rate - true_p)))
        res.spearman[t] = _spearman(-est_rate, -true_p)
        res.top3_recall[t] = len(set(np.argsort(est_rate)[:3].tolist()) & weak_set) / 3.0
    return res
