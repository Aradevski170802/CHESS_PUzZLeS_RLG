"""
Trained alternative to the rule-based weakness scorer in player_profiler.py.

Why this exists
────────────────
player_profiler.build_profile() derives weakness_scores with hand-tuned
if/else thresholds (see _compute_weakness_scores). This module adds
build_profile_ml() as a drop-in alternative with the exact same signature
and return type -- only the weakness-scoring step is learned from data
instead of hard-coded, per IMPROVEMENT_PLAN.md's guidance to add this
"as a new function alongside the existing rule-based build_profile()
rather than replacing it outright, so you can A/B the two before
committing to a swap."

The honest data-reality caveat
────────────────────────────────
As of writing this, only 4 real players have any mined-puzzle history at
all (data/user_puzzles/*.json: 9-33 puzzles each), and none of them have a
persisted, feature-level game-analysis record to train or evaluate against
directly. A held-out-player split over N=4 cannot produce a statistically
meaningful accuracy/MAE/confidence interval for ANY model, no matter how
good the model is.

So the model here is trained and validated on SIMULATED players
(generate_synthetic_dataset), where the ground-truth per-category weakness
is known exactly and the "observable" features are noisy correlates of it:
  1. Seven coarse summary stats, linked to weakness the way the rule-based
     function assumes (middlegame errors -> middlegame-tactic categories,
     endgame errors -> endgame categories, high blunder rate -> Hanging
     Piece / Fork).
  2. Twenty-three per-category observed error rates, drawn from a
     Dirichlet-multinomial process whose concentration is boosted for the
     true weak categories -- mirroring what MoveError.category now actually
     measures in production (src/classifier/stockfish_analyzer.py classifies
     each missed best move via src/puzzles/generator._classify_tactic), with
     realistic sampling noise for players with few recorded errors.
Both blocks are genuinely noisy correlates, not a direct readout of the
ground truth, so the model has to recover real (if imperfect) statistical
signal rather than memorise a formula. Held-out-player evaluation here means
genuine held-out-SIMULATED-player K-fold cross-validation (each synthetic
player contributes exactly one row, so a fold boundary is a player
boundary), which IS statistically meaningful at n=hundreds.

build_profile_ml() itself works on real Chess.com game analyses at
inference time (same input type as build_profile()) -- only the *training
and validation* data is simulated. Treat the cross-validation numbers this
module reports as evidence the methodology is sound, not as a claim about
real-world accuracy on real players; that claim can only be made once more
real players have accumulated mined-puzzle history.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from src.classifier.player_profiler import (
    PlayerProfile,
    _ENDGAME_CATS,
    _MIDDLEGAME_CATS,
    build_profile,
)
from src.classifier.stockfish_analyzer import GameAnalysis
from src.data.puzzle_loader import WEAKNESS_CATEGORIES

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("data/processed/weakness_model.joblib")

_COARSE_FEATURE_NAMES: list[str] = [
    "middlegame_error_rate",
    "endgame_error_rate",
    "blunders_per_game",
    "mistakes_per_game",
    "inaccuracies_per_game",
    "accuracy_estimate_norm",
    "log_games_analysed",
]

# One observed-error-rate feature per tactic category (see MoveError.category
# in stockfish_analyzer.py), appended after the coarse features above.
FEATURE_NAMES: list[str] = _COARSE_FEATURE_NAMES + [
    f"error_rate__{cat}" for cat in WEAKNESS_CATEGORIES
]

_BLUNDER_WEAK_CATS = {"Hanging Piece", "Fork"}


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (real usage path)
# ─────────────────────────────────────────────────────────────────────────────

def _profile_to_features(profile: PlayerProfile) -> np.ndarray:
    """
    Turn an already-built PlayerProfile into the fixed-order feature vector
    the model expects. Reuses build_profile()'s aggregation (games_analysed,
    blunders_total, errors_by_phase, errors_by_category, accuracy_estimate,
    ...) rather than re-deriving it, so build_profile_ml() and build_profile()
    are always looking at the same underlying facts about the player.

    Returns a vector of length len(FEATURE_NAMES): 7 coarse summary stats
    followed by one observed-error-rate feature per tactic category.
    """
    total_errors = max(1, sum(profile.errors_by_phase.values()))
    games = max(1, profile.games_analysed)

    middlegame_rate = profile.errors_by_phase.get("middlegame", 0) / total_errors
    endgame_rate = profile.errors_by_phase.get("endgame", 0) / total_errors
    blunder_per_game = profile.blunders_total / games
    mistake_per_game = profile.mistakes_total / games
    inaccuracy_per_game = profile.inaccuracies_total / games
    accuracy_norm = profile.accuracy_estimate / 100.0
    games_log = math.log1p(profile.games_analysed)

    coarse_features = np.array(
        [
            middlegame_rate,
            endgame_rate,
            blunder_per_game,
            mistake_per_game,
            inaccuracy_per_game,
            accuracy_norm,
            games_log,
        ],
        dtype=float,
    )

    category_rates = np.array(
        [profile.errors_by_category.get(cat, 0) / total_errors for cat in WEAKNESS_CATEGORIES],
        dtype=float,
    )

    return np.concatenate([coarse_features, category_rates])


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic training data
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_one_player(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate one synthetic player: a known latent per-category weakness
    vector, plus two blocks of noisy "observable" features that correlate
    with it but never reveal the exact weak categories directly:

      1. Coarse phase/severity summary stats -- same domain links as
         player_profiler._compute_weakness_scores (middlegame errors ->
         middlegame-tactic categories, endgame errors -> endgame categories,
         high blunder rate -> Hanging Piece / Fork). These only identify a
         *group* of several categories, not which one specifically.

      2. Per-category observed error rates, drawn from a Dirichlet-
         multinomial process whose concentration is boosted for the true
         weak categories -- mirroring what MoveError.category now measures
         in production (each missed best move is classified via
         generator._classify_tactic). This block can identify a *specific*
         category, but is noisy, especially for players with few recorded
         errors (n_observed_errors as low as 5, matching how sparse the
         real mined-puzzle data currently is).
    """
    n_cats = len(WEAKNESS_CATEGORIES)

    # Baseline: everyone is a little weak everywhere (skewed low, mean ~0.29).
    true_weakness = rng.beta(2, 5, size=n_cats)

    # 1-3 categories are genuinely weak for this player.
    n_weak = int(rng.integers(1, 4))
    weak_idx = rng.choice(n_cats, size=n_weak, replace=False)
    true_weakness[weak_idx] = rng.uniform(0.65, 0.95, size=n_weak)
    true_weakness = np.clip(true_weakness, 0.0, 1.0)

    weak_cats = {WEAKNESS_CATEGORIES[i] for i in weak_idx}
    has_middlegame_weak = bool(weak_cats & set(_MIDDLEGAME_CATS))
    has_endgame_weak = bool(weak_cats & set(_ENDGAME_CATS))
    has_blunder_weak = bool(weak_cats & _BLUNDER_WEAK_CATS)

    middlegame_rate = np.clip(rng.normal(0.55 if has_middlegame_weak else 0.30, 0.15), 0.0, 1.0)
    endgame_rate = np.clip(rng.normal(0.45 if has_endgame_weak else 0.15, 0.12), 0.0, 1.0)
    blunder_per_game = max(0.0, rng.normal(1.8 if has_blunder_weak else 0.5, 0.6))
    mistake_per_game = max(0.0, rng.normal(2.0, 0.8))
    inaccuracy_per_game = max(0.0, rng.normal(3.0, 1.0))
    accuracy_norm = np.clip(rng.normal(0.75 - 0.05 * n_weak, 0.10), 0.1, 1.0)
    games_analysed = int(rng.integers(5, 60))
    games_log = math.log1p(games_analysed)

    coarse_features = np.array(
        [
            middlegame_rate,
            endgame_rate,
            blunder_per_game,
            mistake_per_game,
            inaccuracy_per_game,
            accuracy_norm,
            games_log,
        ],
        dtype=float,
    )

    # Higher true_weakness -> higher Dirichlet concentration -> that
    # category is more likely to dominate the observed (sampled) error
    # counts, but a small n_observed_errors keeps this genuinely noisy.
    concentration = 1.0 + true_weakness * 8.0
    category_probs = rng.dirichlet(concentration)
    # Guard against floating-point drift pushing the sum a hair past 1.0,
    # which np.random.Generator.multinomial can reject.
    category_probs = category_probs / category_probs.sum()
    n_observed_errors = int(rng.integers(5, 40))
    category_counts = rng.multinomial(n_observed_errors, category_probs)
    category_rates = category_counts / max(1, n_observed_errors)

    features = np.concatenate([coarse_features, category_rates])
    return features, true_weakness


def generate_synthetic_dataset(
    n_players: int = 300, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, Y): X is (n_players, len(FEATURE_NAMES)) features, Y is (n_players, 23) labels."""
    rng = np.random.default_rng(seed)
    n_cats = len(WEAKNESS_CATEGORIES)
    X = np.zeros((n_players, len(FEATURE_NAMES)))
    Y = np.zeros((n_players, n_cats))
    for i in range(n_players):
        X[i], Y[i] = _simulate_one_player(rng)
    return X, Y


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def _new_model(seed: int) -> RandomForestRegressor:
    # RandomForestRegressor natively supports multi-output y (n_samples, n_outputs),
    # so no MultiOutputRegressor wrapper is needed.
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=3,
        random_state=seed,
    )


@dataclass
class FoldMetrics:
    fold: int
    mae: float
    baseline_mae: float
    top_weak_category_in_top3: float  # fraction of test players where the true
    # top-1 weak category was in the model's predicted top-3


@dataclass
class TrainingReport:
    n_players: int
    n_splits: int
    fold_metrics: list[FoldMetrics] = field(default_factory=list)
    mean_mae: float = 0.0
    std_mae: float = 0.0
    mean_baseline_mae: float = 0.0
    feature_importances: dict[str, float] = field(default_factory=dict)


def cross_validate(
    X: np.ndarray, Y: np.ndarray, *, n_splits: int = 5, seed: int = 42
) -> TrainingReport:
    """
    Held-out-player K-fold CV. Each row of X/Y is one (synthetic) player, so
    a fold split is automatically a player split -- no player's data leaks
    across train/test within a fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics: list[FoldMetrics] = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        model = _new_model(seed)
        model.fit(X[train_idx], Y[train_idx])
        preds = model.predict(X[test_idx])

        mae = float(np.mean(np.abs(preds - Y[test_idx])))

        baseline_vector = Y[train_idx].mean(axis=0)
        baseline_pred = np.tile(baseline_vector, (len(test_idx), 1))
        baseline_mae = float(np.mean(np.abs(baseline_pred - Y[test_idx])))

        hits = 0
        for row_pred, row_true in zip(preds, Y[test_idx]):
            true_top = int(np.argmax(row_true))
            pred_top3 = set(np.argsort(row_pred)[-3:])
            if true_top in pred_top3:
                hits += 1
        top3_rate = hits / len(test_idx) if len(test_idx) else 0.0

        fold_metrics.append(FoldMetrics(fold_idx, mae, baseline_mae, top3_rate))

    maes = [f.mae for f in fold_metrics]
    baseline_maes = [f.baseline_mae for f in fold_metrics]

    return TrainingReport(
        n_players=X.shape[0],
        n_splits=n_splits,
        fold_metrics=fold_metrics,
        mean_mae=float(np.mean(maes)),
        std_mae=float(np.std(maes)),
        mean_baseline_mae=float(np.mean(baseline_maes)),
    )


def fit_model(X: np.ndarray, Y: np.ndarray, seed: int = 42) -> RandomForestRegressor:
    model = _new_model(seed)
    model.fit(X, Y)
    return model


def _feature_importances(model: RandomForestRegressor) -> dict[str, float]:
    return {name: float(imp) for name, imp in zip(FEATURE_NAMES, model.feature_importances_)}


def train_and_evaluate(
    n_players: int = 300, seed: int = 42, n_splits: int = 5
) -> tuple[TrainingReport, RandomForestRegressor]:
    X, Y = generate_synthetic_dataset(n_players, seed)
    report = cross_validate(X, Y, n_splits=n_splits, seed=seed)
    final_model = fit_model(X, Y, seed=seed)
    report.feature_importances = _feature_importances(final_model)
    return report, final_model


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model: RandomForestRegressor, path: Path = DEFAULT_MODEL_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path = DEFAULT_MODEL_PATH) -> RandomForestRegressor:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained weakness model at {path}. Run "
            f"`python -m src.classifier.ml_weakness_model` first to train and save one."
        )
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# Public API (drop-in alternative to player_profiler.build_profile)
# ─────────────────────────────────────────────────────────────────────────────

def build_profile_ml(
    analyses: list[GameAnalysis],
    username: str,
    estimated_elo: int,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> PlayerProfile:
    """
    Drop-in ML alternative to player_profiler.build_profile(). Same
    signature, same PlayerProfile shape -- only weakness_scores is produced
    by a trained model instead of hand-tuned rules. Everything else
    (openings, error breakdown, game record) is identical because this
    calls build_profile() first and only overrides weakness_scores.
    """
    profile = build_profile(analyses, username, estimated_elo)

    successful = [a for a in analyses if not a.failed]
    if not successful:
        # build_profile() already returned a flat 0.5-everywhere profile in
        # this case -- there is nothing for a model to predict from, and no
        # need to require a trained model file to exist just to hit this path.
        return profile

    model = load_model(model_path)
    features = _profile_to_features(profile).reshape(1, -1)
    raw_pred = model.predict(features)[0]
    clipped = np.clip(raw_pred, 0.0, 1.0)
    profile.weakness_scores = {
        cat: float(score) for cat, score in zip(WEAKNESS_CATEGORIES, clipped)
    }
    return profile


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-players", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    report, model = train_and_evaluate(args.n_players, args.seed, args.n_splits)
    save_model(model, args.out)

    logger.info(
        "Synthetic players: %d (%d-fold held-out-player CV)", report.n_players, report.n_splits
    )
    logger.info(
        "Mean MAE: %.4f (+/- %.4f)  |  naive mean-baseline MAE: %.4f",
        report.mean_mae, report.std_mae, report.mean_baseline_mae,
    )
    for fm in report.fold_metrics:
        logger.info(
            "  fold %d: mae=%.4f  baseline_mae=%.4f  true-top-weak-in-predicted-top3=%.0f%%",
            fm.fold, fm.mae, fm.baseline_mae, fm.top_weak_category_in_top3 * 100,
        )
    top_features = sorted(report.feature_importances.items(), key=lambda kv: -kv[1])[:3]
    logger.info("Top features: %s", ", ".join(f"{k}={v:.3f}" for k, v in top_features))
    logger.info("Model saved -> %s", args.out)
    logger.warning(
        "This model is trained and validated on %d SIMULATED players. Only 4 real "
        "players currently have mined-puzzle history (data/user_puzzles/*.json), "
        "too few for a statistically meaningful held-out-player evaluation on real "
        "data. Real games still flow through this exact code path via "
        "build_profile_ml(); treat the numbers above as evidence the methodology "
        "is sound, not as a claim about real-world accuracy.",
        report.n_players,
    )


if __name__ == "__main__":
    main()
