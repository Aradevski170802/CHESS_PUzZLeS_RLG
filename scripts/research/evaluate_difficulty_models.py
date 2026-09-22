"""
Prequential evaluation of the difficulty / skill models on REAL solve logs.

Prequential ("predict, then learn"): events are replayed in time order; each
model predicts P(solve) for an attempt BEFORE it sees the outcome, then
updates. Every prediction is therefore out-of-sample, which is the honest way
to score online models on a single stream of data (Dawid's prequential
principle), and no train/test split is wasted on a tiny dataset.

Models
───────
  Base rate               running solve rate so far (Laplace-smoothed)
  Static Elo              σ((player Elo − puzzle rating) / 173.7) — the app's
                          original assumption: Chess.com Elo vs puzzle rating
  Glicko-2 (symmetric)    difficulty_fitter as originally designed: player
                          AND puzzle updated on every attempt
  Glicko-2 (pool frozen)  Lichess puzzle ratings held fixed; players and mined
                          puzzles updated
  IRT, θ only             irt_model with no category offsets (a single
                          ability, i.e. Glicko-like)
  IRT, per category       irt_model with per-category offsets — the new model

Metrics: log loss and Brier score (lower = better), AUC, and a paired
bootstrap CI for each model's log-loss difference vs Static Elo.

Caveat stated up front: this is ~120 events from 5 players, two of whom
contribute most of them. It can show whether a model is badly miscalibrated;
it cannot establish small differences. The CIs say which is which.

Usage: python -m scripts.research.evaluate_difficulty_models
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.analysis.difficulty_fitter import (
    MINED_PUZZLE_RD,
    POOL_PUZZLE_RD,
    fit_from_sessions,
    is_mined_puzzle,
    load_solve_events,
)
from src.recommender.irt_model import IRTLearner

SESSIONS = Path("data/sessions")
OUT = Path("eval/research")


def _elos() -> dict[str, float]:
    out = {}
    for p in SESSIONS.glob("*.json"):
        s = json.loads(p.read_text("utf-8"))
        out[s.get("username") or p.stem] = float(s.get("estimatedElo") or 1500)
    return out


def _categories() -> dict[tuple, str]:
    """(username, ts, puzzleId) -> category, from the raw history entries."""
    out = {}
    for p in SESSIONS.glob("*.json"):
        s = json.loads(p.read_text("utf-8"))
        u = s.get("username") or p.stem
        for h in s.get("history", []):
            out[(u, h.get("ts"), h.get("puzzleId"))] = h.get("category", "")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events = load_solve_events(SESSIONS)
    elos, cats = _elos(), _categories()
    y = np.array([1.0 if e.solved else 0.0 for e in events])
    preds: dict[str, list[float]] = defaultdict(list)

    # Base rate and static Elo
    s, n = 1.0, 2.0
    for e in events:
        preds["Base rate"].append(s / n)
        s, n = s + (1.0 if e.solved else 0.0), n + 1.0
        preds["Static Elo"].append(1 / (1 + 10 ** ((e.puzzle_rating - elos.get(e.username, 1500)) / 400)))

    # Glicko-2, both designs (predictions captured before each update)
    for name, frozen in (("Glicko-2 (symmetric)", False), ("Glicko-2 (pool frozen)", True)):
        fit_from_sessions(SESSIONS, freeze_pool=frozen,
                          on_event=lambda ev, p, name=name: preds[name].append(p))

    # IRT learners, one per player
    for name, use_cat in (("IRT, θ only", False), ("IRT, per category", True)):
        learners: dict[str, IRTLearner] = {}
        for e in events:
            L = learners.setdefault(e.username, IRTLearner.new(elo=elos.get(e.username)))
            cat = cats.get((e.username, e.ts.strftime("%Y-%m-%dT%H:%M:%SZ"), e.puzzle_id), "") if use_cat else ""
            rd = MINED_PUZZLE_RD if is_mined_puzzle(e.puzzle_id) else POOL_PUZZLE_RD
            preds[name].append(L.update(cat, e.puzzle_rating, e.solved, rd))

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(y), size=(5000, len(y)))

    def logloss(p):
        p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    ref = logloss(preds["Static Elo"])
    report = {"events": len(events), "players": len({e.username for e in events}),
              "solve_rate": round(float(y.mean()), 3),
              "mined_share": round(float(np.mean([is_mined_puzzle(e.puzzle_id) for e in events])), 3),
              "models": {}}
    for name, p in preds.items():
        p = np.asarray(p)
        ll = logloss(p)
        diff = ll - ref
        boot = diff[idx].mean(axis=1)
        try:
            auc = float(roc_auc_score(y, p))
        except ValueError:
            auc = None
        report["models"][name] = {
            "log_loss": round(float(ll.mean()), 4),
            "brier": round(float(np.mean((p - y) ** 2)), 4),
            "auc": round(auc, 3) if auc is not None else None,
            "mean_prediction": round(float(p.mean()), 3),
            "logloss_minus_static_elo": round(float(diff.mean()), 4),
            "ci95": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
        }

    (OUT / "difficulty_model_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 6))
    bins = np.linspace(0, 1, 6)
    for name in ("Static Elo", "Glicko-2 (symmetric)", "Glicko-2 (pool frozen)", "IRT, per category"):
        p = np.asarray(preds[name])
        which = np.clip(np.digitize(p, bins) - 1, 0, 4)
        xs = [p[which == b].mean() for b in range(5) if (which == b).sum() >= 3]
        ys = [y[which == b].mean() for b in range(5) if (which == b).sum() >= 3]
        ax.plot(xs, ys, marker="o", label=f"{name} (LL {report['models'][name]['log_loss']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.set_xlabel("predicted P(solve)")
    ax.set_ylabel("observed solve rate")
    ax.set_title(f"Prequential calibration on real logs (n = {len(y)} attempts)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "difficulty_calibration.png", dpi=140)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
