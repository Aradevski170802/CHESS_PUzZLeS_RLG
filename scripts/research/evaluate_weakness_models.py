"""
Evaluate the weakness models on REAL players (the Chess.com research cohort).

Question: from a player's past games, which model best predicts the tactics
they will miss in their FUTURE games?

Protocol
────────
* Temporal split per player: the oldest games (TRAIN_FRACTION) are the
  player's history; the newest games are the test period. This is exactly the
  app's use case and cannot leak future information into the features.
* Unit of evaluation: every critical position (opportunity) in the test
  games is a Bernoulli event — the player missed it (1) or found it (0).
  Each model outputs P(miss | player, category); scored by log loss and Brier
  (proper scoring rules) and AUC.
* Population-level parameters (base rates, calibration maps, the learned
  model) are always fitted on OTHER players: 5-fold cross-validation grouped
  by player. A player's own history is used only for their own predictions.
* Models that output uncalibrated scores (the rule-based scorer, the RF
  trained on simulated players) are mapped to probabilities by isotonic
  regression fitted on the training folds — giving each its best monotone
  calibration, so the comparison is about information, not scale.

Models
───────
  Global base rate          one miss rate for everyone
  Category base rate        per-category population miss rate
  Player base rate          the player's own overall miss rate
  Rule-based scorer         player_profiler._compute_weakness_scores (legacy)
  RF (synthetic-trained)    ml_weakness_model's saved model (trained on simulated players)
  Empirical Bayes           production profiler: category hit rate shrunk to
                            the player's own rate (player_profiler.aggregate_opportunities)
  Hierarchical EB           shrink toward player rate × population category
                            effect; shrinkage strength chosen by inner CV
  RF (real-data)            RandomForest on (player, category) rows trained on
                            real cohort players

Also reported: split-half reliability of per-category hit rates (how much
stable, person-specific signal exists at all), and population parameters
that calibrate the simulator in src/evaluation/simulation.py.

Usage: python -m scripts.research.evaluate_weakness_models
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from src.classifier.ml_weakness_model import DEFAULT_MODEL_PATH, _profile_to_features, load_model
from src.classifier.player_profiler import SHRINK_STRENGTH, build_profile
from src.classifier.stockfish_analyzer import GameAnalysis, MoveError, Opportunity
from src.data.puzzle_loader import WEAKNESS_CATEGORIES

ANALYSIS = Path("data/research/cohort/analysis")
OUT = Path("eval/research")
TRAIN_FRACTION = 2 / 3
K = len(WEAKNESS_CATEGORIES)
CIDX = {c: i for i, c in enumerate(WEAKNESS_CATEGORIES)}
EPS = 1e-4


def _to_analysis(g: dict) -> GameAnalysis:
    return GameAnalysis(
        errors=[MoveError(move_number=0, fen_before="", move_uci="", cp_loss=0,
                          severity=e["severity"], phase=e["phase"], category=e["category"],
                          wp_loss=e["wp_loss"]) for e in g["errors"]],
        opportunities=[Opportunity(**o) for o in g["opportunities"]],
        player_won=g["won"], num_moves=g["num_moves"], avg_cp_loss=g["avg_cp_loss"],
        avg_wp_loss=g["avg_wp_loss"], time_class=g["time_class"], end_time=g["end_time"],
    )


def load_players() -> list[dict]:
    # Never let a norms file fitted on the whole cohort leak into scoring:
    # the population-norms model below is fitted fold by fold instead.
    import src.classifier.player_profiler as pp
    pp.load_category_norms = lambda *a, **k: None
    players = []
    for path in sorted(ANALYSIS.glob("*.json")):
        d = json.loads(path.read_text("utf-8"))
        games = sorted(d["games"], key=lambda g: g["end_time"] or 0)   # oldest first
        if len(games) < 15:
            continue
        cut = int(len(games) * TRAIN_FRACTION)
        train = [_to_analysis(g) for g in games[:cut]]
        test_opps = [o for g in games[cut:] for o in g["opportunities"] if o["category"] in CIDX]
        profile = build_profile(train, d["id"], d["rating"])
        tr_h, tr_n = np.zeros(K), np.zeros(K)
        for a in train:
            for o in a.opportunities:
                if o.category in CIDX:
                    tr_n[CIDX[o.category]] += 1
                    tr_h[CIDX[o.category]] += o.hit
        te_miss, te_n = np.zeros(K), np.zeros(K)
        for o in test_opps:
            te_n[CIDX[o["category"]]] += 1
            te_miss[CIDX[o["category"]]] += 0 if o["hit"] else 1
        # split-half (odd / even training games) for reliability
        halves = [(np.zeros(K), np.zeros(K)), (np.zeros(K), np.zeros(K))]
        for i, a in enumerate(train):
            h, n = halves[i % 2]
            for o in a.opportunities:
                if o.category in CIDX:
                    n[CIDX[o.category]] += 1
                    h[CIDX[o.category]] += o.hit
        players.append({"id": d["id"], "band": d["band"], "rating": d["rating"], "profile": profile,
                        "tr_h": tr_h, "tr_n": tr_n, "te_miss": te_miss, "te_n": te_n, "halves": halves})
    return players


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-z))


# ── model predictions: each returns a (K,) vector of P(miss) for one player ──

def pred_global(train_players, p):
    h = sum(q["tr_h"].sum() for q in train_players)
    n = sum(q["tr_n"].sum() for q in train_players)
    return np.full(K, 1 - h / n)


def category_rates(train_players):
    h = sum(q["tr_h"] for q in train_players)
    n = sum(q["tr_n"] for q in train_players)
    g = 1 - h.sum() / n.sum()
    return np.where(n > 0, (n - h + 2 * g) / (n + 2), g), g


def pred_category(train_players, p):
    return category_rates(train_players)[0]


def pred_player(train_players, p):
    n = p["tr_n"].sum()
    g = pred_global(train_players, p)[0]
    return np.full(K, (n - p["tr_h"].sum() + 4 * g) / (n + 4))


def pred_empirical_bayes(train_players, p):
    stats = p["profile"].opportunity_stats
    base = p["profile"].overall_hit_rate or 0.5
    return np.array([1 - stats.get(c, {}).get("rate", base) for c in WEAKNESS_CATEGORIES])


def pred_hier_eb(train_players, p, strength):
    cat, g = category_rates(train_players)
    player_miss = pred_player(train_players, p)[0]
    prior = sig(logit(player_miss) + logit(cat) - logit(g))
    misses = p["tr_n"] - p["tr_h"]
    return (misses + strength * prior) / (p["tr_n"] + strength)


def _score_vec(p, kind, rf_syn=None):
    if kind == "rule":
        return np.array([p["profile"].weakness_scores.get(c, 0.5) for c in WEAKNESS_CATEGORIES])
    feats = _profile_to_features(p["profile"]).reshape(1, -1)
    return np.clip(rf_syn.predict(feats)[0], 0, 1)


def calibrated(train_players, p, kind, rf_syn=None):
    """Isotonic map from a score to P(miss), fitted on training players' test events."""
    xs, ys, ws = [], [], []
    for q in train_players:
        s = _score_vec(q, kind, rf_syn)
        for k in range(K):
            if q["te_n"][k] > 0:
                xs.append(s[k]); ys.append(q["te_miss"][k] / q["te_n"][k]); ws.append(q["te_n"][k])
    iso = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip")
    iso.fit(xs, ys, sample_weight=ws)
    return iso.predict(_score_vec(p, kind, rf_syn))


def _rf_rows(players, cat_rates, g):
    X, y, w = [], [], []
    for q in players:
        pm = (q["tr_n"].sum() - q["tr_h"].sum() + 4 * g) / (q["tr_n"].sum() + 4)
        coarse = _profile_to_features(q["profile"])[:7]
        for k in range(K):
            own = (q["tr_n"][k] - q["tr_h"][k] + 2 * pm) / (q["tr_n"][k] + 2)
            X.append(np.concatenate([coarse, [pm, own, math.log1p(q["tr_n"][k]), cat_rates[k]],
                                     np.eye(K)[k]]))
            y.append(q["te_miss"][k] / q["te_n"][k] if q["te_n"][k] > 0 else np.nan)
            w.append(q["te_n"][k])
    return np.array(X), np.array(y), np.array(w)


def write_norms(players: list[dict], strengths: list[int]) -> dict:
    """Population category offsets from ALL players' history games (for the app)."""
    from datetime import date
    h = sum(p["tr_h"] for p in players)
    n = sum(p["tr_n"] for p in players)
    g = h.sum() / n.sum()
    offsets = {WEAKNESS_CATEGORIES[k]: round(float(logit(np.array([h[k] / n[k]]))[0] - logit(np.array([g]))[0]), 4)
               for k in range(K) if n[k] >= 50}
    norms = {"offsets": offsets, "shrink_strength": float(np.median(strengths)),
             "overall_hit_rate": round(float(g), 4), "players": len(players),
             "opportunities": int(n.sum()), "fitted": str(date.today()),
             "source": "scripts/research/evaluate_weakness_models.py --write-norms"}
    path = Path("src/data/category_norms.json")
    path.write_text(json.dumps(norms, indent=1), encoding="utf-8")
    return norms


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-norms", action="store_true",
                    help="also fit population category norms on all players for the app")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    players = load_players()
    n = len(players)
    print(f"{n} players with ≥ 15 analysed games")
    rf_syn = load_model(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH.exists() else None

    names = ["Global base rate", "Category base rate", "Player base rate", "Rule-based scorer",
             "RF (synthetic-trained)", "EB, shrink to player rate", "EB, shrink to population norms",
             "RF (real-data)"]
    preds = {m: np.zeros((n, K)) for m in names}
    chosen_strength = []
    folds = GroupKFold(n_splits=5)
    idx = np.arange(n)
    for tr, te in folds.split(idx, groups=idx):
        train_players = [players[i] for i in tr]
        # inner choice of hierarchical-EB strength on the training players
        best, best_ll = None, 1e9
        for s in (1, 2, 4, 8, 16, 32):
            ll = 0.0
            for q in train_players:
                pr = np.clip(pred_hier_eb(train_players, q, s), EPS, 1 - EPS)
                ll -= (q["te_miss"] * np.log(pr) + (q["te_n"] - q["te_miss"]) * np.log(1 - pr)).sum()
            if ll < best_ll:
                best, best_ll = s, ll
        chosen_strength.append(best)
        cat, g = category_rates(train_players)
        X, y, w = _rf_rows(train_players, cat, g)
        ok = ~np.isnan(y)
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=20, max_features=0.5,
                                   random_state=0, n_jobs=-1)
        rf.fit(X[ok], y[ok], sample_weight=w[ok])
        for i in te:
            p = players[i]
            preds["Global base rate"][i] = pred_global(train_players, p)
            preds["Category base rate"][i] = pred_category(train_players, p)
            preds["Player base rate"][i] = pred_player(train_players, p)
            preds["Rule-based scorer"][i] = calibrated(train_players, p, "rule")
            if rf_syn is not None:
                preds["RF (synthetic-trained)"][i] = calibrated(train_players, p, "rf", rf_syn)
            preds["EB, shrink to player rate"][i] = pred_empirical_bayes(train_players, p)
            preds["EB, shrink to population norms"][i] = pred_hier_eb(train_players, p, best)
            Xi, _, _ = _rf_rows([p], cat, g)
            preds["RF (real-data)"][i] = np.clip(rf.predict(Xi), EPS, 1 - EPS)
    if rf_syn is None:
        names.remove("RF (synthetic-trained)")

    te_miss = np.array([p["te_miss"] for p in players])
    te_n = np.array([p["te_n"] for p in players])
    total = te_n.sum()

    def per_player_ll(P):
        P = np.clip(P, EPS, 1 - EPS)
        ll = -(te_miss * np.log(P) + (te_n - te_miss) * np.log(1 - P)).sum(axis=1)
        return ll, te_n.sum(axis=1)

    # event-level vectors for AUC
    ev_y, ev_idx = [], []
    for i in range(n):
        for k in range(K):
            m, t = int(te_miss[i, k]), int(te_n[i, k])
            ev_y += [1] * m + [0] * (t - m)
            ev_idx += [(i, k)] * t
    ev_y = np.array(ev_y)
    rng = np.random.default_rng(0)
    boot = rng.integers(0, n, size=(3000, n))
    ref_name = "Category base rate"
    ref_ll, cnt = per_player_ll(preds[ref_name])
    results = {}
    for m in names:
        ll, cnt = per_player_ll(preds[m])
        P = np.clip(preds[m], EPS, 1 - EPS)
        brier = (((1 - P) ** 2) * te_miss + (P ** 2) * (te_n - te_miss)).sum() / total
        p_ev = np.array([preds[m][i, k] for i, k in ev_idx])
        diff = (ll - ref_ll)
        bs = diff[boot].sum(axis=1) / cnt[boot].sum(axis=1)
        # per-player ranking quality on categories with ≥ 3 test opportunities
        rhos, rec = [], []
        for i in range(n):
            ok = te_n[i] >= 3
            if ok.sum() >= 4:
                obs = te_miss[i, ok] / te_n[i, ok]
                pr = preds[m][i, ok]
                ra, rb = np.argsort(np.argsort(pr)), np.argsort(np.argsort(obs))
                if ra.std() > 0 and rb.std() > 0:
                    rhos.append(np.corrcoef(ra, rb)[0, 1])
                top_true = set(np.argsort(-obs)[:3])
                top_pred = set(np.argsort(-pr)[:3])
                rec.append(len(top_true & top_pred) / 3)
        results[m] = {
            "log_loss": round(float(ll.sum() / total), 4),
            "brier": round(float(brier), 4),
            "auc": round(float(roc_auc_score(ev_y, p_ev)), 4),
            "logloss_minus_category_base": round(float(diff.sum() / total), 5),
            "ci95": [round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)],
            "within_player_spearman": round(float(np.mean(rhos)), 3) if rhos else None,
            "top3_recall": round(float(np.mean(rec)), 3) if rec else None,
        }

    # split-half reliability of per-category hit rates (Spearman-Brown corrected)
    rel = []
    for k in range(K):
        a, b = [], []
        for p in players:
            (h1, n1), (h2, n2) = p["halves"]
            if n1[k] >= 3 and n2[k] >= 3:
                a.append(h1[k] / n1[k] - (h1.sum() / max(1, n1.sum())))
                b.append(h2[k] / n2[k] - (h2.sum() / max(1, n2.sum())))
        if len(a) >= 20:
            r = float(np.corrcoef(a, b)[0, 1])
            rel.append({"category": WEAKNESS_CATEGORIES[k], "players": len(a),
                        "split_half_r": round(r, 3), "spearman_brown": round(2 * r / (1 + r), 3) if r > -1 else None})

    # population parameters for the simulator
    tot_n = sum(p["tr_n"] for p in players)
    dev = []
    for p in players:
        base = p["tr_h"].sum() / max(1, p["tr_n"].sum())
        for k in range(K):
            if p["tr_n"][k] >= 5:
                r = (p["tr_h"][k] + 0.5) / (p["tr_n"][k] + 1)
                dev.append((logit(r) - logit(base), 1 / (p["tr_n"][k] * r * (1 - r))))
    dev = np.array(dev)
    true_var = max(0.0, float(dev[:, 0].var() - dev[:, 1].mean())) if len(dev) else None
    sim_params = {
        "opportunities_per_player_train": round(float(np.mean([p["tr_n"].sum() for p in players])), 1),
        "category_frequency": {WEAKNESS_CATEGORIES[k]: round(float(tot_n[k] / tot_n.sum()), 4) for k in range(K)},
        "observed_sd_of_category_logit_deviation": round(float(dev[:, 0].std()), 3) if len(dev) else None,
        "noise_corrected_delta_sd": round(math.sqrt(true_var), 3) if true_var is not None else None,
        "overall_hit_rate": round(float(sum(p["tr_h"].sum() for p in players) / tot_n.sum()), 3),
    }

    # Do weaknesses cluster within tactic families? Correlate players' category
    # deviations for category pairs within vs across the families used by the
    # IRT learner's optional family prior.
    from src.recommender.irt_model import TACTIC_FAMILIES
    fam_of = {c: fam for fam, cs in TACTIC_FAMILIES.items() for c in cs}
    devs = np.full((n, K), np.nan)
    for i, p in enumerate(players):
        base = logit(np.array([(p["tr_h"].sum() + 0.5) / (p["tr_n"].sum() + 1)]))[0]
        ok = p["tr_n"] >= 5
        devs[i, ok] = logit((p["tr_h"][ok] + 0.5) / (p["tr_n"][ok] + 1)) - base
    within, across = [], []
    for a in range(K):
        for b in range(a + 1, K):
            both = ~np.isnan(devs[:, a]) & ~np.isnan(devs[:, b])
            if both.sum() >= 20:
                r = float(np.corrcoef(devs[both, a], devs[both, b])[0, 1])
                ca, cb = WEAKNESS_CATEGORIES[a], WEAKNESS_CATEGORIES[b]
                (within if fam_of.get(ca) == fam_of.get(cb) else across).append(r)
    family_corr = ({"within": float(np.mean(within)), "across": float(np.mean(across)),
                    "pairs_within": len(within), "pairs_across": len(across)}
                   if within and across else None)

    report = {
        "players": n, "by_band": {int(b): sum(p["band"] == b for p in players) for b in range(5)},
        "family_correlation": family_corr,
        "test_opportunities": int(total), "train_fraction": TRAIN_FRACTION,
        "hier_eb_strength_chosen": chosen_strength,
        "reference_model": ref_name, "models": results,
        "reliability": rel, "simulator_calibration": sim_params,
    }
    if args.write_norms:
        report["norms_written"] = write_norms(players, chosen_strength)
    (OUT / "cohort_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("reliability",)}, indent=2))
    print("\nreliability (split-half, Spearman-Brown):")
    for r in sorted(rel, key=lambda r: -r["split_half_r"]):
        print(f"  {r['category']:18s} n={r['players']:3d} r={r['split_half_r']:+.3f} SB={r['spearman_brown']}")


if __name__ == "__main__":
    main()
