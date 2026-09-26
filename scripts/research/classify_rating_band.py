"""
Objective 1 of the dissertation portfolio: "a chess player classification model
that predicts player rating group ... from gameplay data with statistically
significant performance on unseen players."

Data: the 300-player research cohort (5 rating bands × 60 players), each
player's newest 60 rated games analysed by the production analyzer.

Features describe HOW a player plays, never what their rating is:
the PGN rating headers (player and opponent Elo) are deliberately excluded,
because they would leak the label.

Evaluation
───────────
* Players are the unit: every fold holds out whole players (stratified
  5-fold, repeated 10 times with different shuffles).
* Accuracy, macro-F1, within-one-band accuracy (bands are ordered), and a
  confusion matrix.
* Significance: a permutation test (labels shuffled 500 times, full CV each
  time) gives the probability of the observed accuracy under "no relation
  between play and rating band".
* Also a regression view: predict the rating itself, report MAE in rating
  points against a predict-the-mean baseline.

Usage: python -m scripts.research.classify_rating_band
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    permutation_test_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ANALYSIS = Path("data/research/cohort/analysis")
OUT = Path("eval/research")
BAND_NAMES = ["<1000", "1000–1399", "1400–1799", "1800–2199", "2200+"]

GROUPS = {
    "tactical": ["Fork", "Pin", "Skewer", "Discovered Attack", "Hanging Piece", "Deflection", "Sacrifice"],
    "mating": ["Mating Pattern", "King Safety"],
    "endgame": ["Endgame", "Rook Endgame", "Queen Endgame", "Pawn Endgame", "Bishop Endgame",
                "Knight Endgame", "Promotion"],
    "quiet": ["Quiet Move"],
}

FEATURES = [
    "mean_wp_loss", "median_wp_loss", "mean_cp_loss", "blunders_pg", "mistakes_pg", "inaccuracies_pg",
    "err_share_opening", "err_share_middlegame", "err_share_endgame", "opportunities_pg",
    "hit_rate_all", "hit_rate_tactical", "hit_rate_mating", "hit_rate_endgame", "hit_rate_quiet",
    "mean_moves", "win_rate", "share_blitz",
]


def player_features(d: dict) -> np.ndarray:
    g = d["games"]
    n = len(g)
    wp = np.array([x["avg_wp_loss"] for x in g])
    cp = np.array([min(x["avg_cp_loss"], 300.0) for x in g])
    errs = [e for x in g for e in x["errors"]]
    sev = lambda s: sum(e["severity"] == s for e in errs) / n
    ph = lambda p: sum(e["phase"] == p for e in errs) / max(1, len(errs))
    opps = [o for x in g for o in x["opportunities"]]
    overall = sum(o["hit"] for o in opps) / max(1, len(opps))

    def group_rate(cats, k=5.0):
        sel = [o for o in opps if o["category"] in cats]
        return (sum(o["hit"] for o in sel) + k * overall) / (len(sel) + k)   # shrunk to overall

    return np.array([
        wp.mean(), np.median(wp), cp.mean(), sev("blunder"), sev("mistake"), sev("inaccuracy"),
        ph("opening"), ph("middlegame"), ph("endgame"), len(opps) / n,
        overall, group_rate(GROUPS["tactical"]), group_rate(GROUPS["mating"]),
        group_rate(GROUPS["endgame"]), group_rate(GROUPS["quiet"]),
        np.mean([x["num_moves"] for x in g]),
        np.mean([1.0 if x["won"] else 0.0 for x in g]),
        np.mean([x["time_class"] == "blitz" for x in g]),
    ])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(p.read_text("utf-8")) for p in sorted(ANALYSIS.glob("*.json"))]
    rows = [r for r in rows if len(r["games"]) >= 15]
    X = np.array([player_features(r) for r in rows])
    y = np.array([r["band"] for r in rows])
    rating = np.array([r["rating"] for r in rows], dtype=float)
    print(f"{len(rows)} players, bands {np.bincount(y).tolist()}")

    models = {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0)),
        "Random forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=3, random_state=0, n_jobs=-1),
    }
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
    results = {}
    for name, model in models.items():
        accs, f1s, adj = [], [], []
        for rep in range(10):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rep)
            pred = cross_val_predict(model, X, y, cv=skf)
            accs.append(float((pred == y).mean()))
            f1s.append(float(f1_score(y, pred, average="macro")))
            adj.append(float((np.abs(pred - y) <= 1).mean()))
        skf0 = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        pred0 = cross_val_predict(model, X, y, cv=skf0)
        score, perm_scores, p = permutation_test_score(
            model, X, y, cv=skf0, n_permutations=500, random_state=0, n_jobs=-1)
        results[name] = {
            "accuracy_mean": round(float(np.mean(accs)), 4), "accuracy_sd": round(float(np.std(accs)), 4),
            "macro_f1_mean": round(float(np.mean(f1s)), 4),
            "within_one_band": round(float(np.mean(adj)), 4),
            "chance_accuracy": round(1 / len(BAND_NAMES), 4),
            "permutation_p": float(p), "permutation_null_mean": round(float(np.mean(perm_scores)), 4),
            "confusion_matrix": confusion_matrix(y, pred0).tolist(),
        }
        print(f"{name:20s} acc={np.mean(accs):.3f}±{np.std(accs):.3f}  macroF1={np.mean(f1s):.3f}  "
              f"±1 band={np.mean(adj):.3f}  permutation p={p:.4f} (null mean {np.mean(perm_scores):.3f})")

    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(X, y)
    importances = sorted(zip(FEATURES, rf.feature_importances_), key=lambda kv: -kv[1])

    # Regression view: predict the rating number itself.
    reg = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=0, n_jobs=-1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    rhat = cross_val_predict(reg, X, rating, cv=skf.split(X, y))
    mae = float(np.mean(np.abs(rhat - rating)))
    base = float(np.mean(np.abs(rating - rating.mean())))
    r = float(np.corrcoef(rhat, rating)[0, 1])
    print(f"rating regression: MAE {mae:.0f} vs baseline {base:.0f} rating points, r = {r:.3f}")

    report = {
        "players": len(rows), "classes": BAND_NAMES, "features": FEATURES,
        "leakage_guard": "PGN WhiteElo/BlackElo excluded from every feature",
        "cv": "stratified 5-fold over players, 10 repeats; permutation test 500 shuffles",
        "models": results,
        "top_features_rf": [{"feature": f_, "importance": round(float(v), 4)} for f_, v in importances[:8]],
        "rating_regression": {"mae": round(mae, 1), "baseline_mae": round(base, 1), "pearson_r": round(r, 4)},
    }
    (OUT / "rating_classification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    best = max(results, key=lambda k: results[k]["accuracy_mean"])
    cm = np.array(results[best]["confusion_matrix"], dtype=float)
    cm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5), BAND_NAMES, rotation=30)
    ax.set_yticks(range(5), BAND_NAMES)
    ax.set_xlabel("predicted rating band")
    ax.set_ylabel("true rating band")
    ax.set_title(f"{best}: accuracy {results[best]['accuracy_mean']:.1%} (chance 20%)")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "rating_band_confusion.png", dpi=150)
    print("top features:", [f"{k}={v:.3f}" for k, v in importances[:5]])


if __name__ == "__main__":
    main()
