"""
Evaluate PuzzleNet against the rule-based tagger and the rating baselines.

Writes eval/neural/puzzlenet_evaluation.json with four sections:

  harness    The held-out labeller-validation sample (seed 11): exactly the puzzles
             on which the rule-based tactic_tagger scored kappa = 0.52, none of which
             PuzzleNet saw in training. The same metrics as validate_labeller.py
             (strict and lenient agreement, Cohen's kappa and macro precision on the
             uniform sample, macro recall on the stratified sample, per class), plus
             paired tests on the same puzzles: McNemar's exact test on per-puzzle
             correctness and bootstrap CIs of the differences. Also writes the
             row-normalised confusion matrix the simulation's label-noise model uses.
  test       The hash-based test split (3 %): category metrics with calibration
             (ECE before and after temperature scaling), theme metrics, and rating
             accuracy against baselines: a constant, the miner's fixed formula,
             linear regression on line length and mate, and gradient-boosted trees
             on the hand-built features.
  models     Every other trained model in data/neural/models (the ablations and the
             learning curve), on the same test split.
  by_band    Rating error by true-rating band (regression to the mean).

Usage: python -m scripts.neural.evaluate_puzzlenet [--model data/neural/models/puzzlenet.npz]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.neural.build_dataset import harness_samples                      # noqa: E402
from src.data.puzzle_loader import THEME_CATEGORIES                           # noqa: E402
from src.neural import encoding as E                                          # noqa: E402
from src.neural.dataset import (CATEGORIES, TEST, THEMES, TRAIN, VAL,         # noqa: E402
                                feature_columns, load_dataset)
from src.neural.metrics import (category_metrics, expected_calibration_error,  # noqa: E402
                                rating_metrics, theme_metrics)
from src.neural.predictor import PuzzleNetPredictor                           # noqa: E402
from src.puzzles.generator import _estimate_rating                            # noqa: E402
from src.puzzles.tactic_tagger import tag_line                                # noqa: E402

PARQUET = Path("data/processed/puzzles_full.parquet")
MODELS = Path("data/neural/models")
OUT = Path("eval/neural")
HELDOUT_SEED = 11
MEDIAN_MINED_EVAL_DROP = 312   # median evalDrop of the 151 puzzles mined from real users' games


# ── Helpers ───────────────────────────────────────────────────────────────────

def features(data, rows: np.ndarray, chunk: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    rows = np.sort(rows)
    bits = np.empty((len(rows), E.N_BITS), dtype=np.uint8)
    cont = np.empty((len(rows), E.N_CONT), dtype=np.float32)
    for i in range(0, len(rows), chunk):
        r = rows[i:i + chunk]
        bits[i:i + len(r)] = np.unpackbits(np.asarray(data.bits[r]), axis=1, count=E.N_BITS)
        cont[i:i + len(r)] = np.asarray(data.cont[r], dtype=np.float32)
    return bits, cont


def predict_rows(pred: PuzzleNetPredictor, data, rows: np.ndarray, chunk: int = 20_000) -> dict:
    """PuzzleNet outputs for sorted `rows`, in chunks (bounded memory)."""
    rows = np.sort(rows)
    parts: list[dict] = []
    for i in range(0, len(rows), chunk):
        b, c = features(data, rows[i:i + chunk])
        parts.append(pred.predict_encoded(b, c))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    """Exact McNemar test (binomial on the discordant pairs)."""
    b = int(np.sum(~a_correct & b_correct))
    c = int(np.sum(a_correct & ~b_correct))
    p = float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    return {"only_second_correct": b, "only_first_correct": c, "p_exact": p}


def bootstrap_diff(fn, n: int, rng, reps: int = 2000) -> list[float]:
    vals = [fn(rng.integers(0, n, n)) for _ in range(reps)]
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def labeller_summary(true: np.ndarray, pred: np.ndarray, lenient: np.ndarray,
                     strat_true: np.ndarray, strat_pred: np.ndarray) -> dict:
    per_class, recalls, precisions = {}, [], []
    for i, c in enumerate(CATEGORIES):
        s = strat_true == i
        u = pred == i
        rec = float((strat_pred[s] == i).mean()) if s.any() else None
        prec = float((true[u] == i).mean()) if u.any() else None
        per_class[c] = {"support_stratified": int(s.sum()), "recall": _r(rec),
                        "predicted_uniform": int(u.sum()), "precision": _r(prec)}
        if rec is not None:
            recalls.append(rec)
        if prec is not None:
            precisions.append(prec)
    return {
        "strict_agreement": _r(float((pred == true).mean())),
        "lenient_agreement": _r(float(lenient.mean())),
        "cohens_kappa": _r(float(cohen_kappa_score(true, pred))),
        "macro_recall_stratified": _r(float(np.mean(recalls))),
        "macro_precision_uniform": _r(float(np.mean(precisions))),
        "labels_never_emitted": [c for i, c in enumerate(CATEGORIES)
                                 if not ((pred == i).any() or (strat_pred == i).any())],
        "per_class": per_class,
    }


def _r(x, nd: int = 4):
    if x is None or not math.isfinite(float(x)):
        return None
    return round(float(x), nd)


def read_positions(ids: set[str]) -> dict[str, tuple[str, str, str]]:
    t = pq.read_table(PARQUET, columns=["PuzzleId", "FEN", "Moves", "Themes"],
                      filters=[("PuzzleId", "in", sorted(ids))])
    return {pid: (f, m, th) for pid, f, m, th in zip(*(t.column(c).to_pylist()
            for c in ("PuzzleId", "FEN", "Moves", "Themes")))}


def rules_label(fen: str, moves: str) -> int:
    ms = moves.split()
    import chess
    board = chess.Board(fen)
    board.push_uci(ms[0])
    return CATEGORIES.index(tag_line(board, [chess.Move.from_uci(u) for u in ms[1:]]))


def lenient_hit(pred_idx: int, true_idx: int, themes: str) -> bool:
    cats = {c for t in themes.split() for c in THEME_CATEGORIES.get(t, [])}
    return CATEGORIES[pred_idx] in cats or pred_idx == true_idx


# ── Sections ──────────────────────────────────────────────────────────────────

def harness_section(pred: PuzzleNetPredictor, data, rng) -> tuple[dict, dict]:
    uni, strat = harness_samples(data.cat, HELDOUT_SEED)
    assert np.all(data.split[uni] == 3) and np.all(data.split[strat] == 3), \
        "harness puzzles must be held out of training"
    rows = np.union1d(uni, strat)
    out = predict_rows(pred, data, rows)
    nn_all = dict(zip(rows, out["cat_probs"].argmax(axis=1)))
    ids = {data.puzzle_id[r] for r in rows}
    pos = read_positions(ids)
    rules_all = {}
    t0 = time.time()
    for r in rows:
        fen, moves, _ = pos[data.puzzle_id[r]]
        rules_all[r] = rules_label(fen, moves)
    print(f"  rules tagger on {len(rows):,} harness puzzles: {time.time() - t0:.0f}s", flush=True)

    true_u = data.cat[uni].astype(int)
    true_s = data.cat[strat].astype(int)
    themes_u = [pos[data.puzzle_id[r]][2] for r in uni]
    report = {"seed": HELDOUT_SEED, "uniform_n": int(len(uni)), "stratified_n": int(len(strat)),
              "reference": "Lichess themes resolved by MOTIF_PRIORITY (agreement with another "
                           "automatic tagger, not accuracy against human judgement)"}
    preds = {}
    for name, table in (("rules", rules_all), ("puzzlenet", nn_all)):
        pu = np.array([table[r] for r in uni])
        ps = np.array([table[r] for r in strat])
        len_u = np.array([lenient_hit(p, t, th) for p, t, th in zip(pu, true_u, themes_u)])
        report[name] = labeller_summary(true_u, pu, len_u, true_s, ps)
        preds[name] = (pu, ps)

    (ru, rs), (nu, ns) = preds["rules"], preds["puzzlenet"]
    report["paired"] = {
        "mcnemar_uniform": mcnemar(ru == true_u, nu == true_u),
        "accuracy_diff_ci95": bootstrap_diff(
            lambda i: float((nu[i] == true_u[i]).mean() - (ru[i] == true_u[i]).mean()),
            len(uni), rng),
        "kappa_diff_ci95": bootstrap_diff(
            lambda i: cohen_kappa_score(true_u[i], nu[i]) - cohen_kappa_score(true_u[i], ru[i]),
            len(uni), rng, reps=500),
        "macro_recall_diff_ci95": bootstrap_diff(
            lambda i: _macro_recall(true_s[i], ns[i]) - _macro_recall(true_s[i], rs[i]),
            len(strat), rng, reps=500),
    }
    labels = list(range(len(CATEGORIES)))
    conf = {}
    for name, (_, ps) in preds.items():
        m = confusion_matrix(true_s, ps, labels=labels).astype(float)
        conf[name] = np.round(m / np.maximum(m.sum(axis=1, keepdims=True), 1), 5).tolist()
    report["top_confusions_puzzlenet"] = _top_confusions(true_u, nu)
    return report, {"labels": CATEGORIES, **conf}


def _macro_recall(t: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean([(p[t == i] == i).mean() for i in np.unique(t)]))


def _top_confusions(t: np.ndarray, p: np.ndarray, k: int = 12) -> list[dict]:
    pairs, counts = np.unique(np.stack([t, p], axis=1)[t != p], axis=0, return_counts=True)
    order = np.argsort(-counts)[:k]
    return [{"true": CATEGORIES[pairs[i][0]], "pred": CATEGORIES[pairs[i][1]],
             "count": int(counts[i])} for i in order]


def rating_baselines(data, test_rows: np.ndarray, rng) -> dict:
    """Non-neural difficulty predictors, each with a constant predictive SD fitted
    on the validation split (so their NLL is comparable)."""
    sd = data.info["rating_sd_train"]
    mean = data.info["rating_mean_train"]
    train = rng.choice(data.indices(TRAIN), 200_000, replace=False)
    val = rng.choice(data.indices(VAL), 50_000, replace=False)
    y = {k: data.rating[r].astype(float) for k, r in (("train", train), ("val", val),
                                                       ("test", test_rows))}
    rd = {k: data.rd[r].astype(float) for k, r in (("val", val), ("test", test_rows))}

    def with_sd(name: str, pred_val: np.ndarray, pred_test: np.ndarray) -> dict:
        resid_var = max(1.0, float(np.mean((y["val"] - pred_val) ** 2 - rd["val"] ** 2)))
        m = rating_metrics(y["test"], pred_test, np.full(len(pred_test), math.sqrt(resid_var)),
                           rd["test"])
        return {"model": name, **{k: _r(v, 3) for k, v in m.items()}}

    out = {}
    out["constant"] = with_sd("training-split mean", np.full(len(val), mean),
                              np.full(len(test_rows), mean))
    formula = lambda rows: np.array([_estimate_rating(MEDIAN_MINED_EVAL_DROP, int(n))   # noqa: E731
                                     for n in data.n_moves[rows]], dtype=float)
    out["miner_formula"] = with_sd(
        f"generator._estimate_rating (eval drop fixed at the mined-puzzle median, "
        f"{MEDIAN_MINED_EVAL_DROP} cp; Lichess puzzles have no eval drop)",
        formula(val), formula(test_rows))

    # Linear regression on line length (one-hot) and the mate verdict.
    mate_col = E.BIT_NAMES.index("global.line_mate")

    def length_mate(rows):
        n = np.minimum(data.n_moves[rows], 12).astype(int)
        onehot = np.eye(13)[n][:, [2, 4, 6, 8, 10, 12]]
        bits = np.unpackbits(np.asarray(data.bits[np.sort(rows)]), axis=1, count=E.N_BITS)
        order = np.argsort(np.argsort(rows))       # back to the order of `rows`
        mate = bits[order, mate_col:mate_col + 1].astype(float)
        return np.hstack([onehot, mate, onehot * mate])

    Xtr = length_mate(train)
    coef, *_ = np.linalg.lstsq(np.hstack([Xtr, np.ones((len(Xtr), 1))]), y["train"], rcond=None)
    lin = lambda rows: np.hstack([length_mate(rows), np.ones((len(rows), 1))]) @ coef   # noqa: E731
    out["length_mate_linear"] = with_sd("least squares on line length x mate verdict",
                                        lin(val), lin(test_rows))

    # Gradient-boosted trees on the hand-built features only.
    from sklearn.ensemble import HistGradientBoostingRegressor
    cols = feature_columns("engineered")

    def eng(rows):
        """The hand-built features only, in the order of `rows`. They are the tail of
        the layout, so the columns are sliced before anything is widened to float32:
        the full 3,329-column matrix would be 2.2 GB for the test split."""
        order = np.argsort(np.argsort(rows))
        b, c = features(data, rows)
        x = np.hstack([b[:, E.OFFSETS["tactic_m1"]:].astype(np.float32), c])
        assert x.shape[1] == len(cols)
        return x[order]

    gbr = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                        random_state=0)
    t0 = time.time()
    gbr.fit(eng(train[:100_000]), (y["train"][:100_000] - mean) / sd)
    out["gbdt_engineered"] = with_sd(
        f"HistGradientBoosting on the {len(cols)} hand-built features "
        f"(100k training puzzles, {time.time() - t0:.0f}s)",
        gbr.predict(eng(val)) * sd + mean, gbr.predict(eng(test_rows)) * sd + mean)
    return out


def category_baseline_gbdt(data, test_rows: np.ndarray, rng) -> dict:
    """Gradient-boosted trees on the hand-built features, for the category task."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    cols = feature_columns("engineered")

    def eng(rows):
        b, c = features(data, rows)
        return np.hstack([b[:, E.OFFSETS["tactic_m1"]:].astype(np.float32), c])

    train = np.sort(rng.choice(data.indices(TRAIN), 100_000, replace=False))
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=63,
                                         random_state=0)
    t0 = time.time()
    clf.fit(eng(train), data.cat[train])
    probs = np.zeros((len(test_rows), len(CATEGORIES)))
    for i in range(0, len(test_rows), 20_000):
        r = test_rows[i:i + 20_000]
        probs[i:i + len(r)][:, clf.classes_] = clf.predict_proba(eng(r))
    m = category_metrics(data.cat[test_rows].astype(int), probs, len(CATEGORIES))
    return {"model": f"HistGradientBoosting on the {len(cols)} hand-built features "
                     f"(100k training puzzles, {time.time() - t0:.0f}s)",
            **{k: _r(v) for k, v in m.items()}}


def model_section(pred: PuzzleNetPredictor, data, test_rows: np.ndarray, *, full: bool) -> dict:
    out = predict_rows(pred, data, test_rows)
    y = data.cat[test_rows].astype(int)
    res = {"cat": {k: _r(v) for k, v in category_metrics(y, out["cat_probs"], len(CATEGORIES)).items()},
           "rating": {k: _r(v, 3) for k, v in rating_metrics(
               data.rating[test_rows].astype(float), out["rating"], out["rating_sd_points"],
               data.rd[test_rows].astype(float)).items()}}
    meta = pred.net.meta
    res["setup"] = {"hidden": list(pred.net.config.hidden), "heads": list(pred.net.config.heads),
                    "features": meta.get("features"), "train_size": meta.get("train_size"),
                    "epochs": meta.get("epochs"), "n_parameters": pred.net.n_parameters(),
                    "temperature": round(pred.net.temperature, 4),
                    "var_scale": round(pred.net.var_scale, 4)}
    if not full:
        return res
    # Calibration: the same logits at T = 1 vs the fitted temperature.
    t_fit = pred.net.temperature
    pred.net.temperature = 1.0
    raw = predict_rows(pred, data, test_rows)["cat_probs"]
    pred.net.temperature = t_fit
    res["calibration"] = {"ece_uncalibrated": _r(expected_calibration_error(y, raw)),
                          "ece_calibrated": _r(expected_calibration_error(y, out["cat_probs"])),
                          "reliability": _reliability(y, out["cat_probs"])}
    ty = np.unpackbits(data.themes[test_rows], axis=1, count=len(THEMES)).astype(float)
    tm = theme_metrics(ty, out["theme_probs"], THEMES)
    res["themes"] = {"bce_sum": _r(tm["bce_sum"]), "micro_f1": _r(tm["micro_f1"]),
                     "macro_ap": _r(tm["macro_ap"]),
                     "ap_per_theme": {k: _r(v, 3) for k, v in sorted(tm["ap_per_theme"].items(),
                                                                      key=lambda kv: -kv[1])}}
    per_class = {}
    pred_cls = out["cat_probs"].argmax(axis=1)
    for i, c in enumerate(CATEGORIES):
        s = y == i
        u = pred_cls == i
        per_class[c] = {"support": int(s.sum()),
                        "recall": _r(float((pred_cls[s] == i).mean())) if s.any() else None,
                        "precision": _r(float((y[u] == i).mean())) if u.any() else None}
    res["per_class"] = per_class
    bands = [(0, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 4000)]
    yr = data.rating[test_rows].astype(float)
    res["by_band"] = [{"band": f"{lo}-{hi}", "n": int(m.sum()),
                       "bias": _r(float(np.mean(out["rating"][m] - yr[m])), 1),
                       "mae": _r(float(np.mean(np.abs(out["rating"][m] - yr[m]))), 1)}
                      for lo, hi in bands if (m := (yr >= lo) & (yr < hi)).any()]
    return res


def _reliability(y: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> list[dict]:
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                         "confidence": _r(conf[m].mean()), "accuracy": _r(correct[m].mean())})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=MODELS / "puzzlenet.npz")
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(0)
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_dataset(mmap=True)
    pred = PuzzleNetPredictor.load(args.model)
    test_rows = data.indices(TEST)

    report: dict = {"model": str(args.model), "test_n": int(len(test_rows))}
    out_path = OUT / "puzzlenet_evaluation.json"

    def save() -> None:
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("harness ...", flush=True)
    report["harness"], confusion = harness_section(pred, data, rng)
    save()
    print(json.dumps({k: {m: report["harness"][k][m] for m in
                          ("strict_agreement", "cohens_kappa", "macro_recall_stratified",
                           "macro_precision_uniform")}
                      for k in ("rules", "puzzlenet")}, indent=1), flush=True)
    print("test split ...", flush=True)
    report["test"] = {"puzzlenet": model_section(pred, data, test_rows, full=True)}
    save()
    if not args.skip_baselines:
        print("baselines ...", flush=True)
        report["test"]["rating_baselines"] = rating_baselines(data, test_rows, rng)
        save()
        report["test"]["category_gbdt"] = category_baseline_gbdt(data, test_rows, rng)
        save()
    report["models"] = {}
    for path in sorted(MODELS.glob("*.npz")):
        if path.resolve() == args.model.resolve():
            continue
        print(f"model {path.stem} ...", flush=True)
        report["models"][path.stem] = model_section(PuzzleNetPredictor.load(path), data,
                                                    test_rows, full=False)
        save()
    report["seconds"] = round(time.time() - t0)
    save()
    (OUT / "label_confusion.json").write_text(json.dumps({"confusion": confusion}, indent=1),
                                              encoding="utf-8")
    print(f"done in {report['seconds']}s -> {OUT / 'puzzlenet_evaluation.json'}")


if __name__ == "__main__":
    main()
