"""
Difficulty for a HUMAN of a given rating, learned from real players.

PuzzleNet's difficulty head predicts a Lichess puzzle rating: how hard a position is
for the Lichess population. The application needs something else, and the prequential
replay showed the difference painfully: mined puzzles were rated 493 points harder
than they behave, because a position from a player's own game is easier for its owner
than its content suggests (scripts/research/evaluate_difficulty_models.py).

The cohort already answers the question directly. Every one of the 40,067 critical
positions of 300 real players carries the player's rating and whether they actually
found the move. That is the same question Maia asks ("would a human of rating R play
this?"), restricted to the positions this system cares about, and measured on the
distribution the application runs on. Maia itself is the natural comparison: a model
trained on millions of human games, used here without any fitting to this cohort
beyond a calibration.

    stage 1  --build   encode every cohort position the way the labeller sees it
                       (engine line cut to NEURAL_PV_PLIES, engine mate verdict)
    stage 1b --maia    Maia's probability of the engine's best move in every cohort
                       position, at all nine Maia levels (src/neural/maia.py)
    stage 2  default   freeze PuzzleNet's trunk, train a small head on
                       [trunk activations, player rating] -> P(found it), and score
                       it against baselines with GroupKFold over PLAYERS, so no
                       player appears in both training and test

Baselines
    global base rate            one number for everyone
    player rating only          logistic regression on the rating
    category base rate          the labeller's category, one rate each
    PuzzleNet difficulty        sigma((player rating - predicted puzzle rating)/173.7),
                                i.e. using the population difficulty head as-is - the
                                thing that failed on mined puzzles
    trunk + rating (this model)
With stage 1b done, also
    Maia at the player's level  Maia's log-odds of the best move at the level nearest
                                the player's rating, calibrated by logistic regression
    Maia, all levels + rating   the nine levels' log-odds and the rating, logistic
    trunk + rating + Maia       this model with Maia's nine log-odds as extra inputs

Output: eval/neural/human_difficulty.json
Usage:  python -m scripts.neural.human_difficulty --build
        python -m scripts.neural.human_difficulty --maia
        python -m scripts.neural.human_difficulty
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import chess        # noqa: E402
import chess.pgn    # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.neural import encoding as E                       # noqa: E402
from src.puzzles.labeller import NEURAL_PV_PLIES           # noqa: E402

COHORT = Path("data/research/cohort")
CACHE = Path("data/neural/human")
OUT = Path("eval/neural/human_difficulty.json")
LOGIT_SCALE = 173.7178


# ── Stage 1: build ────────────────────────────────────────────────────────────

def _cohort_positions():
    """Every usable cohort position, in the fixed order all stages share: yields
    (engine-line row, board before the critical move, (rating, hit, band, time class)).
    That order is what keeps positions.npz and maia.npz aligned row for row."""
    from scripts.neural.relabel_cohort import load_pvs
    rows = load_pvs()
    print(f"{len(rows):,} cached engine lines", flush=True)

    # player rating and hit/miss live in the analysis files, keyed by (player, game, opp)
    meta: dict[tuple, tuple] = {}
    for path in sorted((COHORT / "analysis").glob("*.json")):
        d = json.loads(path.read_text("utf-8"))
        for gi, g in enumerate(d["games"]):
            for oi, o in enumerate(g["opportunities"]):
                meta[(path.stem, gi, oi)] = (g.get("player_rating"), bool(o["hit"]),
                                             d.get("band"), g.get("time_class"))

    pgns: dict[tuple, str] = {}
    for path in sorted((COHORT / "analysis").glob("*.json")):
        d = json.loads(path.read_text("utf-8"))
        games = json.loads((COHORT / "games" / path.name).read_text("utf-8"))["games"]
        by_end = {int(g["end_time"]): g["pgn"] for g in games if g.get("end_time")}
        for gi, g in enumerate(d["games"]):
            pgn = by_end.get(int(g["end_time"] or 0))
            if pgn:
                pgns[(path.stem, gi)] = pgn

    parsed: dict[tuple, tuple] = {}      # (player, game) -> (start board, moves), once
    for r in rows:
        key = (r["player"], r["game"], r["opp"])
        info, pgn = meta.get(key), pgns.get((r["player"], r["game"]))
        if info is None or pgn is None or not info[0] or not r["pv"]:
            continue
        if (r["player"], r["game"]) not in parsed:
            game = chess.pgn.read_game(io.StringIO(pgn))
            parsed[(r["player"], r["game"])] = (game.board(), list(game.mainline_moves()))
        start, moves = parsed[(r["player"], r["game"])]
        board = start.copy()
        for mv in moves[:r["ply"]]:
            board.push(mv)
        yield r, board, info


def build() -> None:
    bits, cont, hit, rating, player, cat = [], [], [], [], [], []
    t0 = time.time()
    for i, (r, board, info) in enumerate(_cohort_positions()):
        line = [chess.Move.from_uci(u) for u in r["pv"]][:NEURAL_PV_PLIES]
        b, c = E.encode_line(board, line, mate=r["mate"])
        bits.append(np.packbits(b))
        cont.append(c.astype(np.float16))
        hit.append(info[1])
        rating.append(info[0])
        player.append(r["player"])
        cat.append(r["stored"])
        if (i + 1) % 10_000 == 0:
            print(f"  {i + 1:,} encoded ({time.time() - t0:.0f}s)", flush=True)

    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE / "positions.npz",
        bits=np.stack(bits), cont=np.stack(cont),
        hit=np.array(hit, dtype=np.uint8), rating=np.array(rating, dtype=np.int16),
        player=np.array(player), category=np.array(cat),
    )
    print(f"{len(bits):,} positions -> {CACHE / 'positions.npz'} "
          f"({time.time() - t0:.0f}s, hit rate {np.mean(hit):.1%})")


def maia() -> None:
    """Stage 1b: Maia's probability of the engine's best move in every cohort
    position, at all nine levels. A probability lc0 prints as 0.00 % is floored."""
    from src.neural.maia import MAIA_LEVELS, Maia
    fens, best, hit, rating = [], [], [], []
    for r, board, info in _cohort_positions():
        fens.append(board.fen())
        best.append(r["pv"][0])
        hit.append(info[1])
        rating.append(info[0])
    print(f"{len(fens):,} positions x {len(MAIA_LEVELS)} Maia levels", flush=True)

    p = np.zeros((len(fens), len(MAIA_LEVELS)), dtype=np.float32)
    t0 = time.time()
    for j, level in enumerate(MAIA_LEVELS):
        with Maia(level) as net:
            for i, (fen, uci) in enumerate(zip(fens, best)):
                p[i, j] = net.probability(chess.Board(fen), chess.Move.from_uci(uci))
        print(f"  Maia-{level} done ({time.time() - t0:.0f}s, "
              f"mean P(best) {p[:, j].mean():.3f})", flush=True)

    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / "maia.npz", p=p, levels=np.array(MAIA_LEVELS, dtype=np.int16),
                        hit=np.array(hit, dtype=np.uint8),
                        rating=np.array(rating, dtype=np.int16))
    print(f"-> {CACHE / 'maia.npz'}")


# ── Stage 2: train and score ──────────────────────────────────────────────────

def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"log_loss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4),
            "brier": round(float(np.mean((p - y) ** 2)), 4),
            "auc": round(float(roc_auc_score(y, p)), 4) if 0 < y.mean() < 1 else None}


def _cluster_bootstrap(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, groups: np.ndarray,
                       reps: int = 2000, seed: int = 0) -> dict:
    """Mean log-loss difference a - b (negative: a is better) with a 95% interval from
    resampling PLAYERS, because one player's positions are not independent."""
    def loss(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    d = loss(pa) - loss(pb)
    _, inv = np.unique(groups, return_inverse=True)
    sums, counts = np.bincount(inv, weights=d), np.bincount(inv)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(reps):
        idx = rng.integers(0, len(sums), len(sums))
        stats.append(sums[idx].sum() / counts[idx].sum())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"log_loss_difference": round(float(d.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def _maia_features(d) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Maia's log-odds of the best move (n x 9) and at the player's nearest level
    (n), from stage 1b, after checking that its rows line up with positions.npz."""
    path = CACHE / "maia.npz"
    if not path.exists():
        return None
    m = np.load(path)
    if not (len(m["p"]) == len(d["hit"]) and np.array_equal(m["hit"], d["hit"])
            and np.array_equal(m["rating"], d["rating"])):
        sys.exit(f"{path} does not line up with positions.npz; rerun --maia")
    p = np.clip(m["p"].astype(np.float64), 5e-5, 1 - 1e-4)
    logit = np.log(p) - np.log1p(-p)
    levels = m["levels"].astype(float)
    rating = d["rating"].astype(float)
    nearest = np.abs(rating[:, None] - levels[None, :]).argmin(axis=1)   # ties -> lower
    covered = (rating >= levels[0] - 50) & (rating < levels[-1] + 50)
    info = {"levels": [int(v) for v in levels], "engine": "lc0 v0.32.1, nodes=1",
            "positions_within_maia_range": round(float(covered.mean()), 4),
            "mean_p_best_at_nearest_level": round(float(m["p"][np.arange(len(p)), nearest].mean()), 4)}
    return logit, logit[np.arange(len(p)), nearest], info


def evaluate(model_path: Path, folds: int = 5) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from src.neural.dataset import make_input
    from src.neural.predictor import PuzzleNetPredictor

    d = np.load(CACHE / "positions.npz", allow_pickle=False)
    bits = np.unpackbits(d["bits"], axis=1, count=E.N_BITS)
    cont = d["cont"].astype(np.float32)
    y = d["hit"].astype(float)
    rating = d["rating"].astype(np.float32)
    player, category = d["player"], d["category"]
    print(f"{len(y):,} positions, {len(set(player))} players, hit rate {y.mean():.1%}")
    maia_feats = _maia_features(d)

    net = PuzzleNetPredictor.load(model_path)
    x = make_input(bits, cont, net.cont_mean, net.cont_std, net.columns)
    trunk = []                                   # frozen 256-d representation
    for i in range(0, len(x), 8192):
        acts = net.net._trunk(x[i:i + 8192])
        trunk.append(acts[-1])
    trunk = np.concatenate(trunk)
    puzzle_rating = np.concatenate([net.net.predict(x[i:i + 8192])["rating_mean"]
                                    for i in range(0, len(x), 8192)]) * net.rating_sd + net.rating_mean
    r_std = ((rating - rating.mean()) / rating.std()).reshape(-1, 1)

    names = ["global base rate", "player rating only", "category base rate",
             "PuzzleNet puzzle difficulty", "trunk + rating (human difficulty)"]
    if maia_feats is not None:
        names += ["Maia at the player's level", "Maia, all levels + rating",
                  "trunk + rating + Maia"]
    preds = {k: np.zeros(len(y)) for k in names}
    gkf = GroupKFold(n_splits=folds)
    t0 = time.time()
    for fold, (tr, te) in enumerate(gkf.split(trunk, y, groups=player), 1):
        preds["global base rate"][te] = y[tr].mean()

        lr = LogisticRegression(max_iter=1000).fit(r_std[tr], y[tr])
        preds["player rating only"][te] = lr.predict_proba(r_std[te])[:, 1]

        rates = {c: y[tr][category[tr] == c].mean() for c in set(category[tr])}
        preds["category base rate"][te] = [rates.get(c, y[tr].mean()) for c in category[te]]

        # the population difficulty head, used the way the recommender would use it
        preds["PuzzleNet puzzle difficulty"][te] = 1 / (1 + np.exp(
            -(rating[te] - puzzle_rating[te]) / LOGIT_SCALE))

        mlp = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=60,
                            random_state=0, early_stopping=True)
        mlp.fit(np.hstack([trunk[tr], r_std[tr]]), y[tr])
        preds["trunk + rating (human difficulty)"][te] = mlp.predict_proba(
            np.hstack([trunk[te], r_std[te]]))[:, 1]

        if maia_feats is not None:
            logit, near, _ = maia_feats
            cal = LogisticRegression(max_iter=1000).fit(near[tr].reshape(-1, 1), y[tr])
            preds["Maia at the player's level"][te] = cal.predict_proba(near[te].reshape(-1, 1))[:, 1]

            mx = np.hstack([logit, r_std])
            lr_all = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            preds["Maia, all levels + rating"][te] = lr_all.fit(mx[tr], y[tr]).predict_proba(mx[te])[:, 1]

            scale = StandardScaler().fit(logit[tr])
            both = np.hstack([trunk, r_std, scale.transform(logit)])
            mlp2 = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=60,
                                 random_state=0, early_stopping=True)
            preds["trunk + rating + Maia"][te] = mlp2.fit(both[tr], y[tr]).predict_proba(both[te])[:, 1]
        print(f"  fold {fold}/{folds} done ({time.time() - t0:.0f}s)", flush=True)

    report = {"positions": int(len(y)), "players": int(len(set(player))),
              "hit_rate": round(float(y.mean()), 4), "folds": folds,
              "model": str(model_path), "pv_plies": NEURAL_PV_PLIES,
              "grouping": "GroupKFold over players: no player is in both train and test",
              "models": {k: _metrics(y, p) for k, p in preds.items()}}
    pairs = [("trunk + rating (human difficulty)", "category base rate")]
    if maia_feats is not None:
        report["maia"] = maia_feats[2]
        pairs += [("Maia, all levels + rating", "category base rate"),
                  ("Maia, all levels + rating", "trunk + rating (human difficulty)"),
                  ("trunk + rating + Maia", "trunk + rating (human difficulty)")]
    report["paired_comparisons"] = {f"{a} vs {b}": _cluster_bootstrap(y, preds[a], preds[b], player)
                                    for a, b in pairs}
    by_band = {}
    for lo, hi in ((0, 1200), (1200, 1600), (1600, 2000), (2000, 3000)):
        m = (rating >= lo) & (rating < hi)
        if m.sum() > 200:
            by_band[f"{lo}-{hi}"] = {"n": int(m.sum()), "hit_rate": round(float(y[m].mean()), 3),
                                     **{k: _metrics(y[m], p[m])["log_loss"] for k, p in preds.items()}}
    report["by_rating_band"] = by_band
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"models": report["models"],
                      "paired_comparisons": report["paired_comparisons"]}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="stage 1: encode the cohort positions")
    ap.add_argument("--maia", action="store_true", help="stage 1b: Maia probabilities")
    ap.add_argument("--model", type=Path, default=Path("data/neural/models/puzzlenet.npz"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    if args.build:
        build()
    elif args.maia:
        maia()
    else:
        evaluate(args.model, args.folds)


if __name__ == "__main__":
    main()
