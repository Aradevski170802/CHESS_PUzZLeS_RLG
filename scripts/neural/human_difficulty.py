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
distribution the application runs on.

    stage 1  --build   encode every cohort position the way the labeller sees it
                       (engine line cut to NEURAL_PV_PLIES, engine mate verdict)
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

Output: eval/neural/human_difficulty.json
Usage:  python -m scripts.neural.human_difficulty --build
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

def build() -> None:
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

    bits, cont, hit, rating, player, cat = [], [], [], [], [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        key = (r["player"], r["game"], r["opp"])
        info, pgn = meta.get(key), pgns.get((r["player"], r["game"]))
        if info is None or pgn is None or not info[0] or not r["pv"]:
            continue
        game = chess.pgn.read_game(io.StringIO(pgn))
        board = game.board()
        for j, mv in enumerate(game.mainline_moves()):
            if j == r["ply"]:
                break
            board.push(mv)
        line = [chess.Move.from_uci(u) for u in r["pv"]][:NEURAL_PV_PLIES]
        b, c = E.encode_line(board, line, mate=r["mate"])
        bits.append(np.packbits(b))
        cont.append(c.astype(np.float16))
        hit.append(info[1])
        rating.append(info[0])
        player.append(r["player"])
        cat.append(r["stored"])
        if (i + 1) % 10_000 == 0:
            print(f"  {i + 1:,}/{len(rows):,} encoded ({time.time() - t0:.0f}s)", flush=True)

    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE / "positions.npz",
        bits=np.stack(bits), cont=np.stack(cont),
        hit=np.array(hit, dtype=np.uint8), rating=np.array(rating, dtype=np.int16),
        player=np.array(player), category=np.array(cat),
    )
    print(f"{len(bits):,} positions -> {CACHE / 'positions.npz'} "
          f"({time.time() - t0:.0f}s, hit rate {np.mean(hit):.1%})")


# ── Stage 2: train and score ──────────────────────────────────────────────────

def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"log_loss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4),
            "brier": round(float(np.mean((p - y) ** 2)), 4),
            "auc": round(float(roc_auc_score(y, p)), 4) if 0 < y.mean() < 1 else None}


def evaluate(model_path: Path, folds: int = 5) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.neural_network import MLPClassifier
    from src.neural.dataset import make_input
    from src.neural.predictor import PuzzleNetPredictor

    d = np.load(CACHE / "positions.npz", allow_pickle=False)
    bits = np.unpackbits(d["bits"], axis=1, count=E.N_BITS)
    cont = d["cont"].astype(np.float32)
    y = d["hit"].astype(float)
    rating = d["rating"].astype(np.float32)
    player, category = d["player"], d["category"]
    print(f"{len(y):,} positions, {len(set(player))} players, hit rate {y.mean():.1%}")

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

    preds = {k: np.zeros(len(y)) for k in
             ("global base rate", "player rating only", "category base rate",
              "PuzzleNet puzzle difficulty", "trunk + rating (human difficulty)")}
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
        print(f"  fold {fold}/{folds} done ({time.time() - t0:.0f}s)", flush=True)

    report = {"positions": int(len(y)), "players": int(len(set(player))),
              "hit_rate": round(float(y.mean()), 4), "folds": folds,
              "model": str(model_path), "pv_plies": NEURAL_PV_PLIES,
              "grouping": "GroupKFold over players: no player is in both train and test",
              "models": {k: _metrics(y, p) for k, p in preds.items()}}
    by_band = {}
    for lo, hi in ((0, 1200), (1200, 1600), (1600, 2000), (2000, 3000)):
        m = (rating >= lo) & (rating < hi)
        if m.sum() > 200:
            by_band[f"{lo}-{hi}"] = {"n": int(m.sum()), "hit_rate": round(float(y[m].mean()), 3),
                                     **{k: _metrics(y[m], p[m])["log_loss"] for k, p in preds.items()}}
    report["by_rating_band"] = by_band
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["models"], indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="stage 1: encode the cohort positions")
    ap.add_argument("--model", type=Path, default=Path("data/neural/models/puzzlenet.npz"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    if args.build:
        build()
    else:
        evaluate(args.model, args.folds)


if __name__ == "__main__":
    main()
