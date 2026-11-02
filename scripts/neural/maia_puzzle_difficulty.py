"""
Puzzle difficulty for players in general: PuzzleNet against Maia on held-out
Lichess puzzles.

PuzzleNet's difficulty head predicts a puzzle's Lichess rating from the position and
its solution line. Maia (src/neural/maia.py) gives a second route to the same number:
a puzzle whose solution humans of every rating are unlikely to play is a hard one.
This compares the two on a random sample of PuzzleNet's TEST split, puzzles the
network never trained on:

    mean rating         one number for every puzzle (its RMSE is the ratings' spread)
    PuzzleNet           the deployed network's rating prediction, no fitting here
    Maia                ridge regression on Maia's log-probability of the solution
                        (first move, and every solver move together) at all nine
                        levels, plus the number of solver moves
    PuzzleNet + Maia    the same regression with PuzzleNet's prediction added

The regressions are fitted with 5-fold cross-validation inside the sample, so every
puzzle is scored by a model that never saw it. Lichess's own ratings carry noise (each
puzzle has a rating deviation), so no model can reach an RMSE of zero. Differences in
RMSE get a paired bootstrap 95% interval over puzzles.

The Maia and PuzzleNet features are cached (data/neural/maia_puzzles.npz); --reuse
scores them again without re-running the engines.

Output: eval/neural/maia_puzzle_difficulty.json
Usage:  python -m scripts.neural.maia_puzzle_difficulty [--n 10000] [--reuse]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import chess        # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.neural.dataset import TEST                         # noqa: E402
from src.neural.maia import MAIA_LEVELS, Maia               # noqa: E402

META = Path("data/neural/meta.npz")
PARQUET = Path("data/processed/puzzles_full.parquet")
CACHE = Path("data/neural/maia_puzzles.npz")
OUT = Path("eval/neural/maia_puzzle_difficulty.json")


def sample_puzzles(n: int, seed: int) -> list[dict]:
    import pyarrow.parquet as pq
    meta = np.load(META)
    test_ids = meta["puzzle_id"][meta["split"] == TEST]
    ids = set(np.random.default_rng(seed).choice(test_ids, size=n, replace=False).tolist())
    table = pq.read_table(PARQUET, columns=["PuzzleId", "FEN", "Moves", "Rating"],
                          filters=[("PuzzleId", "in", ids)])
    rows = table.to_pylist()
    rows.sort(key=lambda r: r["PuzzleId"])
    return rows


def solver_positions(fen: str, moves: str) -> list[tuple[str, str]]:
    """(position, move) for every solver move: Moves[0] is the opponent's setup move,
    then the solver and the opponent alternate."""
    ms = moves.split()
    board = chess.Board(fen)
    board.push_uci(ms[0])
    out = []
    for i, uci in enumerate(ms[1:]):
        if i % 2 == 0:
            out.append((board.fen(), uci))
        board.push_uci(uci)
    return out


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    from scipy.stats import spearmanr
    err = pred - y
    return {"rmse": round(float(np.sqrt(np.mean(err ** 2))), 1),
            "mae": round(float(np.mean(np.abs(err))), 1),
            "r2": round(float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)), 4),
            "spearman": round(float(spearmanr(y, pred).statistic), 4)}


def _paired_rmse(y: np.ndarray, a: np.ndarray, b: np.ndarray, reps: int = 2000,
                 seed: int = 0) -> dict:
    """RMSE(a) - RMSE(b) with a paired bootstrap 95% interval over puzzles
    (negative: a is the better model)."""
    ea, eb = (a - y) ** 2, (b - y) ** 2
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(reps):
        idx = rng.integers(0, len(y), len(y))
        diffs.append(np.sqrt(ea[idx].mean()) - np.sqrt(eb[idx].mean()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"rmse_difference": round(float(np.sqrt(ea.mean()) - np.sqrt(eb.mean())), 1),
            "ci95": [round(float(lo), 1), round(float(hi), 1)]}


def compute_features(n: int, seed: int) -> dict:
    from src.neural.predictor import get_predictor
    t0 = time.time()
    rows = sample_puzzles(n, seed)
    steps = [solver_positions(r["FEN"], r["Moves"]) for r in rows]
    print(f"{len(rows):,} test puzzles, {sum(map(len, steps)):,} solver moves "
          f"({time.time() - t0:.0f}s)", flush=True)

    net = get_predictor()
    if net is None:
        sys.exit("PuzzleNet model not found")
    examples = []
    for r in rows:
        ms = r["Moves"].split()
        board = chess.Board(r["FEN"])
        board.push_uci(ms[0])
        examples.append((board, [chess.Move.from_uci(u) for u in ms[1:]], None))
    puzzlenet = np.concatenate([[p.rating for p in net.predict_many(examples[i:i + 2048])]
                                for i in range(0, len(examples), 2048)])
    print(f"PuzzleNet done ({time.time() - t0:.0f}s)", flush=True)

    first = np.zeros((len(rows), len(MAIA_LEVELS)))
    line = np.zeros((len(rows), len(MAIA_LEVELS)))
    for j, level in enumerate(MAIA_LEVELS):
        with Maia(level) as maia:
            for i, puzzle_steps in enumerate(steps):
                logs = [np.log(maia.probability(chess.Board(fen), chess.Move.from_uci(uci)))
                        for fen, uci in puzzle_steps]
                first[i, j], line[i, j] = logs[0], sum(logs)
        print(f"  Maia-{level} done ({time.time() - t0:.0f}s)", flush=True)

    feats = {"ids": np.array([r["PuzzleId"] for r in rows]),
             "y": np.array([r["Rating"] for r in rows], dtype=float),
             "puzzlenet": puzzlenet, "first": first, "line": line,
             "n_moves": np.array([len(s) for s in steps], dtype=float)}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **feats)
    return feats


def main() -> None:
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--reuse", action="store_true", help="score the cached features again")
    args = ap.parse_args()

    t0 = time.time()
    if args.reuse and CACHE.exists():
        feats = dict(np.load(CACHE))
    else:
        feats = compute_features(args.n, args.seed)
    y, puzzlenet, first, line = feats["y"], feats["puzzlenet"], feats["first"], feats["line"]
    n_moves = feats["n_moves"].reshape(-1, 1)
    x_maia = np.hstack([first, line, n_moves])
    x_both = np.hstack([x_maia, puzzlenet.reshape(-1, 1)])
    preds = {"mean rating": np.zeros(len(y)), "PuzzleNet": puzzlenet,
             "Maia": np.zeros(len(y)), "PuzzleNet + Maia": np.zeros(len(y))}
    for tr, te in KFold(5, shuffle=True, random_state=args.seed).split(x_maia):
        preds["mean rating"][te] = y[tr].mean()
        for name, x in (("Maia", x_maia), ("PuzzleNet + Maia", x_both)):
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
            preds[name][te] = model.fit(x[tr], y[tr]).predict(x[te])

    # How Maia's view of a puzzle changes with rating: P(first move) by level
    by_level = {str(lv): round(float(np.exp(first[:, j]).mean()), 4)
                for j, lv in enumerate(MAIA_LEVELS)}
    bands = {}
    for lo, hi in ((0, 1200), (1200, 1600), (1600, 2000), (2000, 3500)):
        m = (y >= lo) & (y < hi)
        bands[f"{lo}-{hi}"] = {"n": int(m.sum()),
                               **{k: _metrics(y[m], p[m])["rmse"] for k, p in preds.items()}}
    report = {"puzzles": int(len(y)), "split": "test (never used to train PuzzleNet)",
              "seed": args.seed, "solver_moves": int(feats["n_moves"].sum()),
              "maia": {"levels": list(MAIA_LEVELS), "engine": "lc0 v0.32.1, nodes=1",
                       "mean_p_first_move_by_level": by_level},
              "models": {k: _metrics(y, p) for k, p in preds.items()},
              "paired_comparisons": {
                  "PuzzleNet + Maia vs PuzzleNet": _paired_rmse(y, preds["PuzzleNet + Maia"],
                                                                puzzlenet),
                  "Maia vs PuzzleNet": _paired_rmse(y, preds["Maia"], puzzlenet)},
              "rmse_by_rating_band": bands,
              "seconds": round(time.time() - t0)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"models": report["models"],
                      "paired_comparisons": report["paired_comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
