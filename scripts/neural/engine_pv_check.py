"""
Deployment check: label the ENGINE's principal variation, as game analysis does.

A Lichess puzzle comes with its solution line, which ends where the tactic is
complete. In game analysis the labeller sees something else: Stockfish's principal
variation from the critical position, cut to PV_PLIES_FOR_TAGGING (7) plies, plus
the engine's mate verdict. This script reproduces that on puzzles from the held-out
harness sample (seed 11, uniform draw). Stockfish analyses the position after the
opponent's move with the analyzer's own settings (MultiPV 2, ANALYSIS_NODES nodes,
1 thread), and both labellers are scored against the Lichess category on the
engine line and on the solution line of the same puzzles. PuzzleNet is also scored
on shorter cuts of the engine line (1, 3 and 5 plies), to choose the cut the
analyzer should use.

Engine lines are cached in data/neural/engine_pvs.json (gitignored), so the
labelling stage can be rerun without Stockfish.
Output: eval/neural/engine_pv_check.json

Usage: python -m scripts.neural.engine_pv_check [--n 2000] [--workers 6]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import chess                  # noqa: E402
import chess.engine           # noqa: E402
import numpy as np            # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.classifier.stockfish_analyzer import (ANALYSIS_NODES, MATE_CP,  # noqa: E402
                                               PV_PLIES_FOR_TAGGING, _pov_cp, find_stockfish)

CACHE = Path("data/neural/engine_pvs.json")
OUT = Path("eval/neural/engine_pv_check.json")
CUTS = (1, 3, 5, 7)


def _analyse_chunk(args: tuple[str, list[tuple[str, str, str]]]) -> list[dict]:
    sf_path, items = args
    out = []
    with chess.engine.SimpleEngine.popen_uci(sf_path) as engine:
        engine.configure({"Threads": 1, "Hash": 32})
        for pid, fen, moves in items:
            ms = moves.split()
            board = chess.Board(fen)
            board.push_uci(ms[0])
            infos = engine.analyse(board, chess.engine.Limit(nodes=ANALYSIS_NODES), multipv=2)
            if isinstance(infos, dict):
                infos = [infos]
            pv = [m.uci() for m in (infos[0].get("pv") or [])][:PV_PLIES_FOR_TAGGING]
            score = _pov_cp(infos[0]["score"], board.turn)
            out.append({"id": pid, "pv": pv, "mate": bool(score is not None and score >= MATE_CP),
                        "best_matches_solution": bool(pv and pv[0] == ms[1])})
    return out


def compute_pvs(n: int, workers: int) -> list[dict]:
    from scripts.neural.build_dataset import harness_samples
    from scripts.neural.evaluate_puzzlenet import HELDOUT_SEED, read_positions
    from src.neural.dataset import load_dataset
    data = load_dataset(mmap=True)
    uni, _ = harness_samples(data.cat, HELDOUT_SEED)
    rows = np.sort(np.random.default_rng(0).choice(uni, n, replace=False))
    pos = read_positions({data.puzzle_id[r] for r in rows})
    items = [(data.puzzle_id[r], pos[data.puzzle_id[r]][0], pos[data.puzzle_id[r]][1]) for r in rows]
    sf = find_stockfish()
    if not sf:
        raise SystemExit("Stockfish not found (see stockfish_analyzer.find_stockfish)")
    chunks = [items[i::workers] for i in range(workers)]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        results = [r for part in pool.map(_analyse_chunk, [(sf, c) for c in chunks]) for r in part]
    print(f"engine lines for {len(results)} puzzles in {time.time() - t0:.0f}s", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({"nodes": ANALYSIS_NODES, "rows": results}), encoding="utf-8")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", type=Path, default=Path("data/neural/models/puzzlenet.npz"))
    ap.add_argument("--pvs-only", action="store_true", help="compute and cache engine lines, then stop")
    args = ap.parse_args()

    rows = (json.loads(CACHE.read_text("utf-8"))["rows"] if CACHE.exists()
            else compute_pvs(args.n, args.workers))
    if args.pvs_only:
        return

    from scripts.neural.evaluate_puzzlenet import read_positions
    from sklearn.metrics import cohen_kappa_score
    from src.data.puzzle_loader import resolve_primary_category
    from src.neural.dataset import CATEGORIES
    from src.neural.predictor import PuzzleNetPredictor
    from src.puzzles.tactic_tagger import tag_line

    pos = read_positions({r["id"] for r in rows})
    net = PuzzleNetPredictor.load(args.model)
    truth, labels = [], {k: [] for k in
                         ["rules_solution", "rules_engine_7", "puzzlenet_solution"]
                         + [f"puzzlenet_engine_{c}" for c in CUTS]}
    examples = {k: [] for k in labels if k.startswith("puzzlenet")}
    for r in rows:
        fen, moves, themes = pos[r["id"]]
        ms = moves.split()
        board = chess.Board(fen)
        board.push_uci(ms[0])
        solution = [chess.Move.from_uci(u) for u in ms[1:]]
        pv = [chess.Move.from_uci(u) for u in r["pv"]]
        truth.append(resolve_primary_category(themes))
        labels["rules_solution"].append(tag_line(board, solution))
        labels["rules_engine_7"].append(tag_line(board, pv, mate=r["mate"]))
        examples["puzzlenet_solution"].append((board, solution, None))
        for c in CUTS:
            examples[f"puzzlenet_engine_{c}"].append((board, pv[:c], r["mate"]))
    for k, ex in examples.items():
        labels[k] = [p.category for p in net.predict_many(ex)]

    truth = np.array(truth)
    report = {"n": len(rows), "nodes": ANALYSIS_NODES,
              "engine_best_move_matches_solution": round(float(np.mean(
                  [r["best_matches_solution"] for r in rows])), 4),
              "conditions": {}}
    for k, lab in labels.items():
        lab = np.array(lab)
        report["conditions"][k] = {"strict_agreement": round(float((lab == truth).mean()), 4),
                                   "cohens_kappa": round(float(cohen_kappa_score(truth, lab)), 4)}
    agree = np.mean([r["best_matches_solution"] for r in rows])
    report["note"] = (f"The engine's first move equals the puzzle's first solution move in "
                      f"{agree:.1%} of positions; the rest are alternative wins or engine "
                      f"errors at {ANALYSIS_NODES:,} nodes, which neither labeller can fix.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
