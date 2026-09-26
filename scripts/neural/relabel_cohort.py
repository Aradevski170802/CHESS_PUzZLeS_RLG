"""
Does a better labeller make real players' weaknesses measurable?

The cohort evaluation (scripts/research/evaluate_weakness_models.py) found that
per-category hit rates of 300 real players have split-half reliability near zero
when positions are labelled by the rule-based tagger. Label noise attenuates
reliability, so this script asks whether PuzzleNet's labels change that.

Stage 1 (--pvs): rebuild every critical position ("opportunity") of every cohort
game from its PGN, run Stockfish on it with the analyzer's settings (MultiPV 2,
ANALYSIS_NODES nodes, 1 thread), and cache the best line and mate verdict.
Only the 40,067 critical positions are analysed, not whole games. The cache is
appended line by line and the stage is resumable: rerun it to finish what is left.

Stage 2 (default): label every cached line twice, with the rule-based tagger and
with PuzzleNet, and write two relabelled copies of the cohort analyses:

    data/research/cohort/analysis_rules_rerun/   rules on the re-run engine lines
    data/research/cohort/analysis_puzzlenet/     PuzzleNet on the same lines

Both copies use the same engine lines, so they differ only in the labeller. The
rules copy also measures how far a fresh engine run reproduces the stored labels
(the analyzer's hash table carries over between moves of a game; a fresh
analysis of one position can find a slightly different line).

Then run, for each copy:
    python -m scripts.research.evaluate_weakness_models \\
        --analysis-dir data/research/cohort/analysis_puzzlenet --out-name cohort_evaluation_puzzlenet.json

Everything under data/research stays local (third-party game data). Only the
aggregate statistics in eval/neural/cohort_relabel.json are for publication.

Usage: python -m scripts.neural.relabel_cohort --pvs [--workers 8]
       python -m scripts.neural.relabel_cohort
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import chess          # noqa: E402
import chess.engine   # noqa: E402
import chess.pgn      # noqa: E402
import numpy as np    # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.classifier.stockfish_analyzer import (ANALYSIS_NODES, MATE_CP,  # noqa: E402
                                               PV_PLIES_FOR_TAGGING, _pov_cp, find_stockfish)

COHORT = Path("data/research/cohort")
PV_CACHE = COHORT / "opportunity_pvs.jsonl"   # one JSON line per position, appended
OUT_STATS = Path("eval/neural/cohort_relabel.json")


def _positions() -> list[dict]:
    """Every opportunity with its game's PGN position, keyed for write-back."""
    items = []
    for path in sorted((COHORT / "analysis").glob("*.json")):
        d = json.loads(path.read_text("utf-8"))
        games = json.loads((COHORT / "games" / path.name).read_text("utf-8"))["games"]
        pgn_by_end = {int(g["end_time"]): g["pgn"] for g in games if g.get("end_time")}
        for gi, g in enumerate(d["games"]):
            if not g["opportunities"]:
                continue
            pgn = pgn_by_end.get(int(g["end_time"] or 0))
            if pgn is None:
                continue
            for oi, o in enumerate(g["opportunities"]):
                items.append({"player": path.stem, "game": gi, "opp": oi, "ply": o["ply"],
                              "stored": o["category"], "pgn": pgn})
    return items


def _board_at(pgn: str, ply: int) -> chess.Board:
    game = chess.pgn.read_game(io.StringIO(pgn))
    board = game.board()
    for i, mv in enumerate(game.mainline_moves()):
        if i == ply:
            break
        board.push(mv)
    return board


def _analyse(args: tuple[str, list[dict]]) -> list[dict]:
    sf, items = args
    out = []
    with chess.engine.SimpleEngine.popen_uci(sf) as engine:
        engine.configure({"Threads": 1, "Hash": 32})
        for it in items:
            board = _board_at(it["pgn"], it["ply"])
            infos = engine.analyse(board, chess.engine.Limit(nodes=ANALYSIS_NODES), multipv=2)
            if isinstance(infos, dict):
                infos = [infos]
            pv = [m.uci() for m in (infos[0].get("pv") or [])][:PV_PLIES_FOR_TAGGING]
            score = _pov_cp(infos[0]["score"], board.turn)
            out.append({k: it[k] for k in ("player", "game", "opp", "ply", "stored")}
                       | {"pv": pv, "mate": bool(score is not None and score >= MATE_CP)})
    return out


def _done_keys() -> set[tuple]:
    if not PV_CACHE.exists():
        return set()
    keys = set()
    with PV_CACHE.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:      # a line cut short by a crash
                continue
            keys.add((r["player"], r["game"], r["opp"]))
    return keys


def load_pvs() -> list[dict]:
    rows, seen = [], set()
    with PV_CACHE.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (r["player"], r["game"], r["opp"])
            if key not in seen:
                seen.add(key)
                rows.append(r)
    return rows


def compute_pvs(workers: int) -> None:
    """Analyse every critical position not already in the cache.

    Results are appended a line at a time, so an interrupted run (or an engine
    that dies, which happened once under memory pressure) loses only the chunk in
    flight: running the command again picks up where it stopped.
    """
    items = _positions()
    done = _done_keys()
    todo = [i for i in items if (i["player"], i["game"], i["opp"]) not in done]
    sf = find_stockfish()
    if not sf:
        raise SystemExit("Stockfish not found")
    print(f"{len(items):,} opportunities in {len({i['player'] for i in items})} players; "
          f"{len(done):,} cached, {len(todo):,} to do", flush=True)
    if not todo:
        return
    chunks = [todo[i::workers * 8] for i in range(workers * 8)]
    chunks = [c for c in chunks if c]
    t0, written, failed = time.time(), 0, 0
    with PV_CACHE.open("a", encoding="utf-8") as out, ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(_analyse, (sf, c)) for c in chunks]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                part = fut.result()
            except Exception as exc:          # a dead engine must not lose the run
                failed += 1
                print(f"  chunk failed ({type(exc).__name__}); rerun to finish it", flush=True)
                continue
            for r in part:
                out.write(json.dumps(r) + "\n")
            out.flush()
            written += len(part)
            if n % workers == 0 or n == len(chunks):
                rate = written / max(1e-9, time.time() - t0)
                print(f"  {written:,}/{len(todo):,} ({time.time() - t0:.0f}s, {rate:.1f}/s)", flush=True)
    print(f"cached {written:,} engine lines in {time.time() - t0:.0f}s"
          f"{f', {failed} chunk(s) failed' if failed else ''} -> {PV_CACHE}")


def relabel(cut: int) -> None:
    from src.neural.predictor import get_predictor
    from src.puzzles.tactic_tagger import tag_line
    net = get_predictor()
    if net is None:
        raise SystemExit("no PuzzleNet model installed (src/data/models/puzzlenet.npz)")
    rows = load_pvs()
    pgns = {}
    for it in _positions():
        pgns[(it["player"], it["game"], it["opp"])] = it["pgn"]

    rules, examples = [], []
    for r in rows:
        board = _board_at(pgns[(r["player"], r["game"], r["opp"])], r["ply"])
        pv = [chess.Move.from_uci(u) for u in r["pv"]]
        rules.append(tag_line(board, pv, mate=r["mate"]) if pv else "General")
        examples.append((board, pv[:cut], r["mate"]))
    neural = []
    for i in range(0, len(examples), 2000):
        neural.extend(p.category for p in net.predict_many(examples[i:i + 2000]))

    labels = {"analysis_rules_rerun": rules, "analysis_puzzlenet": neural}
    for folder, labs in labels.items():
        out_dir = COHORT / folder
        out_dir.mkdir(exist_ok=True)
        by_player: dict[str, dict] = {}
        for r, lab in zip(rows, labs):
            d = by_player.get(r["player"])
            if d is None:
                d = by_player[r["player"]] = json.loads(
                    (COHORT / "analysis" / f"{r['player']}.json").read_text("utf-8"))
            d["games"][r["game"]]["opportunities"][r["opp"]]["category"] = lab
        for path in sorted((COHORT / "analysis").glob("*.json")):
            d = by_player.get(path.stem) or json.loads(path.read_text("utf-8"))
            (out_dir / path.name).write_text(json.dumps(d), encoding="utf-8")

    stored = np.array([r["stored"] for r in rows])
    rules, neural = np.array(rules), np.array(neural)
    stats = {
        "opportunities": len(rows), "players": len({r["player"] for r in rows}),
        "nodes": ANALYSIS_NODES, "pv_cut_for_puzzlenet": cut,
        "rules_rerun_reproduces_stored_label": round(float((rules == stored).mean()), 4),
        "puzzlenet_agrees_with_rules_rerun": round(float((neural == rules).mean()), 4),
        "category_share": {
            name: {c: round(n / len(labs), 4) for c, n in Counter(labs).most_common()}
            for name, labs in (("stored_rules", stored), ("rules_rerun", rules),
                               ("puzzlenet", neural))
        },
    }
    OUT_STATS.parent.mkdir(parents=True, exist_ok=True)
    OUT_STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "category_share"}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pvs", action="store_true", help="stage 1: compute and cache engine lines")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cut", type=int, default=PV_PLIES_FOR_TAGGING,
                    help="plies of the engine line PuzzleNet sees (from engine_pv_check.py)")
    args = ap.parse_args()
    if args.pvs:
        compute_pvs(args.workers)
    else:
        relabel(args.cut)


if __name__ == "__main__":
    main()
