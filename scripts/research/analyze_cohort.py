"""
Run the production game analyzer over every cohort player's games.

Uses exactly the code path the app uses (stockfish_analyzer.analyze_game:
node-limited MultiPV search, win-% grading, opportunity logging, line-level
tactic tagging), so the cohort evaluation measures the shipped system.

Parallelism is across processes, one game per task, which keeps a long
unattended run robust (one failing game cannot stall a shared thread pool).
Throughput is bounded by the CPU running the engines: a head-to-head benchmark
found threads and processes equally fast (127 s vs 129 s for 16 games).

Output: one compact JSON per player in data/research/cohort/analysis/, keyed
by the pseudonymous id. Resumable — players already analysed are skipped.

Usage: python -m scripts.research.analyze_cohort [--workers 14] [--max-games 60]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path

from src.classifier.stockfish_analyzer import analyze_game, find_stockfish

COHORT = Path("data/research/cohort")
OUT = COHORT / "analysis"


def compact(a) -> dict:
    return {
        "end_time": a.end_time, "time_class": a.time_class,
        "player_rating": a.player_rating, "opponent_rating": a.opponent_rating,
        "won": a.player_won, "num_moves": a.num_moves,
        "avg_cp_loss": round(a.avg_cp_loss, 2), "avg_wp_loss": round(a.avg_wp_loss, 3),
        "errors": [{"category": e.category, "severity": e.severity, "wp_loss": e.wp_loss,
                    "phase": e.phase} for e in a.errors],
        "opportunities": [asdict(o) for o in a.opportunities],
    }


def _one_game(args) -> tuple[str, int, dict | None]:
    pid, idx, pgn, username, sf = args
    a = analyze_game(pgn, username, sf)
    return pid, idx, (None if a.failed else compact(a))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--max-games", type=int, default=60,
                    help="newest N games per player (split later into train/test by time)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(COHORT / "analyze.log", encoding="utf-8"),
                                  logging.StreamHandler()])
    log = logging.getLogger("analyze")
    sf = find_stockfish()
    manifest = json.loads((COHORT / "manifest_private.json").read_text("utf-8"))["players"]
    pending = [(u, v) for u, v in manifest.items() if not (OUT / f"{v['id']}.json").exists()]
    # Interleave rating bands so a partially-analysed cohort stays balanced.
    by_band: dict[int, list] = {}
    for item in pending:
        by_band.setdefault(item[1]["band"], []).append(item)
    todo = []
    while any(by_band.values()):
        for band in sorted(by_band):
            if by_band[band]:
                todo.append(by_band[band].pop(0))
    log.info("%d players to analyse (%d already done), %d processes",
             len(todo), len(manifest) - len(todo), args.workers)

    def tasks():
        for username, info in todo:
            data = json.loads((COHORT / "games" / f"{info['id']}.json").read_text("utf-8"))
            games = data["games"][: args.max_games]
            yield info, len(games), [(info["id"], i, g["pgn"], username, sf) for i, g in enumerate(games)]

    results: dict[str, dict] = {}
    started = time.time()
    done_players = 0
    with ProcessPoolExecutor(args.workers) as ex:
        gen = tasks()
        inflight = set()
        exhausted = False
        while not exhausted or inflight:
            # keep about 3 tasks per worker in flight
            while not exhausted and len(inflight) < 3 * args.workers:
                try:
                    info, n, jobs = next(gen)
                except StopIteration:
                    exhausted = True
                    break
                results[info["id"]] = {"info": info, "n": n, "games": {}}
                inflight |= {ex.submit(_one_game, j) for j in jobs}
            if not inflight:
                break
            finished, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for f in finished:
                pid, idx, game = f.result()
                r = results[pid]
                r["games"][idx] = game
                if len(r["games"]) == r["n"]:
                    info = r["info"]
                    games = [g for i, g in sorted(r["games"].items()) if g is not None]
                    (OUT / f"{pid}.json").write_text(json.dumps({
                        "id": pid, "band": info["band"], "rating": info["rating"],
                        "time_class": info["time_class"], "games": games,
                    }), encoding="utf-8")
                    del results[pid]
                    done_players += 1
                    rate = (time.time() - started) / done_players
                    log.info("[%d/%d] %s band=%d games=%d opps=%d | %.0fs/player avg",
                             done_players, len(todo), pid, info["band"], len(games),
                             sum(len(g["opportunities"]) for g in games), rate)


if __name__ == "__main__":
    main()
