"""
Re-check stored mined puzzles against the generator's stricter uniqueness rule.

Puzzles mined before the change had only their first move checked for a second
solution, on a 0.1 s search. The generator now requires every player move to beat
the engine's second choice by MIN_CLARITY_CP, and re-checks the first move at
VERIFY_DEPTH (src/puzzles/generator.py). This replays every stored mined puzzle with
Stockfish under the same rule:

  first move not unique at VERIFY_DEPTH    -> reported as failing; with --apply it
                                              moves to data/user_puzzles/retired/
  a later player move not unique at        -> with --apply the line is cut so it
    SOLUTION_DEPTH                            ends on the last unique player move
                                              (the full line is kept in originalMoves)

A player move counts as unique only if it is still the engine's first choice and
beats the second by MIN_CLARITY_CP. Dry run by default: nothing is written except the
report. With --apply each changed file is backed up first (<user>.json.bak).

Output: eval/research/mined_uniqueness.json
Usage:  python scripts/revalidate_mined_puzzles.py [--apply]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classifier.stockfish_analyzer import find_stockfish          # noqa: E402
from src.puzzles.generator import (                                   # noqa: E402
    MIN_CLARITY_CP, SOLUTION_DEPTH, USER_PUZZLES_DIR, VERIFY_DEPTH,
    _difficulty_tier, _estimate_rating, _is_unique_best, _pov_cp,
)

OUT = Path("eval/research/mined_uniqueness.json")


def check_line(engine: chess.engine.SimpleEngine, fen: str, moves: list[str]) -> dict:
    """Replay one puzzle. Returns how many of its moves survive the rule and the
    engine's margin at every player move (None when the stored move is no longer
    the engine's first choice)."""
    board = chess.Board(fen)
    board.push_uci(moves[0])                         # the opponent's setup move
    keep, margins = 1, []
    for i in range(1, len(moves), 2):                # player moves at odd indices
        depth = VERIFY_DEPTH if i == 1 else SOLUTION_DEPTH
        infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=2)
        first = (infos[0].get("pv") or [None])[0] if infos else None
        same = first is not None and first.uci() == moves[i]
        if same and len(infos) >= 2:
            margins.append(_pov_cp(infos[0]["score"], board.turn)
                           - _pov_cp(infos[1]["score"], board.turn))
        else:
            margins.append(None if not same else 10_000)   # single legal move
        if not (same and _is_unique_best(infos, board.turn)):
            break
        keep = i + 1                                 # this player move is unique
        board.push_uci(moves[i])
        if i + 1 < len(moves):
            board.push_uci(moves[i + 1])             # the opponent's forced reply
    return {"keep": keep if keep > 1 else 0, "margins": margins}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="cut ambiguous lines and retire puzzles whose first move fails")
    args = ap.parse_args()

    path = find_stockfish()
    if not path:
        sys.exit("Stockfish not found (see src/classifier/stockfish_analyzer.py)")

    report = {"rule": f"every player move beats the second choice by {MIN_CLARITY_CP} cp "
                      f"(first move at depth {VERIFY_DEPTH}, later moves at depth "
                      f"{SOLUTION_DEPTH})",
              "users": {}}
    with chess.engine.SimpleEngine.popen_uci(path) as engine:
        engine.configure({"Threads": 1, "Hash": 64})
        # The report is published, so users are numbered rather than named.
        for n, file in enumerate(sorted(USER_PUZZLES_DIR.glob("*.json")), start=1):
            alias = f"user_{n}"
            puzzles = json.loads(file.read_text("utf-8"))
            kept, retired, stats = [], [], {"puzzles": 0, "multi_move": 0, "unchanged": 0,
                                            "cut": 0, "first_move_fails": 0}
            for p in puzzles:
                moves = p.get("Moves", "").split()
                if p.get("source") != "generated" or len(moves) < 2:
                    kept.append(p)
                    continue
                stats["puzzles"] += 1
                stats["multi_move"] += len(moves) > 2
                res = check_line(engine, p["FEN"], moves)
                entry = report["users"].setdefault(alias, {"stats": stats, "puzzles": []})
                entry["puzzles"].append({"id": p["PuzzleId"], "moves": len(moves), **res})
                if res["keep"] == 0:
                    stats["first_move_fails"] += 1
                    retired.append(p)
                    print(f"  {file.stem}: {p['PuzzleId']} first move not unique "
                          f"(margins {res['margins']})")
                    continue
                if res["keep"] < len(moves):
                    stats["cut"] += 1
                    print(f"  {file.stem}: {p['PuzzleId']} cut {len(moves)} -> "
                          f"{res['keep']} moves (margins {res['margins']})")
                    if args.apply:
                        # The logged attempts were made on the full line; keep it so
                        # evaluations of those attempts still see what was served.
                        p.setdefault("originalMoves", p["Moves"])
                        p["Moves"] = " ".join(moves[:res["keep"]])
                        p["playerMoves"] = res["keep"] // 2
                        # The formula rating grows with line length; PuzzleNet's
                        # estimate was made on the old line and is redone on load.
                        if p.get("ratingModel", "heuristic") == "heuristic" and "evalDrop" in p:
                            p["Rating"] = _estimate_rating(p["evalDrop"], res["keep"])
                            p["DifficultyTier"] = _difficulty_tier(p["Rating"])
                        p["puzzlenetRating"] = p["puzzlenetRd"] = None
                else:
                    stats["unchanged"] += 1
                kept.append(p)
            report["users"].setdefault(alias, {"stats": stats, "puzzles": []})
            print(f"{file.stem}: {stats}")

            if args.apply and (stats["cut"] or retired):
                shutil.copy2(file, file.with_suffix(".json.bak"))
                file.write_text(json.dumps(kept, indent=2), encoding="utf-8")
                if retired:
                    rdir = USER_PUZZLES_DIR / "retired"
                    rdir.mkdir(exist_ok=True)
                    (rdir / file.name).write_text(json.dumps(retired, indent=2),
                                                  encoding="utf-8")

    totals = {k: sum(u["stats"][k] for u in report["users"].values())
              for k in ("puzzles", "multi_move", "unchanged", "cut", "first_move_fails")}
    report["totals"] = totals
    report["applied"] = args.apply
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("totals:", totals, "| applied" if args.apply else "| dry run, nothing changed")


if __name__ == "__main__":
    main()
