"""
CLI puzzle quality evaluator — iterate until the generator produces good puzzles.

Usage
-----
python eval/run_evaluation.py --username chescam_sakcs
python eval/run_evaluation.py --username chescam_sakcs --elo 1050
python eval/run_evaluation.py --username chescam_sakcs --elo 1050 --depth 20
python eval/run_evaluation.py --username chescam_sakcs --no-engine   # structural only
python eval/run_evaluation.py --username chescam_sakcs --top 10      # show 10 best/worst

The report is also saved to eval/reports/{username}_{timestamp}.json so you
can compare quality across generator iterations without re-running.

Evaluation loop workflow
------------------------
1. Tweak generator constants (PUZZLE_THRESHOLD, MIN_SOLUTION_DEPTH, etc.)
2. Re-generate: POST /api/generate/puzzles  or run generate_from_games() directly
3. Re-run this script — compare avg_score between iterations
4. Repeat until avg_score >= 0.70 and engine_agrees >= 75 %
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess.engine

from eval.puzzle_evaluator import (
    CLARITY_THRESHOLD,
    DIFFICULTY_WINDOW,
    MIN_SOLUTION_DEPTH,
    PuzzleEvaluation,
    evaluate_batch,
    summarise,
)
from src.classifier.stockfish_analyzer import find_stockfish
from src.puzzles.generator import load_user_puzzles

# ── Generator parameter reference (printed in recommendations) ────────────────
_GENERATOR_PARAMS = """
Generator constants to tune  (src/puzzles/generator.py):
  PUZZLE_THRESHOLD   : min eval-drop (cp) for Stockfish mode  [current: 120, try 150-200]
  DETECT_TIME        : Stockfish time per position (s)         [current: 0.05, try 0.10]
  MAX_PER_GAME       : max puzzles per game                    [current: 3]
  CONTINUATION_MOVES : ply of continuation in Stockfish mode   [current: 4]
  SOLUTION_DEPTH     : Stockfish depth for continuation line   [current: 16]
  MIN_SOLUTION_DEPTH : heuristic quality filter (new)          [current: 3]
"""


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate quality of generated puzzles for a user",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_GENERATOR_PARAMS,
    )
    ap.add_argument("--username", required=True, help="Chess.com username")
    ap.add_argument("--elo",      type=int, default=None,
                    help="Player Elo for difficulty-gap metric")
    ap.add_argument("--depth",    type=int, default=18,
                    help="Stockfish analysis depth (default 18)")
    ap.add_argument("--top",      type=int, default=5,
                    help="Number of best/worst puzzles to show (default 5)")
    ap.add_argument("--no-engine", action="store_true",
                    help="Skip Stockfish — structural analysis only (fast)")
    ap.add_argument("--threshold", type=float, default=0.60,
                    help="Overall score threshold for 'pass' (default 0.60)")
    args = ap.parse_args()

    puzzles = load_user_puzzles(args.username)
    if not puzzles:
        print(f"No puzzles found for '{args.username}'. Run the generator first.")
        sys.exit(1)

    sf_path = None
    if not args.no_engine:
        sf_path = find_stockfish()
        if not sf_path:
            print("WARNING: Stockfish not found - falling back to structural analysis.\n")

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{'='*52}")
    print(f"  Puzzle Quality Report  -  {args.username}")
    print(f"{'='*52}")
    print(f"  Puzzles     : {len(puzzles)}", end="")
    if args.elo:
        print(f"   |   Player Elo : {args.elo}", end="")
    print()
    if sf_path:
        print(f"  Stockfish   : {Path(sf_path).name}  (depth {args.depth})")
    else:
        print("  Stockfish   : NOT USED - structural metrics only")
    print(f"  Evaluated   : {datetime.now():%Y-%m-%d %H:%M}")
    print(f"{'='*52}\n")

    # ── Run evaluation ────────────────────────────────────────────────────────
    evals: list[PuzzleEvaluation] = []
    total = len(puzzles)

    if sf_path:
        with chess.engine.SimpleEngine.popen_uci(sf_path) as engine:
            engine.configure({"Threads": 1, "Hash": 64})

            def cb(done, tot):
                bar_len = 30
                filled  = int(bar_len * done / tot)
                bar     = "#" * filled + "." * (bar_len - filled)
                print(f"\r  [{bar}] {done}/{tot}", end="", flush=True)

            evals = evaluate_batch(
                puzzles, engine,
                depth=args.depth,
                player_elo=args.elo,
                progress_callback=cb,
            )
        print("\n")
    else:
        evals = evaluate_batch(
            puzzles, None,
            player_elo=args.elo,
        )

    _print_report(evals, args)
    _save_report(args.username, evals, args)


# ── Report printer ────────────────────────────────────────────────────────────

def _print_report(evals: list[PuzzleEvaluation], args) -> None:
    n = len(evals)
    if n == 0:
        print("No evaluations produced.")
        return

    s = summarise(evals)

    def pct(x, total=n):
        return f"{x:>3}/{total}  ({100*x/total:5.1f} %)"

    sep = "-" * 52

    print("METRIC                 RESULT")
    print(sep)

    if s["engine_run"]:
        agrees = s["engine_agrees"]
        print(f"Engine agrees        : {pct(agrees)}  <- solution = engine best")
    else:
        print("Engine agrees        : (not run)")

    clarity = s["clarity_ok"]
    print(f"Clear solution       : {pct(clarity)}  <- eval gap >={CLARITY_THRESHOLD}cp")

    deep = s["deep"]
    print(f"Sufficient depth     : {pct(deep)}  <- >={MIN_SOLUTION_DEPTH} ply")

    no_dual = s["no_dual"]
    print(f"No dual solution     : {pct(no_dual)}  <- unique best move")

    tactical = s["tactical"]
    print(f"Tactical move        : {pct(tactical)}  <- capture/check/threat")

    if args.elo:
        fit = s["difficulty_fit"]
        print(f"Difficulty fit       : {pct(fit)}  <- +/-{DIFFICULTY_WINDOW} Elo of player")

    trivial = s["trivial"]
    print(f"Trivial puzzles      : {trivial:>3}/{n}  <- flag for removal")
    print(sep)

    passed = sum(1 for e in evals if e.passed(args.threshold))
    avg = s["avg_score"]
    grade = "GOOD" if avg >= 0.70 else "FAIR" if avg >= 0.50 else "POOR"
    print(f"OVERALL SCORE        :  {avg:.3f} / 1.000   [{grade}]")
    print(f"Puzzles passing {args.threshold:.2f}  :  {passed}/{n}")
    if s["avg_clarity_cp"] is not None:
        print(f"Avg clarity          :  {s['avg_clarity_cp']} cp")
    print(f"Avg solution depth   :  {s['avg_depth']:.1f} ply")
    print()

    # ── Category breakdown ────────────────────────────────────────────────────
    cats: dict[str, list[PuzzleEvaluation]] = defaultdict(list)
    for e in evals:
        cats[e.category].append(e)

    print("CATEGORY BREAKDOWN")
    print(sep)
    print(f"  {'Category':<22} {'Count':>5}  {'Elo range':<14}  {'Avg score'}")
    for cat, group in sorted(cats.items(), key=lambda x: -len(x[1])):
        ratings = [e.rating for e in group]
        avg_s   = sum(g.overall_score for g in group) / len(group)
        flag    = " (!)" if avg_s < 0.50 else ""
        print(f"  {cat:<22} {len(group):>5}  "
              f"{min(ratings)}-{max(ratings):<9}  {avg_s:.2f}{flag}")
    print()

    # ── Recommendations ───────────────────────────────────────────────────────
    print("RECOMMENDATIONS")
    print(sep)
    recs: list[str] = []

    if s["engine_run"] and s["engine_agrees"] < n * 0.70:
        pct_bad = 100 * (1 - s["engine_agrees"] / n)
        recs.append(
            f"{pct_bad:.0f}% of solutions not confirmed by Stockfish. "
            "Try raising DETECT_TIME to 0.10 s or PUZZLE_THRESHOLD to 150 cp."
        )
    if s["trivial"] > 0:
        recs.append(
            f"{s['trivial']} trivial captures detected. The generator's MIN_SOLUTION_DEPTH "
            "filter should remove these - re-run after updating generator.py."
        )
    if s["deep"] < n * 0.70:
        shortage = n - s["deep"]
        recs.append(
            f"{shortage} puzzles have depth < {MIN_SOLUTION_DEPTH} ply. "
            "Increase CONTINUATION_MOVES or add post-generation depth filter."
        )
    n_dual = n - s["no_dual"]
    if n_dual > n * 0.10:
        recs.append(
            f"{n_dual} puzzles may have dual solutions (clarity gap < 50 cp). "
            "Stockfish mode: run multi-PV check during extraction; "
            "heuristic mode: raise FORK_MIN_VALUE or PUZZLE_THRESHOLD."
        )
    if not s["engine_run"]:
        recs.append(
            "Run with Stockfish enabled (remove --no-engine) for the most "
            "important metric: engine_agrees."
        )
    if not recs:
        recs.append(
            f"Quality looks solid (avg {s['avg_score']:.2f}). "
            "Consider raising PUZZLE_THRESHOLD to 200 cp to generate harder puzzles."
        )

    for r in recs:
        print(f"  * {r}")
    print()

    # ── Top / bottom N ────────────────────────────────────────────────────────
    ranked = sorted(evals, key=lambda e: -e.overall_score)

    print(f"TOP {args.top} (highest quality)")
    print(sep)
    for e in ranked[:args.top]:
        _fmt_ev(e)

    print(f"\nBOTTOM {args.top} (lowest quality)")
    print(sep)
    for e in ranked[-args.top:][::-1]:
        _fmt_ev(e)
    print()


def _fmt_ev(e: PuzzleEvaluation) -> None:
    flags: list[str] = []
    if e.engine_agrees is True:   flags.append("[ok]engine")
    if e.engine_agrees is False:  flags.append("[!!]engine")
    if e.is_trivial:              flags.append("trivial")
    if e.has_dual_solution:       flags.append("dual?")
    if not e.is_tactical:         flags.append("non-tactical")

    clarity = f"gap={e.clarity_cp}cp" if e.clarity_cp is not None else "gap=?"
    print(
        f"  {e.puzzle_id:<16}  {e.category:<20}  "
        f"Elo={e.rating:<5}  depth={e.solution_depth}  "
        f"{clarity}  score={e.overall_score:.2f}  "
        + "  ".join(flags)
    )


# ── Save JSON report ──────────────────────────────────────────────────────────

def _save_report(
    username: str,
    evals: list[PuzzleEvaluation],
    args,
) -> None:
    out_dir = Path("eval/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path  = out_dir / f"{username.lower()}_{stamp}.json"

    report = {
        "username":       username,
        "player_elo":     args.elo,
        "engine_depth":   args.depth if not args.no_engine else None,
        "generated_at":   datetime.now().isoformat(),
        "threshold":      args.threshold,
        "summary":        summarise(evals),
        "evaluations": [
            {
                "puzzle_id":       e.puzzle_id,
                "category":        e.category,
                "rating":          e.rating,
                "solution_depth":  e.solution_depth,
                "engine_agrees":   e.engine_agrees,
                "clarity_cp":      e.clarity_cp,
                "has_dual":        e.has_dual_solution,
                "is_trivial":      e.is_trivial,
                "is_tactical":     e.is_tactical,
                "difficulty_gap":  e.difficulty_gap,
                "overall_score":   e.overall_score,
                "issues":          e.issues,
            }
            for e in evals
        ],
    }

    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report saved -> {path}\n")


if __name__ == "__main__":
    main()
