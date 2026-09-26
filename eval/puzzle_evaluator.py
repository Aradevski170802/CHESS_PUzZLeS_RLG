"""
Puzzle quality evaluator — scores generated puzzles on five dimensions.

Dimensions (and weights in the composite score)
------------------------------------------------
engine_agrees   (0.30) — Stockfish confirms the solution is the best move
clarity_cp      (0.25) — CP gap between PV1 and PV2: ≥150 = clear, <50 = dual risk
solution_depth  (0.20) — Total ply in move sequence: ≥3 = non-trivial
not_trivial     (0.15) — Not a simple undefended-piece capture with no follow-up
difficulty_fit  (0.10) — Puzzle Elo within ±300 of the player's Elo

Typical thresholds used as targets when tuning the generator:
  clarity_cp ≥ 150    → engine_agrees ≥ 70 %   → overall_score ≥ 0.70

Run via: python eval/run_evaluation.py --username <name> --elo <elo>
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import chess
import chess.engine

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
CLARITY_THRESHOLD  = 150   # cp gap ≥ this → puzzle has a clear single answer
DUAL_THRESHOLD     = 50    # cp gap < this → likely dual solution
TRIVIAL_THRESHOLD  = 400   # cp gap this large + no recapture → trivially obvious
DIFFICULTY_WINDOW  = 300   # |puzzle_rating − player_elo| ≤ this → good fit
MIN_SOLUTION_DEPTH = 3     # ply count (incl. opp move): ≥3 to avoid pure 1-movers
MATE_CP            = 9_000 # substitute for forced-mate scores

_PIECE_VAL = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0,
}


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class PuzzleEvaluation:
    puzzle_id: str
    fen: str            # setup FEN (position before opp's last move)
    moves: list[str]    # [opp_move, solution_move, optional_continuation…]
    rating: int
    category: str

    # Engine results (None when engine not run)
    engine_best_move: Optional[str] = None
    engine_agrees: Optional[bool]   = None   # None = engine not run
    pv1_cp: Optional[int]           = None
    clarity_cp: Optional[int]       = None   # PV1 score − PV2 score
    has_dual_solution: bool         = False

    # Structural
    solution_depth: int = 0       # len(moves)
    is_tactical: bool   = False   # capture / check / checkmate
    is_trivial: bool    = False   # free undefended piece, no follow-up

    # Player-fitness
    player_elo: Optional[int]     = None
    difficulty_gap: Optional[int] = None
    difficulty_fits: bool         = True

    # Bandit relevance
    category_is_weak: Optional[bool] = None   # None = no bandit data

    # Composite
    overall_score: float      = 0.0
    issues: list[str]         = field(default_factory=list)

    def passed(self, threshold: float = 0.60) -> bool:
        return self.overall_score >= threshold


# ── Main entry point ──────────────────────────────────────────────────────────

def evaluate_puzzle(
    puzzle: dict,
    engine: Optional[chess.engine.SimpleEngine],
    *,
    depth: int = 18,
    player_elo: Optional[int] = None,
    weak_categories: Optional[set[str]] = None,
) -> PuzzleEvaluation:
    """
    Score a single puzzle.

    Parameters
    ----------
    puzzle          : Lichess-schema dict (FEN, Moves, Rating, PrimaryCategory …)
    engine          : open Stockfish engine, or None for structural-only analysis
    depth           : Stockfish depth (default 18 for reliable multi-PV eval)
    player_elo      : player's current Elo (used for difficulty_gap check)
    weak_categories : categories where player solve_rate < 0.60 (from bandit)

    Returns
    -------
    PuzzleEvaluation dataclass — call .overall_score for the composite.
    """
    # ── Unpack fields ─────────────────────────────────────────────────────────
    fen      = puzzle.get("FEN") or puzzle.get("fen", "")
    moves_raw = puzzle.get("Moves") or puzzle.get("moves", "")
    moves    = moves_raw.split() if isinstance(moves_raw, str) else list(moves_raw or [])
    pid      = puzzle.get("PuzzleId") or puzzle.get("id", "?")
    rating   = int(puzzle.get("Rating") or puzzle.get("rating") or 1200)
    category = puzzle.get("PrimaryCategory") or puzzle.get("primaryCategory", "General")

    ev = PuzzleEvaluation(
        puzzle_id=pid, fen=fen, moves=moves,
        rating=rating, category=category,
        player_elo=player_elo,
    )

    if len(moves) < 2:
        ev.issues.append("fewer than 2 moves in solution")
        return ev

    # ── Apply opponent's move to reach the actual puzzle position ─────────────
    try:
        board = chess.Board(fen)
        opp_move = chess.Move.from_uci(moves[0])
        if opp_move not in board.legal_moves:
            ev.issues.append("illegal opponent move in puzzle")
            return ev
        board.push(opp_move)
    except Exception as exc:
        ev.issues.append(f"FEN/move error: {exc}")
        return ev

    intended = moves[1]   # the move we expect the player to find

    # ── Structural: solution depth ────────────────────────────────────────────
    ev.solution_depth = len(moves)

    # ── Structural: is the intended move tactical? ────────────────────────────
    try:
        sol_move = chess.Move.from_uci(intended)
        if sol_move in board.legal_moves:
            target = board.piece_at(sol_move.to_square)
            b2 = board.copy()
            b2.push(sol_move)
            ev.is_tactical = bool(target or b2.is_check() or b2.is_checkmate())
    except Exception:
        pass

    # ── Structural: triviality check ──────────────────────────────────────────
    # Trivial = captures a piece that was completely undefended, no follow-up
    try:
        sol_move = chess.Move.from_uci(intended)
        if sol_move in board.legal_moves:
            target = board.piece_at(sol_move.to_square)
            if target is not None:
                opp_side = not board.turn
                # Is the target defended at all before we capture?
                defended = board.is_attacked_by(opp_side, sol_move.to_square)
                if not defended and ev.solution_depth <= 3:
                    ev.is_trivial = True
                    ev.issues.append("trivial: undefended piece, ≤3-ply solution")
    except Exception:
        pass

    # ── Difficulty fitness ────────────────────────────────────────────────────
    if player_elo is not None:
        ev.difficulty_gap  = abs(rating - player_elo)
        ev.difficulty_fits = ev.difficulty_gap <= DIFFICULTY_WINDOW

    # ── Category relevance ────────────────────────────────────────────────────
    if weak_categories is not None:
        ev.category_is_weak = category in weak_categories

    # ── Engine analysis ───────────────────────────────────────────────────────
    if engine is not None:
        try:
            infos = engine.analyse(
                board,
                chess.engine.Limit(depth=depth),
                multipv=3,
            )

            scores: list[int] = []
            for info in infos:
                sc = info.get("score")
                if sc is not None:
                    pov = sc.pov(board.turn)
                    if pov.is_mate():
                        m = pov.mate()
                        scores.append(MATE_CP if (m and m > 0) else -MATE_CP)
                    else:
                        v = pov.score()
                        if v is not None:
                            scores.append(v)

            if scores:
                ev.pv1_cp = scores[0]

            # Best move
            if infos:
                pv = infos[0].get("pv") or []
                if pv:
                    ev.engine_best_move = pv[0].uci()
                    ev.engine_agrees    = (ev.engine_best_move == intended)

            # Clarity and dual-solution detection
            if len(scores) >= 2:
                ev.clarity_cp         = scores[0] - scores[1]
                ev.has_dual_solution  = ev.clarity_cp < DUAL_THRESHOLD

        except Exception as exc:
            ev.issues.append(f"engine error: {exc}")

    # ── Composite score ───────────────────────────────────────────────────────
    score = 0.0

    # 1. Engine validity (0.30)
    if ev.engine_agrees is True:
        score += 0.30
    elif ev.engine_agrees is False:
        ev.issues.append("Stockfish disagrees — solution may not be optimal")
    # None (engine not run) → contribution skipped

    # 2. Clarity (0.25)
    if ev.clarity_cp is not None:
        if ev.clarity_cp >= CLARITY_THRESHOLD:
            score += 0.25
        elif ev.clarity_cp >= 80:
            score += 0.12
            ev.issues.append(f"low clarity {ev.clarity_cp}cp (want ≥{CLARITY_THRESHOLD})")
        else:
            ev.issues.append(f"very low clarity {ev.clarity_cp}cp — dual solution risk")

    # 3. Sufficient depth (0.20)
    if ev.solution_depth >= MIN_SOLUTION_DEPTH:
        score += 0.20
    else:
        ev.issues.append(f"depth {ev.solution_depth} too shallow (want ≥{MIN_SOLUTION_DEPTH})")

    # 4. Non-trivial (0.15)
    if not ev.is_trivial:
        score += 0.15

    # 5. Difficulty fit (0.10)
    if ev.difficulty_fits:
        score += 0.10
    elif ev.difficulty_gap is not None:
        ev.issues.append(
            f"difficulty mismatch: puzzle {rating} vs player {player_elo} "
            f"(gap {ev.difficulty_gap})"
        )

    ev.overall_score = round(score, 3)
    return ev


# ── Batch evaluation ──────────────────────────────────────────────────────────

def evaluate_batch(
    puzzles: list[dict],
    engine: Optional[chess.engine.SimpleEngine],
    *,
    depth: int = 18,
    player_elo: Optional[int] = None,
    weak_categories: Optional[set[str]] = None,
    progress_callback=None,
) -> list[PuzzleEvaluation]:
    """
    Evaluate a list of puzzles with the same engine instance.

    progress_callback(done, total) is called after each puzzle if provided.
    """
    results: list[PuzzleEvaluation] = []
    total = len(puzzles)
    for i, p in enumerate(puzzles):
        ev = evaluate_puzzle(
            p, engine,
            depth=depth,
            player_elo=player_elo,
            weak_categories=weak_categories,
        )
        results.append(ev)
        if progress_callback:
            progress_callback(i + 1, total)
    return results


# ── Summary helper ────────────────────────────────────────────────────────────

def summarise(evals: list[PuzzleEvaluation]) -> dict:
    """Return a concise summary dict for serialisation or display."""
    n = len(evals)
    if n == 0:
        return {"total": 0}

    engine_run = [e for e in evals if e.engine_agrees is not None]
    agrees     = sum(1 for e in engine_run if e.engine_agrees)
    clear      = sum(1 for e in evals if e.clarity_cp is not None and e.clarity_cp >= CLARITY_THRESHOLD)
    deep       = sum(1 for e in evals if e.solution_depth >= MIN_SOLUTION_DEPTH)
    no_dual    = sum(1 for e in evals if not e.has_dual_solution)
    trivial    = sum(1 for e in evals if e.is_trivial)
    tactical   = sum(1 for e in evals if e.is_tactical)
    fit        = sum(1 for e in evals if e.difficulty_fits)

    avg_score   = sum(e.overall_score for e in evals) / n
    avg_depth   = sum(e.solution_depth for e in evals) / n
    clarity_vals = [e.clarity_cp for e in evals if e.clarity_cp is not None]
    avg_clarity  = sum(clarity_vals) / len(clarity_vals) if clarity_vals else None

    return {
        "total":           n,
        "engine_run":      len(engine_run),
        "engine_agrees":   agrees,
        "clarity_ok":      clear,
        "deep":            deep,
        "no_dual":         no_dual,
        "trivial":         trivial,
        "tactical":        tactical,
        "difficulty_fit":  fit,
        "avg_score":       round(avg_score, 3),
        "avg_depth":       round(avg_depth, 1),
        "avg_clarity_cp":  round(avg_clarity) if avg_clarity is not None else None,
    }
