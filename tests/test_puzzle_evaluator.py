"""Unit tests for eval/puzzle_evaluator.py — structural-only (engine=None), so these
run without a Stockfish binary. Every FEN/move pair below has been hand-verified for
legality and the described tactical outcome; see comments for the reasoning.
"""

import chess

from eval.puzzle_evaluator import (
    DIFFICULTY_WINDOW,
    MIN_SOLUTION_DEPTH,
    evaluate_batch,
    evaluate_puzzle,
    summarise,
)

START_FEN = chess.Board().fen()

# Black king g8, boxed in by its own pawns f7/g7/h7; white rook a1, white king g1;
# black has a spare pawn on b7 (NOT a7 - the a-file must stay clear for the rook) so
# it has a legal "waiting" move before white mates. Black to move first (b7b6), then
# white plays a1a8 which is checkmate (back-rank mate): after a1a8, the rook attacks
# the whole 8th rank up to g8; the king's only neighbours are f8/h8 (both swept by the
# rook) and f7/g7/h7 (occupied by its own pawns).
MATE_SETUP_FEN = "6k1/1p3ppp/8/8/8/8/8/R5K1 b - - 0 1"
MATE_MOVES = "b7b6 a1a8"

# Black king e8, spare pawn a7 for a waiting move, black knight e2 undefended and
# adjacent to the white king e1. Black plays a7a6 (waiting), white plays Kxe2 — a
# free, undefended capture with no follow-up, i.e. a "trivial" puzzle by the
# evaluator's own definition (undefended piece, solution depth <= 3).
TRIVIAL_SETUP_FEN = "4k3/p7/8/8/8/8/4n3/4K3 b - - 0 1"
TRIVIAL_MOVES = "a7a6 e1e2"


def _puzzle(fen, moves, **overrides):
    base = {
        "PuzzleId": "test-puzzle",
        "FEN": fen,
        "Moves": moves,
        "Rating": 1200,
        "PrimaryCategory": "General",
    }
    base.update(overrides)
    return base


class TestStructuralGuards:
    def test_fewer_than_two_moves_is_rejected(self):
        ev = evaluate_puzzle(_puzzle(START_FEN, "e2e4"), engine=None)
        assert ev.overall_score == 0.0
        assert "fewer than 2 moves in solution" in ev.issues

    def test_illegal_opponent_move_is_rejected(self):
        # e2e5 is not a legal pawn move (three squares) from the start position.
        ev = evaluate_puzzle(_puzzle(START_FEN, "e2e5 e7e5"), engine=None)
        assert ev.overall_score == 0.0
        assert "illegal opponent move in puzzle" in ev.issues


class TestMatingPuzzle:
    def test_structural_fields(self):
        ev = evaluate_puzzle(
            _puzzle(MATE_SETUP_FEN, MATE_MOVES, PrimaryCategory="Mating Pattern"),
            engine=None,
            player_elo=1200,
        )
        assert ev.solution_depth == 2
        assert ev.is_tactical is True     # delivers check
        assert ev.is_trivial is False     # no capture at all, so the trivial check never fires
        assert ev.difficulty_fits is True  # rating 1200 == player_elo 1200

    def test_composite_score_with_matching_elo(self):
        # engine=None -> 0, clarity None -> 0, depth 2 < MIN_SOLUTION_DEPTH(3) -> 0,
        # not trivial -> +0.15, difficulty fits -> +0.10  =>  0.25
        ev = evaluate_puzzle(
            _puzzle(MATE_SETUP_FEN, MATE_MOVES, PrimaryCategory="Mating Pattern"),
            engine=None,
            player_elo=1200,
        )
        assert ev.overall_score == pytest_approx_or_exact(0.25)
        assert any("too shallow" in issue for issue in ev.issues)

    def test_composite_score_with_mismatched_elo(self):
        # Same puzzle, but player_elo far from the puzzle rating: loses the 0.10
        # difficulty-fit bonus and difficulty_gap should exceed DIFFICULTY_WINDOW.
        ev = evaluate_puzzle(
            _puzzle(MATE_SETUP_FEN, MATE_MOVES, PrimaryCategory="Mating Pattern"),
            engine=None,
            player_elo=2000,
        )
        assert ev.difficulty_gap == 800
        assert ev.difficulty_gap > DIFFICULTY_WINDOW
        assert ev.difficulty_fits is False
        assert ev.overall_score == pytest_approx_or_exact(0.15)
        assert any("difficulty mismatch" in issue for issue in ev.issues)


class TestTrivialPuzzle:
    def test_flagged_as_trivial(self):
        ev = evaluate_puzzle(
            _puzzle(TRIVIAL_SETUP_FEN, TRIVIAL_MOVES, PrimaryCategory="Hanging Piece"),
            engine=None,
        )
        assert ev.solution_depth == 2
        assert ev.solution_depth <= MIN_SOLUTION_DEPTH
        assert ev.is_tactical is True   # it is a capture
        assert ev.is_trivial is True    # undefended piece, <=3 ply
        assert any("trivial" in issue for issue in ev.issues)

    def test_composite_score_without_player_elo(self):
        # engine=None -> 0, clarity None -> 0, depth 2<3 -> 0, trivial -> +0 (not +0.15),
        # difficulty_fits defaults True when player_elo is None -> +0.10  =>  0.10
        ev = evaluate_puzzle(
            _puzzle(TRIVIAL_SETUP_FEN, TRIVIAL_MOVES, PrimaryCategory="Hanging Piece"),
            engine=None,
        )
        assert ev.overall_score == pytest_approx_or_exact(0.10)


class TestBatchAndSummary:
    def test_evaluate_batch_and_summarise_smoke_test(self):
        puzzles = [
            _puzzle(MATE_SETUP_FEN, MATE_MOVES, PuzzleId="p1", PrimaryCategory="Mating Pattern"),
            _puzzle(TRIVIAL_SETUP_FEN, TRIVIAL_MOVES, PuzzleId="p2", PrimaryCategory="Hanging Piece"),
            _puzzle(START_FEN, "e2e4", PuzzleId="p3"),  # deliberately malformed (1 move)
        ]
        evals = evaluate_batch(puzzles, engine=None, player_elo=1200)
        assert len(evals) == 3

        summary = summarise(evals)
        assert summary["total"] == 3
        assert 0.0 <= summary["avg_score"] <= 1.0
        assert summary["engine_run"] == 0  # engine was None for every puzzle


def pytest_approx_or_exact(value):
    """Small helper so exact-float assertions read clearly without importing pytest
    into every assert line; overall_score is already round()-ed to 3 dp by the evaluator."""
    return round(value, 3)
