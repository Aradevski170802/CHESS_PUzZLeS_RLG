"""Lightweight, engine-free tests for src/classifier/stockfish_analyzer.py.

analyze_game() / analyze_games_parallel() require a live Stockfish process
and are intentionally out of scope for offline unit tests here, the same
convention used for the rest of this suite (see test_app.py's and
test_generator.py's docstrings). This file only covers the MoveError
dataclass contract that player_profiler.py and ml_weakness_model.py depend
on -- in particular the new `category` field populated by classifying the
engine's missed best move via src.puzzles.generator._classify_tactic.
"""

from src.classifier.stockfish_analyzer import MoveError

DUMMY_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class TestMoveErrorCategory:
    def test_category_defaults_to_general(self):
        error = MoveError(
            move_number=1, fen_before=DUMMY_FEN, move_uci="e2e4",
            cp_loss=250, severity="blunder", phase="middlegame",
        )
        assert error.category == "General"

    def test_category_can_be_set_explicitly(self):
        error = MoveError(
            move_number=1, fen_before=DUMMY_FEN, move_uci="e2e4",
            cp_loss=250, severity="blunder", phase="middlegame", category="Fork",
        )
        assert error.category == "Fork"


# ── Win-probability grading, time classes, colour detection ─────────────────

import chess.pgn
import io
import pytest

from src.classifier.stockfish_analyzer import (
    _end_time,
    _player_color,
    severity_of,
    time_class_of,
    win_percent,
)


class TestWinPercent:
    def test_equal_position_is_fifty_percent(self):
        assert win_percent(0) == pytest.approx(50.0)

    def test_is_monotonic_and_symmetric(self):
        assert win_percent(100) > win_percent(0) > win_percent(-100)
        assert win_percent(250) + win_percent(-250) == pytest.approx(100.0)

    def test_mate_scores_saturate(self):
        assert win_percent(9000) > 99.99
        assert win_percent(-9000) < 0.01

    def test_same_centipawn_loss_matters_less_in_a_decided_position(self):
        # The reason severity is graded on win % rather than raw centipawns.
        # The same 300 cp loss is a blunder in a close game but only an
        # inaccuracy once the game is already decided.
        close_game = win_percent(50) - win_percent(-250)
        decided = win_percent(900) - win_percent(600)
        assert severity_of(close_game) == "blunder"
        assert severity_of(decided) == "inaccuracy"


class TestSeverity:
    @pytest.mark.parametrize("loss,expected", [
        (2.0, None), (5.0, "inaccuracy"), (9.9, "inaccuracy"),
        (10.0, "mistake"), (15.0, "blunder"), (60.0, "blunder"),
    ])
    def test_thresholds(self, loss, expected):
        assert severity_of(loss) == expected


class TestTimeClass:
    @pytest.mark.parametrize("tc,expected", [
        ("60", "bullet"), ("120+1", "bullet"), ("180", "blitz"),
        ("300+5", "blitz"), ("600", "rapid"), ("900+10", "rapid"),
        ("1/86400", "daily"), ("", "unknown"), ("-", "unknown"), ("abc", "unknown"),
    ])
    def test_classification(self, tc, expected):
        assert time_class_of(tc) == expected


def _headers(white, black, **extra):
    tags = "".join(f'[{k} "{v}"]\n' for k, v in {"White": white, "Black": black, **extra}.items())
    return chess.pgn.read_game(io.StringIO(tags + "\n1. e4 *")).headers


class TestPlayerColour:
    def test_exact_match_beats_substring(self):
        # Old code: "bob" in "bobby_99" -> wrongly white.
        assert _player_color(_headers("bobby_99", "bob"), "bob") == "black"

    def test_case_insensitive(self):
        assert _player_color(_headers("Magnus", "x"), "magnus") == "white"


class TestEndTime:
    def test_parses_chess_com_end_headers(self):
        h = _headers("a", "b", EndDate="2026.09.01", EndTime="12:30:00")
        assert _end_time(h) == 1788265800

    def test_missing_date_is_none(self):
        assert _end_time(_headers("a", "b", Date="????.??.??")) is None
