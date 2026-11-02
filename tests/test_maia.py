"""Unit tests for src/neural/maia.py.

No lc0 process is started: the rating-level mapping and the parsing of lc0's
VerboseMoveStats output are pure functions, tested on lines copied from a real run.
"""

import chess
from pytest import approx

from src.neural import maia

# Real lc0 v0.32.1 output for the starting position with the Maia-1500 network.
START_LINES = [
    "b1a3  (34  ) N:       0 (+ 0) (P:  0.00%) (WL:  -.-----) (D: -.---) (M:  -.-) "
    "(Q:  0.04017) (U: 0.00008) (S:  0.04024) (V:  -.----) ",
    "e2e4  (322 ) N:       0 (+ 0) (P: 65.14%) (WL:  -.-----) (D: -.---) (M:  -.-) "
    "(Q:  0.04017) (U: 1.13670) (S:  1.17687) (V:  -.----) ",
    "node  (  20) N:       1 (+ 0) (P:  0.00%) (WL:  0.03821) (D: 0.038) (M:  0.0) "
    "(Q:  0.03821) (V:  0.0402) ",
]


def _line(uci, pct):
    return f"{uci}  (100 ) N:       0 (+ 0) (P: {pct:5.2f}%) (WL:  -.-----) (Q:  0.0) "


class TestNearestLevel:
    def test_levels_cover_1100_to_1900(self):
        assert maia.MAIA_LEVELS == (1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900)

    def test_ratings_outside_the_range_get_the_end_levels(self):
        assert maia.nearest_level(600) == 1100
        assert maia.nearest_level(2600) == 1900

    def test_nearest_level_and_ties_go_down(self):
        assert maia.nearest_level(1449) == 1400
        assert maia.nearest_level(1451) == 1500
        assert maia.nearest_level(1450) == 1400


class TestParseMoveStats:
    def test_reads_probabilities_and_skips_the_node_line(self):
        probs = maia.parse_move_stats(START_LINES, chess.Board())
        assert probs == approx({"b1a3": 0.0, "e2e4": 0.6514})

    def test_ignores_unrelated_info_strings(self):
        assert maia.parse_move_stats(["Loading weights", ""], chess.Board()) == {}

    def test_king_takes_rook_castling_becomes_standard_uci(self):
        board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        probs = maia.parse_move_stats([_line("e1h1", 12.5)], board)
        assert probs == approx({"e1g1": 0.125})

    def test_a_rook_move_e1h1_is_not_castling(self):
        board = chess.Board("4k3/8/8/8/8/8/8/K3R3 w - - 0 1")    # rook on e1, king on a1
        assert maia.parse_move_stats([_line("e1h1", 30.0)], board) == approx({"e1h1": 0.3})


class TestFiles:
    def test_missing_weights_are_reported_as_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MAIA_DIR", str(tmp_path))
        assert maia.weights_path(1500) is None
        assert maia.available() is False
