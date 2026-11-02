"""Unit tests for the line-level tactic tagger — hand-built positions, no engine."""

import chess

from src.puzzles.generator import _classify_tactic
from src.puzzles.tactic_tagger import _endgame_label, move_motifs, tag_line, tag_move


def _mv(uci):
    return chess.Move.from_uci(uci)


def _line(*ucis):
    return [chess.Move.from_uci(u) for u in ucis]


class TestMate:
    def test_back_rank_mate_in_one(self):
        board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
        assert tag_move(board, _mv("a1a8")) == "Mating Pattern"

    def test_engine_mate_verdict_overrides_first_move_shape(self):
        board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
        assert tag_line(board, _line("g1f1"), mate=True) == "Mating Pattern"

    def test_mate_in_two_found_by_playing_out_the_line(self):
        # Rook ladder: 1.Rb7+ (only a check) Ke8 2.Ra8#. The old labeller saw
        # only the first move, a check, and never recognised the mate.
        board = chess.Board("8/4k3/R7/8/8/8/8/1R4K1 w - - 0 1")
        line = _line("b1b7", "e7e8", "a6a8")
        assert _classify_tactic(board, line[0]) != "Mating Pattern"    # old
        assert tag_line(board, line) == "Mating Pattern"               # new


class TestFork:
    def test_royal_fork_knight_on_c7(self):
        # Nc7+ attacks the e8 king and the a8 rook; nothing black can take c7.
        board = chess.Board("r3k3/8/8/1N6/8/8/8/4K3 w - - 0 1")
        assert tag_move(board, _mv("b5c7")) == "Fork"

    def test_fork_where_the_forking_piece_is_simply_lost(self):
        # Same fork, but the b8 bishop takes the undefended knight on c7.
        board = chess.Board("rb2k3/8/8/1N6/8/8/8/4K3 w - - 0 1")
        assert "Fork" not in move_motifs(board, _mv("b5c7"))

    def test_check_plus_defended_pawn_is_not_a_fork(self):
        # Qa4+ checks e8 and touches a5, but a5 is defended by b6. The old
        # rule counted the king as a high-value target and called this a fork.
        board = chess.Board("4k3/8/1p6/p7/8/8/8/3QK3 w - - 0 1")
        assert _classify_tactic(board, _mv("d1a4")) == "Fork"          # old: wrong
        assert "Fork" not in move_motifs(board, _mv("d1a4"))           # new: right


class TestPinSkewer:
    def test_absolute_pin(self):
        # Bb5 pins the c6 knight to the e8 king (d7 empty).
        board = chess.Board("4k3/8/2n5/8/8/8/8/4KB2 w - - 0 1")
        assert "Pin" in move_motifs(board, _mv("f1b5"))

    def test_pinned_pawn_is_not_reported(self):
        board = chess.Board("4k3/8/2p5/8/8/8/8/4KB2 w - - 0 1")
        assert "Pin" not in move_motifs(board, _mv("f1b5"))

    def test_skewer_king_in_front_of_queen(self):
        # Rh4+ on the 4th rank: king on d4, queen behind it on a4.
        board = chess.Board("8/8/8/8/q2k4/8/8/4K2R w - - 0 1")
        assert "Skewer" in move_motifs(board, _mv("h1h4"))

    def test_line_with_nothing_behind_is_not_a_skewer(self):
        board = chess.Board("8/8/8/8/k7/8/8/4K2R w - - 0 1")
        assert "Skewer" not in move_motifs(board, _mv("h1h4"))


class TestOtherMotifs:
    def test_promotion_is_detected(self):
        board = chess.Board("8/1P3k2/8/8/8/8/5K2/8 w - - 0 1")
        assert tag_line(board, _line("b7b8q")) == "Promotion"

    def test_en_passant(self):
        board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
        assert "En Passant" in move_motifs(board, _mv("e5d6"))

    def test_hanging_piece_capture(self):
        # Rxe5 wins an undefended knight.
        board = chess.Board("4k3/8/8/4n3/8/8/8/4RK2 w - - 0 1")
        assert "Hanging Piece" in move_motifs(board, _mv("e1e5"))

    def test_capturing_an_undefended_pawn_is_not_hanging_piece(self):
        board = chess.Board("4k3/8/8/4p3/8/8/8/4RK2 w - - 0 1")
        assert "Hanging Piece" not in move_motifs(board, _mv("e1e5"))


class TestEndgameSignatures:
    """Endgame type comes from the material left, not from which piece moved."""

    def test_signatures(self):
        cases = {
            "8/5k2/8/3r4/8/8/2R2PK1/8 w - - 0 1": "Rook Endgame",
            "8/8/4k3/5p2/5P2/5K2/8/8 w - - 0 1": "Pawn Endgame",
            "8/5k2/8/3q4/8/8/2Q2PK1/8 w - - 0 1": "Queen Endgame",
            "8/5k2/8/3b4/8/8/2B2PK1/8 w - - 0 1": "Bishop Endgame",
            "8/5k2/8/3n4/8/8/2N2PK1/8 w - - 0 1": "Knight Endgame",
            "8/5k2/8/3n4/8/8/2R2PK1/8 w - - 0 1": "Endgame",       # mixed, few pieces
        }
        for fen, expected in cases.items():
            assert _endgame_label(chess.Board(fen)) == expected, fen

    def test_opening_position_has_no_endgame_label(self):
        assert _endgame_label(chess.Board()) is None

    def test_motifs_outrank_the_endgame_type(self):
        # A quiet king move in a pawn ending is a "Quiet Move" puzzle, exactly
        # as MOTIF_PRIORITY ranks it for the puzzle pool.
        board = chess.Board("8/8/4k3/5p2/5P2/5K2/8/8 w - - 0 1")
        assert tag_line(board, _line("f3e3"), mate=False) == "Quiet Move"


class TestRobustness:
    def test_empty_line_is_general(self):
        assert tag_line(chess.Board(), []) == "General"

    def test_illegal_move_stops_the_line_without_raising(self):
        tag_line(chess.Board(), _line("e2e4", "e7e5", "a1a8"))

    def test_board_is_not_modified(self):
        board = chess.Board()
        fen = board.fen()
        tag_line(board, _line("e2e4", "e7e5", "g1f3"))
        assert board.fen() == fen
