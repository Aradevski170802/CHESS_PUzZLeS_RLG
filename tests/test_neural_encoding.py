"""PuzzleNet line encoding: layout, colour symmetry, and individual tactical features."""
import chess
import numpy as np
import pytest

from src.neural import encoding as E


def _line(*ucis):
    return [chess.Move.from_uci(u) for u in ucis]


def _bit(bits, name):
    return int(bits[E.BIT_NAMES.index(name)])


def _cont(cont, name):
    return float(cont[E.CONT_FIELDS.index(name)])


def _mirror(m: chess.Move) -> chess.Move:
    return chess.Move(chess.square_mirror(m.from_square), chess.square_mirror(m.to_square), m.promotion)


# Real Lichess puzzles (FEN before the opponent's move; Moves = opponent move + solution),
# with both colours to move.
PUZZLES = [
    ("r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24", "f2g3 e6e7 b2b1 b3c1 b1c1 h6c1"),
    ("5rk1/1p3ppp/pq3b2/8/8/1P1Q1N2/P4PPP/3R2K1 w - - 2 27", "d3d6 f8d8 d6d8 f6d8"),
    ("8/4R3/1p2P3/p4r2/P6p/1P3Pk1/4K3/8 w - - 1 64", "e7f7 f5e5 e2f1 e5e6"),
    ("r1b1k1nr/pp3ppp/2n5/q1bQ4/4N3/6P1/PP2PP1P/R1B1KBNR w KQkq - 3 9",
     "c1d2 c5f2 e4f2 a5d5"),
]


def test_layout_is_consistent():
    assert len(E.BIT_NAMES) == E.N_BITS
    assert len(set(E.BIT_NAMES)) == E.N_BITS
    assert E.N_FEATURES == E.N_BITS + E.N_CONT
    assert E.N_PACKED == (E.N_BITS + 7) // 8
    assert E.OFFSETS["board_s1"] - E.OFFSETS["board_s0"] == 768


@pytest.mark.parametrize("fen,moves", PUZZLES)
def test_encoding_is_identical_for_the_colour_mirrored_position(fen, moves):
    ms = _line(*moves.split())
    setup = chess.Board(fen)
    a = setup.copy()
    a.push(ms[0])
    b = setup.mirror()
    b.push(_mirror(ms[0]))
    bits_a, cont_a = E.encode_line(a, ms[1:])
    bits_b, cont_b = E.encode_line(b, [_mirror(m) for m in ms[1:]])
    assert np.array_equal(bits_a, bits_b)
    assert np.allclose(cont_a, cont_b)


@pytest.mark.parametrize("fen,moves", PUZZLES)
def test_shapes_and_values(fen, moves):
    bits, cont = E.encode_puzzle(fen, moves)
    assert bits.shape == (E.N_BITS,) and bits.dtype == np.uint8
    assert cont.shape == (E.N_CONT,) and cont.dtype == np.float32
    assert set(np.unique(bits)) <= {0, 1}
    assert np.all(np.abs(cont) < 5)


def test_solver_pieces_fill_the_first_planes_for_black_too():
    board = chess.Board("4k3/8/8/8/8/8/8/4K2R b K - 0 1")    # Black to move
    bits, _ = E.encode_line(board, [])
    # Black's king on e8 becomes "my king" on the mirrored square e1.
    assert _bit(bits, "board_s0.my_K_4") == 1
    assert _bit(bits, "board_s0.their_R_63") == 1           # h1 -> h8
    assert _bit(bits, "global.castle_their_k") == 1


def test_previous_move_is_read_from_the_move_stack():
    board = chess.Board()
    board.push_uci("e2e4")
    bits, _ = E.encode_line(board, [])
    # Black to move: White's e2-e4 is encoded mirrored, e7 -> e5 from Black's side.
    assert _bit(bits, f"move_r0.from_{chess.E7}") == 1
    assert _bit(bits, f"move_r0.to_{chess.E5}") == 1
    assert _bit(bits, "move_r0.piece_P") == 1
    no_stack, _ = E.encode_line(chess.Board(board.fen()), [])
    start, end = E.OFFSETS["move_r0"], E.OFFSETS["move_m1"]
    assert not no_stack[start:end].any()


def test_knight_fork_sets_double_attack():
    board = chess.Board("r3k3/8/8/1N6/8/8/8/4K3 w - - 0 1")
    bits, _ = E.encode_line(board, _line("b5c7"))
    assert _bit(bits, "tactic_m1.double_attack") == 1
    assert _bit(bits, "tactic_m1.attacks_K") == 1
    assert _bit(bits, "tactic_m1.attacks_R") == 1
    assert _bit(bits, "move_m1.check") == 1


def test_mate_in_one_is_detected_by_play_out_and_can_be_overridden():
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    bits, _ = E.encode_line(board, _line("a1a8"))
    assert _bit(bits, "global.line_mate") == 1
    assert _bit(bits, "tactic_m1.mate_after") == 1
    quiet, _ = E.encode_line(board, _line("g1f1"))
    assert _bit(quiet, "global.line_mate") == 0
    engine, _ = E.encode_line(board, _line("g1f1"), mate=True)
    assert _bit(engine, "global.line_mate") == 1


def test_pin_is_created():
    # Bb5 pins the c6 knight to the e8 king.
    board = chess.Board("4k3/8/2n5/8/8/8/8/2B1K3 w - - 0 1")
    bits, _ = E.encode_line(board, _line("c1a3"))
    assert _bit(bits, "tactic_m1.creates_pin") == 0
    board = chess.Board("4k3/8/2n5/8/8/8/8/3BK3 w - - 0 1")
    bits, _ = E.encode_line(board, _line("d1a4"))
    assert _bit(bits, "tactic_m1.creates_pin") == 1
    assert _bit(bits, "tactic_m1.pinned_N") == 1


def test_skewer_ray():
    # Rook check along the rank; the queen stands behind the king.
    board = chess.Board("8/8/8/q3k3/8/8/8/6KR w - - 0 1")
    bits, _ = E.encode_line(board, _line("h1h5"))
    assert _bit(bits, "tactic_m1.skewer_ray") == 1


def test_discovered_check():
    board = chess.Board("4k3/8/8/8/4N3/8/8/4R1K1 w - - 0 1")
    bits, _ = E.encode_line(board, _line("e4f6"))
    assert _bit(bits, "tactic_m1.discovered_check") == 1
    assert _bit(bits, "tactic_m1.double_check") == 1


def test_promotion_and_underpromotion_flags():
    board = chess.Board("8/P7/8/8/8/8/k7/4K3 w - - 0 1")
    bits, _ = E.encode_line(board, _line("a7a8n"))
    assert _bit(bits, "move_m1.promotion") == 1
    assert _bit(bits, "move_m1.underpromotion") == 1
    assert _bit(bits, "global.line_underpromotion") == 1


def test_en_passant_capture_is_encoded():
    board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    bits, _ = E.encode_line(board, _line("e5d6"))
    assert _bit(bits, "move_m1.en_passant") == 1
    assert _bit(bits, "move_m1.captures_P") == 1
    assert _bit(bits, "global.line_en_passant") == 1


def test_sacrifice_and_material_gain():
    # Queen takes a defended pawn: en prise, worth more than what it took.
    board = chess.Board("4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1")
    bits, cont = E.encode_line(board, _line("d1d5", "c6d5"))
    assert _bit(bits, "tactic_m1.en_prise") == 1
    assert _bit(bits, "tactic_m1.sacrifice") == 1
    assert _bit(bits, "reply_r1.recapture") == 1
    assert _cont(cont, "material_gain") == pytest.approx(-0.8)   # -9 + 1 pawns / 10


def test_endgame_signature():
    bits, _ = E.encode_line(chess.Board("8/5k2/8/8/8/8/R4K2/8 w - - 0 1"), [])
    assert _bit(bits, "global.eg_rook") == 1 and _bit(bits, "global.eg_pawn") == 0
    bits, _ = E.encode_line(chess.Board("8/5k2/5p2/8/8/5P2/5K2/8 w - - 0 1"), [])
    assert _bit(bits, "global.eg_pawn") == 1


def test_illegal_moves_truncate_the_line():
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    full, _ = E.encode_line(board, _line("a1a8"))
    junk, _ = E.encode_line(board, _line("a1a8", "e2e4", "a8b8"))
    assert np.array_equal(full, junk)


def test_line_length_one_hot():
    board = chess.Board("8/4k3/R7/8/8/8/8/1R4K1 w - - 0 1")
    bits, cont = E.encode_line(board, _line("b1b7", "e7e8", "a6a8"))
    assert _bit(bits, "global.solver_moves_2") == 1
    assert _bit(bits, "global.line_mate") == 1
    assert sum(_bit(bits, f"global.solver_moves_{n}") for n in ("1", "2", "3", "4", "5plus")) == 1
    assert _cont(cont, "line_plies") == pytest.approx(3 / 16)
