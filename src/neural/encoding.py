"""
Line encoding for PuzzleNet.

An example is a position with the solver to move plus the principal line that
follows it: solver move m1, reply r1, solver move m2, reply r2, solver move m3,
and so on. In a Lichess puzzle the line is the solution; in game analysis it is
Stockfish's principal variation; in a mined puzzle it is the forced continuation
the miner built.

Everything is encoded from the solver's point of view. When Black is to move,
squares are mirrored (a1 <-> a8) and the colours are swapped, so "my" pieces always
fill the first six planes and "my" pawns always move up the board. This halves what
the network has to learn, and it makes the encoding colour-symmetric by
construction (tests/test_neural_encoding.py checks that a position and its
colour-mirrored twin give identical vectors).

The vector has two parts:

  bits  (N_BITS binary features)
        * piece-square planes (12 x 64) for three boards: before the key move (s0),
          after it (s1), and after the solver's second move (s3)
        * six move blocks: the opponent move that led to the position (r0), then
          m1 r1 m2 r2 m3. Each block holds from-square, to-square, moving piece,
          captured piece and flags (check, promotion, under-promotion, castling,
          en passant)
        * tactical relations after each solver move: what the moved piece attacks,
          whether it is en prise, discovered and double checks, newly created pins,
          pieces lined up behind an attacked piece, and so on
        * facts about the position and the line: hanging pieces, forcing-move
          uniqueness, castling rights, endgame material signatures, line length,
          and the mate verdict
  cont  (N_CONT continuous features, each roughly in [-2, 2])
        material counts and balance, mobility, king-zone pressure and escape
        squares, and how many replies the opponent had

The encoding contains no engine evaluation, so it can be computed for all 5.9M
Lichess puzzles in minutes. The one exception is the optional `mate` flag, which
lets game analysis pass Stockfish's mate verdict for a PV that was cut short
before the mate.
"""
from __future__ import annotations

from typing import Optional, Sequence

import chess
import numpy as np

ENCODING_VERSION = 1

_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5,
          chess.QUEEN: 9, chess.KING: 100}
_PIECES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
_SLIDERS = (chess.BISHOP, chess.ROOK, chess.QUEEN)

# ── Layout ────────────────────────────────────────────────────────────────────

BOARD_SLOTS = ("s0", "s1", "s3")
MOVE_SLOTS = ("r0", "m1", "r1", "m2", "r2", "m3")
SOLVER_SLOTS = ("m1", "m2", "m3")
REPLY_SLOTS = ("r1", "r2")

MOVE_FIELDS = (
    [f"from_{i}" for i in range(64)] + [f"to_{i}" for i in range(64)]
    + [f"piece_{p}" for p in "PNBRQK"] + [f"captures_{p}" for p in "PNBRQ"]
    + ["check", "promotion", "underpromotion", "castling", "en_passant"]
)
TACTIC_FIELDS = (
    [f"attacks_{p}" for p in "PNBRQK"]
    + ["double_attack", "attacked", "defended", "attacked_by_lower", "en_prise",
       "sacrifice", "discovered_check", "double_check", "discovered_attack",
       "creates_pin"]
    + [f"pinned_{p}" for p in "PNBRQ"]
    + ["skewer_ray", "xray_ray", "quiet", "retreat", "captures_defender",
       "king_move", "next_to_king", "mate_after"]
)
REPLY_FIELDS = ("king_move", "recapture", "interposes")
GLOBAL_FIELDS = (
    [f"their_hanging_{p}" for p in "PNBRQ"] + [f"my_hanging_{p}" for p in "PNBRQ"]
    + ["in_check", "m1_only_check", "m1_only_capture", "m1_best_capture",
       "castle_my_k", "castle_my_q", "castle_their_k", "castle_their_q", "ep_available"]
    + [f"solver_moves_{n}" for n in ("1", "2", "3", "4", "5plus")]
    + ["line_mate", "line_stalemate", "line_promotion", "line_underpromotion",
       "line_castling", "line_en_passant"]
    + ["eg_pawn", "eg_rook", "eg_queen", "eg_queen_rook", "eg_bishop", "eg_knight",
       "no_queens"]
    + ["their_king_back_rank", "their_king_boxed", "m1_enemy_half", "line_ends_on_reply"]
)
CONT_FIELDS = (
    [f"my_{p}" for p in "PNBRQ"] + [f"their_{p}" for p in "PNBRQ"]
    + ["balance_start", "balance_end", "material_gain", "non_pawn_material",
       "legal_moves", "captures", "checks"]
    + [f"{s}_king_pressure" for s in SOLVER_SLOTS] + [f"{s}_king_escapes" for s in SOLVER_SLOTS]
    + [f"{s}_opp_replies" for s in REPLY_SLOTS]
    + ["m1_distance", "line_plies"]
)


def _build_layout() -> tuple[dict[str, int], list[str]]:
    offsets, names = {}, []

    def block(key: str, fields: Sequence[str]) -> None:
        offsets[key] = len(names)
        names.extend(f"{key}.{f}" for f in fields)

    planes = [f"{side}{p}_{sq}" for side in ("my_", "their_") for p in "PNBRQK" for sq in range(64)]
    for s in BOARD_SLOTS:
        block(f"board_{s}", planes)
    for s in MOVE_SLOTS:
        block(f"move_{s}", MOVE_FIELDS)
    for s in SOLVER_SLOTS:
        block(f"tactic_{s}", TACTIC_FIELDS)
    for s in REPLY_SLOTS:
        block(f"reply_{s}", REPLY_FIELDS)
    block("global", GLOBAL_FIELDS)
    return offsets, names


OFFSETS, BIT_NAMES = _build_layout()
N_BITS = len(BIT_NAMES)
N_CONT = len(CONT_FIELDS)
N_FEATURES = N_BITS + N_CONT
N_PACKED = (N_BITS + 7) // 8

_MOVE_IDX = {f: i for i, f in enumerate(MOVE_FIELDS)}
_TACTIC_IDX = {f: i for i, f in enumerate(TACTIC_FIELDS)}
_REPLY_IDX = {f: i for i, f in enumerate(REPLY_FIELDS)}
_GLOBAL_IDX = {f: i for i, f in enumerate(GLOBAL_FIELDS)}
_CONT_IDX = {f: i for i, f in enumerate(CONT_FIELDS)}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _orient(sq: int, me: chess.Color) -> int:
    return sq if me == chess.WHITE else sq ^ 56


def _squares(mask: int):
    return chess.scan_forward(mask)


def _material(board: chess.Board, color: chess.Color) -> int:
    return sum(_VALUE[pt] * chess.popcount(board.pieces_mask(pt, color)) for pt in _PIECES[:5])


def _lowest_attacker_value(board: chess.Board, color: chess.Color, sq: int) -> Optional[int]:
    vals = [_VALUE[board.piece_type_at(a)] for a in _squares(board.attackers_mask(color, sq))]
    return min(vals) if vals else None


def _first_piece_behind(board: chess.Board, origin: int, target: int) -> Optional[int]:
    """The first occupied square beyond `target` on the ray origin -> target."""
    df = np.sign(chess.square_file(target) - chess.square_file(origin))
    dr = np.sign(chess.square_rank(target) - chess.square_rank(origin))
    f, r = chess.square_file(target) + df, chess.square_rank(target) + dr
    while 0 <= f < 8 and 0 <= r < 8:
        sq = chess.square(int(f), int(r))
        if board.piece_type_at(sq):
            return sq
        f, r = f + df, r + dr
    return None


def _pinned_mask(board: chess.Board, color: chess.Color) -> int:
    mask = 0
    for sq in _squares(board.occupied_co[color] & ~board.kings):
        if board.is_pinned(color, sq):
            mask |= chess.BB_SQUARES[sq]
    return mask


def _set_board(bits: np.ndarray, offset: int, board: chess.Board, me: chess.Color) -> None:
    for color in (me, not me):
        side = 0 if color == me else 6
        for pt in _PIECES:
            base = offset + (side + pt - 1) * 64
            for sq in _squares(board.pieces_mask(pt, color)):
                bits[base + _orient(sq, me)] = 1


def _set_move(bits: np.ndarray, offset: int, board: chess.Board, move: chess.Move,
              me: chess.Color) -> None:
    """Encode `move` played from `board` (the position before the move)."""
    bits[offset + _MOVE_IDX[f"from_{_orient(move.from_square, me)}"]] = 1
    bits[offset + _MOVE_IDX[f"to_{_orient(move.to_square, me)}"]] = 1
    mover = board.piece_type_at(move.from_square)
    if mover:
        bits[offset + _MOVE_IDX["piece_" + "PNBRQK"[mover - 1]]] = 1
    if board.is_en_passant(move):
        bits[offset + _MOVE_IDX["captures_P"]] = 1
        bits[offset + _MOVE_IDX["en_passant"]] = 1
    else:
        captured = board.piece_type_at(move.to_square)
        if captured and captured != chess.KING and board.color_at(move.to_square) != board.turn:
            bits[offset + _MOVE_IDX["captures_" + "PNBRQK"[captured - 1]]] = 1
    if board.gives_check(move):
        bits[offset + _MOVE_IDX["check"]] = 1
    if move.promotion:
        bits[offset + _MOVE_IDX["promotion"]] = 1
        if move.promotion != chess.QUEEN:
            bits[offset + _MOVE_IDX["underpromotion"]] = 1
    if board.is_castling(move):
        bits[offset + _MOVE_IDX["castling"]] = 1


def _captured_value(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        return 1
    pt = board.piece_type_at(move.to_square)
    return _VALUE[pt] if pt and pt != chess.KING else 0


def _set_tactic(bits: np.ndarray, cont: np.ndarray, offset: int, slot: str,
                before: chess.Board, move: chess.Move, after: chess.Board,
                me: chess.Color) -> None:
    """Tactical relations created by solver move `move` (before -> after)."""
    them = not me
    to = move.to_square
    moved = after.piece_type_at(to)
    if moved is None:                       # castling moves the king elsewhere
        moved = chess.KING
    t = lambda name: offset + _TACTIC_IDX[name]   # noqa: E731

    targets = 0
    for sq in _squares(after.attacks_mask(to) & after.occupied_co[them]):
        pt = after.piece_type_at(sq)
        bits[t("attacks_" + "PNBRQK"[pt - 1])] = 1
        if pt != chess.PAWN:
            targets += 1
    if targets >= 2:
        bits[t("double_attack")] = 1

    attacked = after.is_attacked_by(them, to)
    defended = after.is_attacked_by(me, to)
    lowest = _lowest_attacker_value(after, them, to) if attacked else None
    lower = lowest is not None and lowest < _VALUE[moved]
    en_prise = attacked and (not defended or lower) and moved != chess.KING
    captured = _captured_value(before, move)
    for name, on in (("attacked", attacked), ("defended", defended),
                     ("attacked_by_lower", lower), ("en_prise", en_prise),
                     ("sacrifice", en_prise and _VALUE[moved] > captured)):
        if on:
            bits[t(name)] = 1

    checkers = after.checkers_mask()
    if checkers:
        if checkers & ~chess.BB_SQUARES[to]:
            bits[t("discovered_check")] = 1
        if chess.popcount(checkers) >= 2:
            bits[t("double_check")] = 1

    # Another slider of mine newly attacks a piece (not a pawn) once this one moved.
    for pt in _SLIDERS:
        for sq in _squares(after.pieces_mask(pt, me) & ~chess.BB_SQUARES[to]):
            gained = after.attacks_mask(sq) & ~before.attacks_mask(sq)
            if gained & after.occupied_co[them] & ~after.pawns:
                bits[t("discovered_attack")] = 1
                break

    pinned_after = _pinned_mask(after, them)
    if pinned_after & ~_pinned_mask(before, them):
        bits[t("creates_pin")] = 1
    for sq in _squares(pinned_after):
        bits[t("pinned_" + "PNBRQK"[after.piece_type_at(sq) - 1])] = 1

    if moved in _SLIDERS:
        for sq in _squares(after.attacks_mask(to) & after.occupied_co[them]):
            front = after.piece_type_at(sq)
            if front not in (chess.KING, chess.QUEEN, chess.ROOK):
                continue
            behind = _first_piece_behind(after, to, sq)
            if behind is None or after.color_at(behind) != them:
                continue
            if front == chess.KING or _VALUE[front] > _VALUE[after.piece_type_at(behind)]:
                bits[t("skewer_ray")] = 1
            else:
                bits[t("xray_ray")] = 1

    if not before.is_capture(move) and not checkers and not move.promotion:
        bits[t("quiet")] = 1
    if chess.square_rank(_orient(to, me)) < chess.square_rank(_orient(move.from_square, me)):
        bits[t("retreat")] = 1
    if before.is_capture(move) and not before.is_en_passant(move):
        guarded = before.attacks_mask(to) & before.occupied_co[them]
        if any(before.is_attacked_by(me, sq) for sq in _squares(guarded)):
            bits[t("captures_defender")] = 1
    if before.piece_type_at(move.from_square) == chess.KING:
        bits[t("king_move")] = 1

    ksq = after.king(them)
    if ksq is not None:
        zone = chess.BB_KING_ATTACKS[ksq]
        if zone & chess.BB_SQUARES[to]:
            bits[t("next_to_king")] = 1
        pressure = sum(1 for sq in _squares(zone) if after.is_attacked_by(me, sq))
        escapes = sum(1 for sq in _squares(zone & ~after.occupied_co[them])
                      if not after.is_attacked_by(me, sq))
        cont[_CONT_IDX[f"{slot}_king_pressure"]] = pressure / 8.0
        cont[_CONT_IDX[f"{slot}_king_escapes"]] = escapes / 8.0
    if checkers and after.is_checkmate():
        bits[t("mate_after")] = 1


def _endgame_signature(board: chess.Board) -> list[str]:
    queens, rooks = board.queens, board.rooks
    bishops, knights = board.bishops, board.knights
    minors = bishops | knights
    out = []
    if not (queens | rooks | minors):
        out.append("eg_pawn")
    if rooks and not (queens | minors):
        out.append("eg_rook")
    if queens and not (rooks | minors):
        out.append("eg_queen")
    if queens and rooks and not minors:
        out.append("eg_queen_rook")
    if bishops and not (queens | rooks | knights):
        out.append("eg_bishop")
    if knights and not (queens | rooks | bishops):
        out.append("eg_knight")
    if not queens:
        out.append("no_queens")
    return out


# ── Public API ────────────────────────────────────────────────────────────────

def encode_line(board: chess.Board, line: Sequence[chess.Move], *,
                mate: Optional[bool] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode `board` (solver to move) and the principal `line` that follows it.

    The opponent move that led to `board` (r0) is read from `board.move_stack`
    when there is one. `mate` overrides the mate verdict; if it is None, the line
    is played out and the verdict is whether it ends in checkmate.

    Returns (bits, cont): a uint8 0/1 vector of length N_BITS and a float32
    vector of length N_CONT. Moves after the first illegal one are ignored.
    """
    bits = np.zeros(N_BITS, dtype=np.uint8)
    cont = np.zeros(N_CONT, dtype=np.float32)
    me, them = board.turn, not board.turn
    g = lambda name: OFFSETS["global"] + _GLOBAL_IDX[name]   # noqa: E731

    if board.move_stack:
        prev = board.copy(stack=1)
        r0 = prev.pop()
        _set_move(bits, OFFSETS["move_r0"], prev, r0, me)

    # Position facts at s0.
    _set_board(bits, OFFSETS["board_s0"], board, me)
    for color, prefix in ((them, "their_hanging_"), (me, "my_hanging_")):
        for sq in _squares(board.occupied_co[color] & ~board.kings):
            if board.is_attacked_by(not color, sq) and not board.is_attacked_by(color, sq):
                bits[g(prefix + "PNBRQK"[board.piece_type_at(sq) - 1])] = 1
    if board.is_check():
        bits[g("in_check")] = 1
    for name, on in (("castle_my_k", board.has_kingside_castling_rights(me)),
                     ("castle_my_q", board.has_queenside_castling_rights(me)),
                     ("castle_their_k", board.has_kingside_castling_rights(them)),
                     ("castle_their_q", board.has_queenside_castling_rights(them)),
                     ("ep_available", board.has_legal_en_passant())):
        if on:
            bits[g(name)] = 1
    for name in _endgame_signature(board):
        bits[g(name)] = 1
    ksq = board.king(them)
    if ksq is not None and chess.square_rank(_orient(ksq, me)) == 7:
        bits[g("their_king_back_rank")] = 1
        front = [sq for sq in _squares(chess.BB_KING_ATTACKS[ksq])
                 if chess.square_rank(_orient(sq, me)) == 6]
        if front and all(board.color_at(sq) == them or board.is_attacked_by(me, sq) for sq in front):
            bits[g("their_king_boxed")] = 1

    for i, p in enumerate("PNBRQ"):
        pt = _PIECES[i]
        scale = 8.0 if pt == chess.PAWN else (1.0 if pt == chess.QUEEN else 2.0)
        cont[_CONT_IDX[f"my_{p}"]] = chess.popcount(board.pieces_mask(pt, me)) / scale
        cont[_CONT_IDX[f"their_{p}"]] = chess.popcount(board.pieces_mask(pt, them)) / scale
    start_balance = _material(board, me) - _material(board, them)
    cont[_CONT_IDX["balance_start"]] = start_balance / 10.0
    npm = sum(_VALUE[pt] * chess.popcount(board.pieces_mask(pt, c))
              for pt in _PIECES[1:5] for c in (me, them))
    cont[_CONT_IDX["non_pawn_material"]] = npm / 62.0

    legal = list(board.legal_moves)
    checks = [m for m in legal if board.gives_check(m)]
    captures = [m for m in legal if board.is_capture(m)]
    cont[_CONT_IDX["legal_moves"]] = len(legal) / 50.0
    cont[_CONT_IDX["captures"]] = len(captures) / 10.0
    cont[_CONT_IDX["checks"]] = len(checks) / 10.0

    # Walk the line.
    b = board.copy(stack=False)
    played: list[chess.Move] = []
    solver_moves = 0
    for i, move in enumerate(line):
        if move not in b.legal_moves:
            break
        solver_turn = i % 2 == 0
        slot = MOVE_SLOTS[i + 1] if i + 1 < len(MOVE_SLOTS) else None
        if slot:
            _set_move(bits, OFFSETS[f"move_{slot}"], b, move, me)
        if move.promotion:
            bits[g("line_promotion")] = 1
            if move.promotion != chess.QUEEN:
                bits[g("line_underpromotion")] = 1
        if b.is_castling(move):
            bits[g("line_castling")] = 1
        if b.is_en_passant(move):
            bits[g("line_en_passant")] = 1

        if i == 0:
            if move in checks and len(checks) == 1:
                bits[g("m1_only_check")] = 1
            if move in captures:
                if len(captures) == 1:
                    bits[g("m1_only_capture")] = 1
                if _captured_value(b, move) >= max(_captured_value(b, c) for c in captures):
                    bits[g("m1_best_capture")] = 1
            if chess.square_rank(_orient(move.to_square, me)) >= 4:
                bits[g("m1_enemy_half")] = 1
            cont[_CONT_IDX["m1_distance"]] = chess.square_distance(move.from_square, move.to_square) / 7.0

        if not solver_turn and slot in REPLY_SLOTS:
            ro = OFFSETS[f"reply_{slot}"]
            cont[_CONT_IDX[f"{slot}_opp_replies"]] = b.legal_moves.count() / 40.0
            if b.piece_type_at(move.from_square) == chess.KING:
                bits[ro + _REPLY_IDX["king_move"]] = 1
            if played and move.to_square == played[-1].to_square:
                bits[ro + _REPLY_IDX["recapture"]] = 1
            if b.is_check() and b.piece_type_at(move.from_square) != chess.KING \
                    and not (b.checkers_mask() & chess.BB_SQUARES[move.to_square]):
                bits[ro + _REPLY_IDX["interposes"]] = 1

        before = b.copy(stack=False) if solver_turn else None
        b.push(move)
        played.append(move)
        if solver_turn:
            solver_moves += 1
            if slot in SOLVER_SLOTS:
                _set_tactic(bits, cont, OFFSETS[f"tactic_{slot}"], slot, before, move, b, me)
            if solver_moves == 1:
                _set_board(bits, OFFSETS["board_s1"], b, me)
            elif solver_moves == 2:
                _set_board(bits, OFFSETS["board_s3"], b, me)

    n = min(solver_moves, 5)
    if n:
        bits[g("solver_moves_" + ("5plus" if n == 5 else str(n)))] = 1
    if played and len(played) % 2 == 0:
        bits[g("line_ends_on_reply")] = 1
    is_mate = b.is_checkmate() if mate is None else bool(mate)
    if is_mate:
        bits[g("line_mate")] = 1
    if b.is_stalemate():
        bits[g("line_stalemate")] = 1
    end_balance = _material(b, me) - _material(b, them)
    cont[_CONT_IDX["balance_end"]] = end_balance / 10.0
    cont[_CONT_IDX["material_gain"]] = (end_balance - start_balance) / 10.0
    cont[_CONT_IDX["line_plies"]] = len(played) / 16.0
    return bits, cont


def encode_puzzle(fen: str, moves: str | Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Encode a puzzle in Lichess convention: FEN before the opponent's move,
    Moves = [opponent move, solution...]."""
    ms = moves.split() if isinstance(moves, str) else list(moves)
    board = chess.Board(fen)
    board.push_uci(ms[0])
    return encode_line(board, [chess.Move.from_uci(u) for u in ms[1:]])


def to_dense(bits: np.ndarray, cont: np.ndarray) -> np.ndarray:
    """Concatenate the two parts into one float32 row (or rows) for the network."""
    return np.concatenate([bits.astype(np.float32), cont.astype(np.float32)], axis=-1)
