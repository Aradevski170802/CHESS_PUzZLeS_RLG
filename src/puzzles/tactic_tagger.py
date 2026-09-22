"""
Line-level tactic tagger — the replacement for generator._classify_tactic.

Why a replacement
──────────────────
scripts/research/validate_labeller.py measured _classify_tactic against the
themes of ~78,000 Lichess puzzles: 31 % strict agreement, Cohen's κ = 0.23,
macro recall 14 %, and 8 of the 23 categories could never be emitted at all.
The failures were structural, not tuning:

  * It looks at ONE move. A mate in 3 starts with a check, so it was labelled
    "Fork" or "King Safety"; the mate only appears on the last move.
  * Its fork rule treated the king as a high-value target, so almost any
    check that also touched a piece became a "Fork" (precision 25 %).
  * Its "Skewer" test asked whether the KING attacked the moved piece's
    square — not a skewer — and "X-Ray" fired on any slider capture
    (precision 0 %).

This tagger fixes each of those:

  * It reads the whole principal variation (the engine's PV in game analysis,
    the solution line for a puzzle), collects every motif the player's moves
    create, and resolves them with the same specificity order the puzzle pool
    uses (MOTIF_PRIORITY): mate > fork > pin > skewer > discovered attack >
    deflection > hanging piece > sacrifice > promotion > en passant > quiet
    move > king safety > endgame type.
  * A line the engine scores as a forced mate is a Mating Pattern, full stop.
  * Fork: the moved piece attacks ≥ 2 targets that are each the king, worth
    more than the attacker, or undefended — and the forking piece is not
    simply lost to a cheaper attacker.
  * Pin / skewer from ray geometry: along a line from the slider, the first
    opponent piece P1 and the next piece P2. P2 worth more than P1 (or the
    king) = pin; P1 the king or worth more than P2 = skewer.
  * Discovered attack: a different slider of ours newly attacks a valuable or
    undefended piece (or the king) once the moved piece gets out of the way.
  * Endgame types from the material signature (only rooks and pawns left =
    Rook Endgame, …), not from which piece happened to move.

Still not detected (and never guessed): Attraction, Interference, Clearance,
Zugzwang, X-Ray Attack. Those need deeper line semantics; emitting nothing is
better than the 0 %-precision guess the old X-Ray rule made.
"""
from __future__ import annotations

from typing import Iterable, Optional

import chess

from src.data.pgn_parser import get_game_phase

_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5,
          chess.QUEEN: 9, chess.KING: 100}
_SLIDERS = (chess.BISHOP, chess.ROOK, chess.QUEEN)
_DIRS = {
    chess.ROOK: [(0, 1), (1, 0), (0, -1), (-1, 0)],
    chess.BISHOP: [(1, 1), (1, -1), (-1, 1), (-1, -1)],
}
_DIRS[chess.QUEEN] = _DIRS[chess.ROOK] + _DIRS[chess.BISHOP]

# Resolution order — mirrors src.data.puzzle_loader.MOTIF_PRIORITY.
PRIORITY = [
    "Mating Pattern", "Fork", "Pin", "Skewer", "Discovered Attack", "X-Ray Attack",
    "Deflection", "Attraction", "Interference", "Clearance", "Hanging Piece",
    "Sacrifice", "Promotion", "En Passant", "Zugzwang", "Quiet Move", "King Safety",
    "Rook Endgame", "Pawn Endgame", "Queen Endgame", "Bishop Endgame",
    "Knight Endgame", "Endgame",
]
_RANK = {c: i for i, c in enumerate(PRIORITY)}

# Which motifs count at each player move of the line. A combination lives in
# the first one or two player moves; incidental pins and pawn grabs deep in
# the continuation were the main source of false positives on puzzles
# Lichess tags as plain "advantage" or "endgame" (validate_labeller.py).
_ALLOWED_BY_DEPTH = [
    None,                                              # key move: everything
    {"Fork", "Skewer", "Discovered Attack", "Sacrifice", "Promotion",
     "En Passant", "Mating Pattern"},                  # second player move
    {"Promotion", "Mating Pattern"},                   # later moves
]


def _v(piece: Optional[chess.Piece]) -> int:
    return _VALUE[piece.piece_type] if piece else 0


def _winnable(board: chess.Board, sq: int, attacker_value: int, owner: chess.Color) -> bool:
    """A target worth attacking: the king, worth more than the attacker, or undefended."""
    p = board.piece_at(sq)
    if p is None or p.color != owner:
        return False
    return (p.piece_type == chess.KING or _VALUE[p.piece_type] > attacker_value
            or not board.is_attacked_by(owner, sq))


_NONE = 10 ** 6


def _cheapest_capturer(board: chess.Board, color: chess.Color, sq: int) -> int:
    """
    Value of the cheapest `color` piece that can take on `sq`. The king only
    counts when `sq` is undefended (it cannot capture into check), and then
    counts as value 0 — Bxf7+ Kxf7 loses the bishop for nothing.
    """
    best = _NONE
    for a in board.attackers(color, sq):
        p = board.piece_at(a)
        if p.piece_type == chess.KING:
            if board.is_attacked_by(not color, sq):
                continue
            best = min(best, 0)
        else:
            best = min(best, _VALUE[p.piece_type])
    return best


def _is_safe(board: chess.Board, sq: int, owner: chess.Color) -> bool:
    """The piece on `sq` cannot simply be won: nothing takes it, or it is
    defended and every capturer is worth at least as much."""
    c = _cheapest_capturer(board, not owner, sq)
    if c == _NONE:
        return True
    return board.is_attacked_by(owner, sq) and c >= _v(board.piece_at(sq))


def _ray_pieces(board: chess.Board, sq: int, df: int, dr: int) -> list[int]:
    """Occupied squares met walking from `sq` in direction (df, dr)."""
    out = []
    f, r = chess.square_file(sq) + df, chess.square_rank(sq) + dr
    while 0 <= f < 8 and 0 <= r < 8:
        s = chess.square(f, r)
        if board.piece_at(s):
            out.append(s)
            if len(out) == 2:
                break
        f, r = f + df, r + dr
    return out


def _pin_or_skewer(board: chess.Board, sq: int) -> set[str]:
    piece = board.piece_at(sq)
    if piece is None or piece.piece_type not in _SLIDERS:
        return set()
    opp = not piece.color
    found = set()
    for df, dr in _DIRS[piece.piece_type]:
        hits = _ray_pieces(board, sq, df, dr)
        if len(hits) < 2:
            continue
        p1, p2 = board.piece_at(hits[0]), board.piece_at(hits[1])
        if p1.color != opp or p2.color != opp:
            continue
        if p1.piece_type != chess.KING and (p2.piece_type == chess.KING or _v(p2) > _v(p1)):
            # a pinned pawn is rarely the point of a combination
            if p2.piece_type in (chess.KING, chess.QUEEN, chess.ROOK) and p1.piece_type != chess.PAWN:
                found.add("Pin")
        elif (p1.piece_type == chess.KING or _v(p1) > _v(p2)) and p2.piece_type != chess.PAWN:
            if p1.piece_type in (chess.KING, chess.QUEEN, chess.ROOK):
                found.add("Skewer")
    return found


def _endgame_label(board: chess.Board) -> Optional[str]:
    kinds = {p.piece_type for p in board.piece_map().values()} - {chess.KING}
    if kinds <= {chess.PAWN}:
        return "Pawn Endgame"
    non_pawn = kinds - {chess.PAWN}
    if non_pawn == {chess.ROOK}:
        return "Rook Endgame"
    if non_pawn in ({chess.QUEEN}, {chess.QUEEN, chess.ROOK}):
        return "Queen Endgame"
    if non_pawn == {chess.BISHOP}:
        return "Bishop Endgame"
    if non_pawn == {chess.KNIGHT}:
        return "Knight Endgame"
    if get_game_phase(board) == "endgame":
        return "Endgame"
    return None


def move_motifs(board: chess.Board, move: chess.Move) -> set[str]:
    """Every motif created by `move` in `board` (board is not modified)."""
    piece = board.piece_at(move.from_square)
    if piece is None or move not in board.legal_moves:
        return set()
    me, opp = piece.color, not piece.color
    target = board.piece_at(move.to_square)
    motifs: set[str] = set()

    if board.is_en_passant(move):
        motifs.add("En Passant")
    if move.promotion:
        motifs.add("Promotion")

    # Undefended capture / removal of a defender, judged before the move.
    if target is not None:
        if not board.is_attacked_by(opp, move.to_square) and target.piece_type != chess.PAWN:
            motifs.add("Hanging Piece")
        else:
            defended = [s for s in board.attacks(move.to_square)
                        if board.piece_at(s) and board.piece_at(s).color == opp
                        and s != move.to_square and board.is_attacked_by(me, s)]
            if defended:
                motifs.add("Deflection")

    after = board.copy(stack=False)
    after.push(move)
    if after.is_checkmate():
        motifs.add("Mating Pattern")
        return motifs

    moved = after.piece_at(move.to_square)
    mv = _v(moved)
    safe = _is_safe(after, move.to_square, me)

    # Fork: two or more winnable targets from a piece that is not simply lost.
    targets = [s for s in after.attacks(move.to_square) if _winnable(after, s, mv, opp)]
    if len(targets) >= 2 and safe:
        motifs.add("Fork")

    motifs |= _pin_or_skewer(after, move.to_square)

    # Discovered attack: another of our sliders newly hits something worth hitting.
    for sq in after.pieces(chess.BISHOP, me) | after.pieces(chess.ROOK, me) | after.pieces(chess.QUEEN, me):
        if sq == move.to_square:
            continue
        slider_v = _v(after.piece_at(sq))
        new = after.attacks(sq) & ~board.attacks(sq)
        if any(_winnable(after, s, slider_v, opp) for s in new):
            motifs.add("Discovered Attack")
            break

    # Sacrifice: the piece is left where the opponent can win it, and it is
    # worth more than whatever it just captured.
    if not safe and mv > _v(target) and moved.piece_type != chess.KING:
        motifs.add("Sacrifice")

    if after.is_check():
        motifs.add("King Safety")
    elif target is None and not move.promotion:
        motifs.add("Quiet Move")
    return motifs


def tag_line(
    board: chess.Board,
    line: Iterable[chess.Move],
    *,
    mate: Optional[bool] = None,
    max_player_moves: int = 4,
) -> str:
    """
    Primary tactic of a line whose first move is the player's.

    board  position before the player's first move (not modified)
    line   player move, reply, player move, … (engine PV or puzzle solution)
    mate   True if the engine scores the line as a forced mate for the player;
           None = decide by playing the line out and checking for mate
    """
    moves = list(line)
    if not moves:
        return "General"
    b = board.copy(stack=False)
    found: set[str] = set()
    first_quiet = False
    for i, mv in enumerate(moves):
        if mv not in b.legal_moves:
            break
        if i % 2 == 0:
            if i // 2 >= max_player_moves:
                break
            m = move_motifs(b, mv)
            if i == 0:
                first_quiet = "Quiet Move" in m
            allowed = _ALLOWED_BY_DEPTH[min(i // 2, len(_ALLOWED_BY_DEPTH) - 1)]
            if allowed is not None:
                m &= allowed
            found |= m
        b.push(mv)
    if mate is None:
        mate = b.is_checkmate()
    if mate:
        return "Mating Pattern"
    if not first_quiet:
        found.discard("Quiet Move")
    found.discard("Mating Pattern")   # a mate inside a truncated line not scored as mate
    if found:
        return min(found, key=lambda c: _RANK.get(c, 99))
    return _endgame_label(board) or "General"


def tag_move(board: chess.Board, move: chess.Move) -> str:
    """Single-move convenience wrapper (drop-in for _classify_tactic)."""
    return tag_line(board, [move])
