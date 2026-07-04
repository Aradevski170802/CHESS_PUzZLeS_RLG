"""
Tactical puzzle generator — turns a player's own game history into puzzles.

Two modes
─────────
  heuristic  (default, no engine needed)
    Scans every position with python-chess alone.  Detects:
      • Hanging-piece captures  (opponent left a piece undefended)
      • Forks  (one move attacks 2+ valuable opponent pieces)
      • Checkmate in one
      • Promotion wins
    Runs in milliseconds per game.

  stockfish  (when a Stockfish binary is available)
    Full eval-drop analysis (≥150 cp gap between best move and played move).
    Much more thorough — finds deep combinations and quiet tactics.
    Set DETECT_TIME low (0.01 s) for speed on modest hardware.

The public entry point `generate_from_games` chooses the mode automatically:
pass stockfish_path=None (or omit) for heuristic; pass a path for Stockfish.

Puzzle format
─────────────
Compatible with the Lichess schema used everywhere else in the app:
  FEN    = position BEFORE the opponent's last move  (Lichess convention)
  Moves  = [opp_last_move, player_tactic_move, continuation…]
The frontend auto-plays Moves[0] (opponent's move) then waits for the player
to find Moves[1].
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Optional

import chess
import chess.pgn

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
PUZZLE_THRESHOLD   = 120    # centipawn drop (slightly lower catches more patterns)
MAX_PER_GAME       = 3      # hard cap on puzzles extracted per game
SKIP_PLIES         = 8      # ignore opening
DETECT_TIME        = 0.05   # Stockfish seconds per position — must be ≥0.05 to reach depth 10+
SOLUTION_DEPTH     = 16     # Stockfish depth for continuation line
CONTINUATION_MOVES = 4
MATE_CP            = 9_000

# Minimum value of a piece that counts as "worth forking"
FORK_MIN_VALUE = 300        # knight / bishop or higher

USER_PUZZLES_DIR = Path("data/user_puzzles")


# ── Piece values ──────────────────────────────────────────────────────────────

def _piece_value(pt: chess.PieceType) -> int:
    return {
        chess.PAWN:   100, chess.KNIGHT: 320, chess.BISHOP: 330,
        chess.ROOK:   500, chess.QUEEN:  900, chess.KING:   20_000,
    }[pt]


# ── Public API ────────────────────────────────────────────────────────────────

def generate_from_games(
    pgn_strings: list[str],
    username: str,
    stockfish_path: Optional[str] = None,
    *,
    max_total: int = 30,
    progress_callback=None,
) -> list[dict]:
    """
    Extract tactical puzzles from a list of PGN strings.

    Parameters
    ----------
    pgn_strings      : raw PGN strings (Chess.com API format)
    username         : Chess.com username — used to determine player colour
    stockfish_path   : path to Stockfish binary, or None for heuristic mode
    max_total        : hard cap on total puzzles returned
    progress_callback: optional callable(done, total)
    """
    all_puzzles: list[dict] = []
    seen_fens:   set[str]   = set()

    engine_fn = (
        _extract_stockfish if stockfish_path else _extract_heuristic
    )

    for i, pgn in enumerate(pgn_strings):
        if len(all_puzzles) >= max_total:
            break
        try:
            puzzles = engine_fn(pgn, username, stockfish_path)
            for p in puzzles:
                if p["FEN"] not in seen_fens:
                    seen_fens.add(p["FEN"])
                    all_puzzles.append(p)
                    if len(all_puzzles) >= max_total:
                        break
            if puzzles:
                logger.info("Game %d: found %d puzzle(s) (total so far: %d)",
                            i + 1, len(puzzles), len(all_puzzles))
        except Exception as exc:
            logger.warning("Game %d failed: %s", i + 1, exc)

        if progress_callback:
            progress_callback(i + 1, len(pgn_strings), len(all_puzzles))

    return all_puzzles[:max_total]


def save_user_puzzles(username: str, puzzles: list[dict]) -> Path:
    """Persist puzzles for a user, merging with any existing ones."""
    USER_PUZZLES_DIR.mkdir(parents=True, exist_ok=True)
    path = USER_PUZZLES_DIR / f"{username.lower()}.json"

    existing: list[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = []

    existing_ids = {p["PuzzleId"] for p in existing}
    new    = [p for p in puzzles if p["PuzzleId"] not in existing_ids]
    merged = existing + new

    path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    logger.info("Saved %d new puzzles for %s (total %d)", len(new), username, len(merged))
    return path


def load_user_puzzles(username: str) -> list[dict]:
    path = USER_PUZZLES_DIR / f"{username.lower()}.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


# ── Heuristic extractor (no engine) ──────────────────────────────────────────

def _extract_heuristic(pgn_str: str, username: str, _sf_path) -> list[dict]:
    """
    Find tactical puzzles using python-chess board analysis only.
    Detects: hanging pieces, forks, checkmate in one, promotion.
    Returns up to MAX_PER_GAME puzzles.
    """
    game = chess.pgn.read_game(io.StringIO(pgn_str))
    if game is None:
        return []

    headers     = game.headers
    white_name  = headers.get("White", "").lower()
    player_side = chess.WHITE if username.lower() in white_name else chess.BLACK
    game_url    = headers.get("Site", "")

    all_moves = list(game.mainline_moves())
    if len(all_moves) < SKIP_PLIES + 2:
        return []

    board  = game.board()
    fen_at: list[str] = []
    for move in all_moves:
        fen_at.append(board.fen())
        board.push(move)

    puzzles: list[dict] = []

    for ply, actual_move in enumerate(all_moves):
        if len(puzzles) >= MAX_PER_GAME:
            break
        if ply < SKIP_PLIES or ply == 0:
            continue

        b = chess.Board(fen_at[ply])
        if b.turn != player_side or b.is_game_over():
            continue

        # Find the best tactic available at this position
        best = _best_heuristic_move(b)
        if best is None:
            continue

        tactic_move, tactic_type, material_gain = best

        # If the player already found the tactic, skip (not a missed opportunity)
        if tactic_move == actual_move:
            continue

        # Build solution sequence
        setup_fen  = fen_at[ply - 1]
        opp_move   = all_moves[ply - 1].uci()
        solution   = [opp_move, tactic_move.uci()]

        # Add one forced continuation (opponent's best recapture, if any)
        b2 = b.copy()
        b2.push(tactic_move)
        if not b2.is_game_over():
            recapture = _forced_recapture(b2, tactic_move.to_square)
            if recapture:
                solution.append(recapture.uci())

        if not _validate_sequence(setup_fen, solution):
            continue

        rating    = _heuristic_rating(tactic_type, material_gain)
        puzzle_id = "gen_" + hashlib.md5(
            (setup_fen + "".join(solution)).encode()
        ).hexdigest()[:8]

        puzzles.append(_make_puzzle(puzzle_id, setup_fen, solution,
                                    rating, tactic_type, game_url, mode="heuristic"))

    return puzzles


def _best_heuristic_move(board: chess.Board) -> Optional[tuple]:
    """
    Return (move, tactic_type, material_gain) for the best tactical move
    available, or None if no clear tactic exists.

    Priority: checkmate > hanging piece > fork > promotion
    """
    opp = not board.turn

    # ── Checkmate in one ─────────────────────────────────────────────────────
    for move in board.legal_moves:
        b2 = board.copy(); b2.push(move)
        if b2.is_checkmate():
            return (move, "Mating Pattern", 9000)

    # ── Hanging-piece capture ─────────────────────────────────────────────────
    # A capture where the captured piece is either undefended, or defended only
    # by pieces more valuable than the attacker.
    best_hang: Optional[tuple] = None
    best_gain = 0

    for move in board.legal_moves:
        target = board.piece_at(move.to_square)
        if target is None or target.color == board.turn:
            continue
        target_val = _piece_value(target.piece_type)

        # After the capture, is the capturing piece lost to a recapture?
        b2 = board.copy(); b2.push(move)
        attacker     = b2.piece_at(move.to_square)
        attacker_val = _piece_value(attacker.piece_type) if attacker else 0

        if b2.is_attacked_by(opp, move.to_square):
            # We can be recaptured — net gain = target - attacker
            net = target_val - attacker_val
        else:
            net = target_val   # free capture

        if net > best_gain and net > 50:
            best_gain = net
            best_hang = (move, "Hanging Piece", net)

    if best_hang:
        return best_hang

    # ── Fork ─────────────────────────────────────────────────────────────────
    for move in board.legal_moves:
        b2 = board.copy(); b2.push(move)
        attacked = [
            sq for sq in b2.attacks(move.to_square)
            if b2.piece_at(sq) and b2.piece_at(sq).color == opp
            and _piece_value(b2.piece_at(sq).piece_type) >= FORK_MIN_VALUE
        ]
        # Moving piece itself must not be immediately capturable for free
        mover_val = _piece_value(b2.piece_at(move.to_square).piece_type) if b2.piece_at(move.to_square) else 0
        safe = not b2.is_attacked_by(opp, move.to_square) or any(
            _piece_value(b2.piece_at(sq).piece_type) >= mover_val
            for sq in b2.attackers(opp, move.to_square)
            if b2.piece_at(sq)
        )
        if len(attacked) >= 2 and safe:
            gain = sum(_piece_value(b2.piece_at(sq).piece_type) for sq in attacked[:2])
            return (move, "Fork", gain)

    # ── Promotion ─────────────────────────────────────────────────────────────
    for move in board.legal_moves:
        if move.promotion == chess.QUEEN:
            return (move, "Promotion", 800)

    return None


def _forced_recapture(board: chess.Board, target_sq: chess.Square) -> Optional[chess.Move]:
    """Return the opponent's recapture on target_sq, if it exists and is unique."""
    recaptures = [
        m for m in board.legal_moves
        if m.to_square == target_sq
    ]
    return recaptures[0] if len(recaptures) == 1 else None


def _heuristic_rating(tactic_type: str, material_gain: int) -> int:
    base = {
        "Mating Pattern": 1000,
        "Fork":           800,
        "Hanging Piece":  650,
        "Promotion":      750,
    }.get(tactic_type, 700)
    base += min(material_gain, 600) * 0.3
    return max(600, min(1800, int(base)))


# ── Stockfish extractor ───────────────────────────────────────────────────────

def _extract_stockfish(pgn_str: str, username: str, stockfish_path: str) -> list[dict]:
    """
    Extract puzzles using Stockfish eval-drop analysis.
    Uses low time limits for speed on modest hardware.
    """
    import chess.engine

    game = chess.pgn.read_game(io.StringIO(pgn_str))
    if game is None:
        return []

    headers     = game.headers
    white_name  = headers.get("White", "").lower()
    player_side = chess.WHITE if username.lower() in white_name else chess.BLACK
    game_url    = headers.get("Site", "")

    all_moves = list(game.mainline_moves())
    if len(all_moves) < SKIP_PLIES + 2:
        return []

    board  = game.board()
    fen_at: list[str] = []
    for move in all_moves:
        fen_at.append(board.fen())
        board.push(move)

    puzzles: list[dict] = []

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        engine.configure({"Threads": 1, "Hash": 16})

        for ply, move in enumerate(all_moves):
            if len(puzzles) >= MAX_PER_GAME:
                break
            if ply < SKIP_PLIES or ply == 0:
                continue

            b = chess.Board(fen_at[ply])
            if b.turn != player_side or b.is_game_over():
                continue

            info_best  = engine.analyse(b, chess.engine.Limit(time=DETECT_TIME))
            best_move  = (info_best.get("pv") or [None])[0]
            if best_move is None or best_move == move:
                continue

            score_best = _pov_cp(info_best["score"], b.turn)
            if score_best is None or abs(score_best) > 800:
                continue

            b_actual     = b.copy(); b_actual.push(move)
            info_actual  = engine.analyse(b_actual, chess.engine.Limit(time=DETECT_TIME))
            score_actual = _pov_cp(info_actual["score"], b_actual.turn)
            if score_actual is None:
                continue

            eval_drop = score_best - (-score_actual)
            if eval_drop < PUZZLE_THRESHOLD:
                continue

            setup_fen = fen_at[ply - 1]
            opp_move  = all_moves[ply - 1].uci()
            solution  = [opp_move, best_move.uci()]

            b_sol = b.copy(); b_sol.push(best_move)
            for _ in range(CONTINUATION_MOVES):
                if b_sol.is_game_over():
                    break
                resp = engine.analyse(b_sol, chess.engine.Limit(depth=SOLUTION_DEPTH))
                pv   = resp.get("pv") or []
                if not pv:
                    break
                nxt = pv[0]
                solution.append(nxt.uci())
                b_sol.push(nxt)
                if b_sol.turn == player_side and len(solution) >= 5:
                    break

            if not _validate_sequence(setup_fen, solution):
                continue

            tactic    = _classify_tactic(b, best_move)
            rating    = _estimate_rating(eval_drop, len(solution))
            puzzle_id = "gen_" + hashlib.md5(
                (setup_fen + "".join(solution)).encode()
            ).hexdigest()[:8]

            puzzles.append(_make_puzzle(puzzle_id, setup_fen, solution,
                                        rating, tactic, game_url, mode="stockfish"))

    return puzzles


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_puzzle(
    puzzle_id: str,
    setup_fen: str,
    solution:  list[str],
    rating:    int,
    tactic:    str,
    game_url:  str,
    *,
    mode: str = "heuristic",
) -> dict:
    return {
        "PuzzleId":        puzzle_id,
        "FEN":             setup_fen,
        "Moves":           " ".join(solution),
        "Rating":          rating,
        "RatingDeviation": 200,
        "Popularity":      100,
        "NbPlays":         0,
        "Themes":          tactic.lower().replace(" ", "") + " personal",
        "GameUrl":         game_url,
        "OpeningTags":     None,
        "PrimaryCategory": tactic,
        "DifficultyTier":  _difficulty_tier(rating),
        "Categories":      [tactic],
        "source":          "generated",
        "generatedBy":     mode,
    }


def _classify_tactic(board: chess.Board, move: chess.Move) -> str:
    piece  = board.piece_at(move.from_square)
    target = board.piece_at(move.to_square)
    if piece is None:
        return "General"

    b2  = board.copy(); b2.push(move)
    opp = not piece.color

    if b2.is_checkmate():
        return "Mating Pattern"

    attacks          = b2.attacks(move.to_square)
    attacked_opp     = [sq for sq in attacks if b2.piece_at(sq) and b2.piece_at(sq).color == opp]
    high_val_attacked = [sq for sq in attacked_opp
                         if b2.piece_at(sq).piece_type in (chess.QUEEN, chess.ROOK, chess.KING)]
    if len(attacked_opp) >= 2 and high_val_attacked:
        return "Fork"

    if target and not board.is_attacked_by(opp, move.to_square):
        return "Hanging Piece"

    if piece.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        opp_king_sq = b2.king(opp)
        if opp_king_sq is not None:
            between = chess.SquareSet(chess.between(move.to_square, opp_king_sq))
            pinned  = [sq for sq in between if b2.piece_at(sq) and b2.piece_at(sq).color == opp]
            if len(pinned) == 1:
                return "Pin"
            if len(pinned) == 0 and move.to_square in b2.attacks(opp_king_sq):
                return "Skewer"

    if b2.is_check():
        opp_king_sq = b2.king(opp)
        if opp_king_sq and opp_king_sq not in b2.attacks(move.to_square):
            return "Discovered Attack"
        return "King Safety"

    if target:
        if _piece_value(piece.piece_type) > _piece_value(target.piece_type) + 100:
            return "Sacrifice"

    if move.promotion:
        return "Promotion"

    if piece.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN) and target:
        return "X-Ray Attack"

    total_pieces = len(b2.piece_map())
    if total_pieces <= 12:
        if piece.piece_type == chess.PAWN:
            return "Pawn Endgame"
        if piece.piece_type == chess.ROOK:
            return "Rook Endgame"
        return "Endgame"

    return "General"


def _pov_cp(score, turn: chess.Color) -> Optional[int]:
    try:
        pov = score.pov(turn)
        if pov.is_mate():
            m = pov.mate()
            return MATE_CP if (m is not None and m > 0) else -MATE_CP
        return pov.score()
    except Exception:
        return None


def _validate_sequence(fen: str, moves: list[str]) -> bool:
    try:
        board = chess.Board(fen)
        for uci in moves:
            m = chess.Move.from_uci(uci)
            if m not in board.legal_moves:
                return False
            board.push(m)
        return True
    except Exception:
        return False


def _estimate_rating(eval_drop: int, solution_len: int) -> int:
    base  = 700
    base += min(eval_drop - PUZZLE_THRESHOLD, 600) * 0.5
    base += (solution_len - 2) * 120
    return max(600, min(2200, int(base)))


def _heuristic_rating(tactic_type: str, material_gain: int) -> int:
    base = {
        "Mating Pattern": 1000,
        "Fork":           800,
        "Hanging Piece":  650,
        "Promotion":      750,
    }.get(tactic_type, 700)
    base += min(material_gain, 600) * 0.3
    return max(600, min(1800, int(base)))


def _difficulty_tier(rating: int) -> str:
    if rating < 1000: return "Beginner"
    if rating < 1300: return "Intermediate"
    if rating < 1600: return "Advanced"
    if rating < 1900: return "Hard"
    return "Expert"
