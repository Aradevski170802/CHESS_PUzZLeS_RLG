"""
Tactical puzzle generator — extracts real puzzles from a player's Chess.com games.

Mode
────
Stockfish-only. Every puzzle goes through four quality gates:

  1. Detection   – the player's move was ≥ PUZZLE_THRESHOLD cp below engine best
  2. Clarity     – PV1-PV2 gap ≥ MIN_CLARITY_CP (unique best move, no dual solutions)
  3. Continuation – at least MIN_PLAYER_MOVES player moves in the solution;
                    opponent responses are "forced" (≥ FORCED_OPP_MARGIN gap)
  4. Verification – re-analyse at VERIFY_DEPTH after extraction to confirm solution[1]
                    is still engine best (guards against low-time detection errors)

Heuristic mode is intentionally removed.  Without engine verification it produces
~40% false-positives that make the puzzle game feel broken.

Puzzle format (Lichess convention)
───────────────────────────────────
  FEN   = position BEFORE the opponent's last move
  Moves = [opp_last_move, player_tactic_move, opp_response, player_tactic_move2, …]
The frontend auto-plays Moves[0] then waits for the player to find Moves[1+].
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
from pathlib import Path
from typing import Optional

import chess
import chess.engine
import chess.pgn

from src.neural.predictor import get_predictor
from src.puzzles.labeller import active_labeller
from src.puzzles.tactic_tagger import tag_line

logger = logging.getLogger(__name__)

# ── Tuning constants ───────────────────────────────────────────────────────────
PUZZLE_THRESHOLD   = 150   # cp: player's move must be this far below engine best
MIN_CLARITY_CP     = 100   # cp: PV1 must beat PV2 by at least this (no dual solutions)
FORCED_OPP_MARGIN  = 80    # cp: opponent's forced response gap — below this = ambiguous
DETECT_TIME        = 0.10  # s:  Stockfish time for blunder-detection pass
VERIFY_DEPTH       = 18    # ply: depth for post-extraction verification
SOLUTION_DEPTH     = 16    # ply: depth used to build the continuation
CONTINUATION_MOVES = 6     # max extra half-moves to append after player's first move
MIN_PLAYER_MOVES   = 1     # puzzle must require at least this many player moves
MAX_PER_GAME       = 5     # hard cap on puzzles extracted per game
SKIP_PLIES         = 8     # ignore the opening
MATE_CP            = 9_000
HEURISTIC_RD       = 150   # nominal deviation of the fixed-formula rating
PUZZLENET_MIN_RD   = 75    # floor on a PuzzleNet puzzle's deviation (see _puzzlenet_rd)

USER_PUZZLES_DIR = Path("data/user_puzzles")


# ── Piece values ──────────────────────────────────────────────────────────────

def _piece_value(pt: chess.PieceType) -> int:
    return {
        chess.PAWN:   100, chess.KNIGHT: 320, chess.BISHOP: 330,
        chess.ROOK:   500, chess.QUEEN:  900, chess.KING:   20_000,
    }[pt]


# ── Public API ────────────────────────────────────────────────────────────────

def generate_from_games(
    pgn_strings:       list[str],
    username:          str,
    stockfish_path:    Optional[str],
    *,
    max_total:         int = 30,
    progress_callback  = None,
    target_categories: list[str] | None = None,
) -> list[dict]:
    """
    Extract verified tactical puzzles from PGN strings.

    When target_categories is provided the function collects up to 3× the
    normal cap, then returns target-category puzzles first so storage and
    session selection naturally favour the player's known weaknesses.
    """
    if not stockfish_path:
        logger.warning("No Stockfish binary found — puzzle generation skipped. "
                       "Install Stockfish and set STOCKFISH_PATH.")
        return []

    # Collect more raw puzzles when we need to filter by category afterwards
    raw_limit    = max_total * 3 if target_categories else max_total
    all_puzzles: list[dict] = []
    seen_fens:   set[str]   = set()

    for i, pgn in enumerate(pgn_strings):
        if len(all_puzzles) >= raw_limit:
            break
        try:
            puzzles = _extract_stockfish(pgn, username, stockfish_path)
            for p in puzzles:
                if p["FEN"] not in seen_fens:
                    seen_fens.add(p["FEN"])
                    all_puzzles.append(p)
                    if len(all_puzzles) >= raw_limit:
                        break
            if puzzles:
                logger.info("Game %d: +%d puzzles (total %d)",
                            i + 1, len(puzzles), len(all_puzzles))
        except Exception as exc:
            logger.warning("Game %d failed: %s", i + 1, exc, exc_info=True)

        if progress_callback:
            progress_callback(i + 1, len(pgn_strings), len(all_puzzles))

    # Targeted categories come first; rest fill the remainder
    if target_categories:
        target_set = set(target_categories)
        targeted   = [p for p in all_puzzles if p["PrimaryCategory"] in target_set]
        others     = [p for p in all_puzzles if p["PrimaryCategory"] not in target_set]
        all_puzzles = targeted + others
        logger.info(
            "Targeted generation: %d/%d puzzles match categories %s",
            len(targeted), len(all_puzzles), target_categories,
        )

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


def load_user_puzzles(username: str, *, rerate: bool = False) -> list[dict]:
    """A user's stored puzzles. With rerate=True, puzzles that only have the
    fixed-formula rating get a PuzzleNet rating (see rerate_puzzles), and the file is
    rewritten once so the work is not repeated."""
    path = USER_PUZZLES_DIR / f"{username.lower()}.json"
    if not path.exists():
        return []
    try:
        puzzles = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if rerate and rerate_puzzles(puzzles):
        try:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(puzzles, indent=2), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            logger.warning("Could not save re-rated puzzles for %s: %s", username, exc)
    return puzzles


# ── Stockfish extractor ───────────────────────────────────────────────────────

def _extract_stockfish(pgn_str: str, username: str, stockfish_path: str) -> list[dict]:
    """
    Four-gate puzzle extraction from a single game PGN.

    Gate 1 – blunder detection (fast, DETECT_TIME per position)
    Gate 2 – clarity check (multi-PV, same fast pass)
    Gate 3 – forced continuation (SOLUTION_DEPTH, up to CONTINUATION_MOVES)
    Gate 4 – solution verification (VERIFY_DEPTH, single position)
    """
    game = chess.pgn.read_game(io.StringIO(pgn_str))
    if game is None:
        return []

    headers     = game.headers
    white_name  = headers.get("White", "").lower()
    player_side = chess.WHITE if username.lower() in white_name else chess.BLACK
    # Chess.com PGNs have [Site "Chess.com"] but [Link "https://..."] for the real URL
    game_url = headers.get("Link", "") or headers.get("URL", "") or headers.get("Site", "")
    if not game_url.startswith("http"):
        game_url = ""

    all_moves = list(game.mainline_moves())
    if len(all_moves) < SKIP_PLIES + 2:
        return []

    # Pre-compute all FENs before the engine opens (faster I/O)
    board  = game.board()
    fen_at: list[str] = []
    for move in all_moves:
        fen_at.append(board.fen())
        board.push(move)

    puzzles: list[dict] = []

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        engine.configure({"Threads": 1, "Hash": 64})

        for ply, actual_move in enumerate(all_moves):
            if len(puzzles) >= MAX_PER_GAME:
                break
            if ply < SKIP_PLIES or ply == 0:
                continue

            b = chess.Board(fen_at[ply])
            if b.turn != player_side or b.is_game_over():
                continue

            # ── Gate 1+2: blunder detection + clarity ──────────────────────
            infos = engine.analyse(
                b, chess.engine.Limit(time=DETECT_TIME), multipv=3
            )
            if not infos:
                continue

            best_move = (infos[0].get("pv") or [None])[0]
            if best_move is None or best_move == actual_move:
                continue  # player found the best move — no puzzle here

            score_best = _pov_cp(infos[0]["score"], b.turn)
            if score_best is None or abs(score_best) > 800:
                continue  # already winning/losing massively — not a clean puzzle

            # Gate 2: clarity — PV1 must dominate PV2
            if len(infos) >= 2:
                score_pv2 = _pov_cp(infos[1]["score"], b.turn)
                if score_pv2 is not None:
                    if (score_best - score_pv2) < MIN_CLARITY_CP:
                        continue  # two moves are nearly equal — dual solution risk

            # Confirm the player actually blundered (eval-drop check)
            b_actual = b.copy()
            b_actual.push(actual_move)
            info_actual  = engine.analyse(b_actual, chess.engine.Limit(time=DETECT_TIME))
            score_actual = _pov_cp(info_actual["score"], b_actual.turn)
            if score_actual is None:
                continue
            eval_drop = score_best - (-score_actual)
            if eval_drop < PUZZLE_THRESHOLD:
                continue

            # ── Gate 3: forced continuation ────────────────────────────────
            setup_fen = fen_at[ply - 1]
            opp_move  = all_moves[ply - 1].uci()
            solution  = [opp_move, best_move.uci()]

            b_sol         = b.copy()
            b_sol.push(best_move)
            player_moves_in_solution = 1

            for _ in range(CONTINUATION_MOVES):
                if b_sol.is_game_over():
                    break

                is_opp_turn = (b_sol.turn != player_side)

                cont_infos = engine.analyse(
                    b_sol,
                    chess.engine.Limit(depth=SOLUTION_DEPTH),
                    multipv=2,
                )
                if not cont_infos:
                    break
                pv = cont_infos[0].get("pv") or []
                if not pv:
                    break
                nxt = pv[0]

                if is_opp_turn:
                    # Opponent's move: check it's forced
                    if len(cont_infos) >= 2:
                        s1 = _pov_cp(cont_infos[0]["score"], b_sol.turn)
                        s2 = _pov_cp(cont_infos[1]["score"], b_sol.turn)
                        if s1 is not None and s2 is not None:
                            if (s1 - s2) < FORCED_OPP_MARGIN:
                                break  # opponent has multiple good options — puzzle ends
                else:
                    player_moves_in_solution += 1

                solution.append(nxt.uci())
                b_sol.push(nxt)

            # Require at least MIN_PLAYER_MOVES player moves
            if player_moves_in_solution < MIN_PLAYER_MOVES:
                continue

            if not _validate_sequence(setup_fen, solution):
                continue

            # ── Gate 4: verification at higher depth ───────────────────────
            # Re-analyse the puzzle start (after opp's setup move) to confirm
            # solution[1] is still the engine's best at a thorough depth.
            b_verify = chess.Board(setup_fen)
            b_verify.push(chess.Move.from_uci(opp_move))
            verify_info = engine.analyse(
                b_verify, chess.engine.Limit(depth=VERIFY_DEPTH)
            )
            verify_best = (verify_info.get("pv") or [None])[0]
            if verify_best is None:
                continue
            if verify_best.uci() != best_move.uci():
                continue  # engine changed its mind at higher depth — skip

            # ── Build the puzzle dict ──────────────────────────────────────
            # Record the clarity gap at detection time for the quality dashboard
            clarity_cp: Optional[int] = None
            if len(infos) >= 2:
                s1 = _pov_cp(infos[0]["score"], b.turn)
                s2 = _pov_cp(infos[1]["score"], b.turn)
                if s1 is not None and s2 is not None:
                    clarity_cp = s1 - s2

            # Label the whole solution line (player move first), trusting the
            # engine's mate verdict. With a trained PuzzleNet installed the network
            # labels the line and estimates its difficulty with an uncertainty;
            # otherwise the rule-based tagger and the fixed formula are used.
            best_cp = _pov_cp(infos[0]["score"], b.turn) if infos else None
            mate_verdict = (best_cp is not None and best_cp >= MATE_CP) or None
            line = [chess.Move.from_uci(u) for u in solution[1:]]
            tactic = tag_line(b, line, mate=mate_verdict)
            rating = _estimate_rating(eval_drop, len(solution))
            rating_dev, rating_model = HEURISTIC_RD, "heuristic"
            net_rating = net_rd = None
            net = get_predictor()
            if net is not None:
                start = chess.Board(setup_fen)          # keeps the opponent's move
                start.push(chess.Move.from_uci(opp_move))
                pred = net.predict_line(start, line, mate=mate_verdict)
                net_rating, net_rd = _clamp_rating(pred.rating), _puzzlenet_rd(pred.rating_sd)
                if use_puzzlenet_rating():
                    rating, rating_dev, rating_model = net_rating, net_rd, "puzzlenet"
                if active_labeller() == "neural":
                    tactic = pred.category
            puzzle_id = "gen_" + hashlib.md5(
                (setup_fen + "".join(solution)).encode()
            ).hexdigest()[:8]

            puzzles.append(_make_puzzle(
                puzzle_id, setup_fen, solution,
                rating, tactic, game_url,
                eval_drop=eval_drop,
                clarity_cp=clarity_cp,
                rating_deviation=rating_dev,
                rating_model=rating_model,
                puzzlenet_rating=net_rating,
                puzzlenet_rd=net_rd,
            ))

    return puzzles


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_puzzle(
    puzzle_id:  str,
    setup_fen:  str,
    solution:   list[str],
    rating:     int,
    tactic:     str,
    game_url:   str,
    *,
    eval_drop:  int = 0,
    clarity_cp: Optional[int] = None,
    rating_deviation: int = 150,
    rating_model: str = "heuristic",
    puzzlenet_rating: Optional[int] = None,
    puzzlenet_rd: Optional[int] = None,
) -> dict:
    player_moves = len([i for i in range(1, len(solution), 2)])  # odd indices
    return {
        "PuzzleId":        puzzle_id,
        "FEN":             setup_fen,
        "Moves":           " ".join(solution),
        "Rating":          rating,
        "RatingDeviation": rating_deviation,
        "ratingModel":     rating_model,
        "puzzlenetRating": puzzlenet_rating,
        "puzzlenetRd":     puzzlenet_rd,
        "Popularity":      100,
        "NbPlays":         0,
        "Themes":          tactic.lower().replace(" ", "") + " personal",
        "GameUrl":         game_url,
        "OpeningTags":     None,
        "PrimaryCategory": tactic,
        "DifficultyTier":  _difficulty_tier(rating),
        "Categories":      [tactic],
        "source":          "generated",
        "generatedBy":     "stockfish",
        "evalDrop":        eval_drop,
        "clarityCp":       clarity_cp,
        "playerMoves":     player_moves,
    }


def _classify_tactic(board: chess.Board, move: chess.Move) -> str:
    piece  = board.piece_at(move.from_square)
    target = board.piece_at(move.to_square)
    if piece is None:
        return "General"

    b2  = board.copy()
    b2.push(move)
    opp = not piece.color

    if b2.is_checkmate():
        return "Mating Pattern"

    if move.promotion:
        return "Promotion"

    attacks      = b2.attacks(move.to_square)
    attacked_opp = [sq for sq in attacks if b2.piece_at(sq) and b2.piece_at(sq).color == opp]
    high_val_attacked = [sq for sq in attacked_opp
                         if b2.piece_at(sq).piece_type in (chess.QUEEN, chess.ROOK, chess.KING)]
    # A double attack on two *loose* pieces is still a fork even when neither is
    # a queen, rook or king — requiring heavy material missed knight-forks-two-
    # minor-pieces entirely.
    loose_attacked = [sq for sq in attacked_opp if not b2.is_attacked_by(opp, sq)]
    if len(attacked_opp) >= 2 and (high_val_attacked or len(loose_attacked) >= 2):
        return "Fork"

    if target and not board.is_attacked_by(opp, move.to_square):
        return "Hanging Piece"

    # Capturing a piece that was defending something else is a deflection /
    # removal of the guard, not an anonymous trade.
    if target:
        defended_by_target = [
            sq for sq in board.attacks(move.to_square)
            if board.piece_at(sq) and board.piece_at(sq).color == opp
            and board.is_attacked_by(piece.color, sq)
        ]
        if defended_by_target:
            return "Deflection"

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
        attacker_val = _piece_value(piece.piece_type)
        target_val   = _piece_value(target.piece_type)
        if attacker_val > target_val + 100:
            return "Sacrifice"

    if piece.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN) and target:
        return "X-Ray Attack"

    total_pieces = len(b2.piece_map())
    if total_pieces <= 12:
        if piece.piece_type == chess.PAWN:
            return "Pawn Endgame"
        if piece.piece_type == chess.ROOK:
            return "Rook Endgame"
        return "Endgame"

    # Neither a capture, a check, nor a promotion, yet the engine says it is
    # decisively best — that is exactly Lichess's "quietMove" motif.  Labelling
    # these "General" left a third of every generated set without a teachable
    # name and polluted the bandit's weakness model.
    if not target and not b2.is_check():
        return "Quiet Move"

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


def _clamp_rating(rating: float) -> int:
    return int(max(400, min(3200, round(rating))))


def _puzzlenet_rd(content_sd: float) -> int:
    """Rating deviation for a PuzzleNet-rated puzzle. The network's content
    uncertainty is floored at PUZZLENET_MIN_RD: a puzzle nobody has played cannot be
    rated more precisely than a well-played Lichess puzzle (RD ~75), and the model
    was calibrated on Lichess puzzles, not on puzzles mined from a player's games."""
    return int(round(max(PUZZLENET_MIN_RD, content_sd)))


def use_puzzlenet_rating() -> bool:
    """Whether a mined puzzle is SERVED at PuzzleNet's difficulty.

    Off by default. On the app's own logs the network rates a mined puzzle 493
    points harder than the formula, while players solve 86.7% of mined puzzles
    (against 80.0% of pool puzzles) because the position comes from their own game:
    prequential log loss is 0.599 with the formula and 0.754 with the network. The
    level does not transfer from Lichess puzzles to a player's own positions, and
    123 attempts are far too few to fit the offset that would fix it, so the
    network's estimate is recorded but not used until there is data to calibrate it.
    """
    return os.environ.get("MINED_RATING", "heuristic").lower() == "puzzlenet"


def rerate_puzzles(puzzles: list[dict]) -> int:
    """Record PuzzleNet's difficulty estimate on mined puzzles that lack one, in
    place. It becomes the serving rating only when MINED_RATING=puzzlenet, in which
    case the formula's estimate is kept as `heuristicRating`. The category is never
    relabelled: it decides where a stored puzzle is served, and changing it silently
    would change a player's training set. Returns how many puzzles changed."""
    net = get_predictor()
    if net is None:
        return 0
    todo = [p for p in puzzles
            if p.get("source") == "generated" and p.get("puzzlenetRating") is None
            and p.get("FEN") and p.get("Moves")]
    changed = 0
    for p in todo:
        try:
            pred = net.predict_puzzle(p["FEN"], p["Moves"])
        except Exception:
            continue
        p["puzzlenetRating"] = _clamp_rating(pred.rating)
        p["puzzlenetRd"] = _puzzlenet_rd(pred.rating_sd)
        if use_puzzlenet_rating():
            p["heuristicRating"] = p.get("Rating")
            p["Rating"] = p["puzzlenetRating"]
            p["RatingDeviation"] = p["puzzlenetRd"]
            p["DifficultyTier"] = _difficulty_tier(p["Rating"])
            p["ratingModel"] = "puzzlenet"
        changed += 1
    return changed


def _estimate_rating(eval_drop: int, solution_len: int) -> int:
    base  = 800
    base += min(eval_drop - PUZZLE_THRESHOLD, 700) * 0.6
    base += (solution_len - 2) * 150    # longer = harder
    return max(700, min(2400, int(base)))


def _difficulty_tier(rating: int) -> str:
    if rating < 1000: return "Beginner"
    if rating < 1300: return "Intermediate"
    if rating < 1600: return "Advanced"
    if rating < 1900: return "Hard"
    return "Expert"
