"""
Parallel Stockfish analysis for player game histories.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 WHAT IS MEASURED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For every move the player makes (after the first 8 half-moves), the engine
evaluates the position with MultiPV 2, i.e. the best move AND the runner-up.
That gives two kinds of evidence:

1. Errors — the player's move lost ground against the best move.
   Loss is graded on WIN PROBABILITY, not raw centipawns: dropping 300 cp at
   +900 changes nothing, dropping 300 cp at +50 loses the game. Centipawns are
   mapped to a win percentage with Lichess's logistic curve

       win% = 50 + 50 · (2 / (1 + e^(−0.00368208 · cp)) − 1)

   and severity uses the same shape of thresholds Lichess uses:
   inaccuracy ≥ 5, mistake ≥ 10, blunder ≥ 15 win-% points lost.

2. Opportunities — positions where exactly one move works. A position is
   "critical" when the best move beats the runner-up by ≥ 10 win-% points,
   the same "single clear solution" idea that defines a puzzle. Each critical
   position is labelled with the tactic of the best move and scored as a HIT
   (player lost < 5 win-% points, i.e. found it or an equivalent) or a MISS.

   Opportunities are what make per-category weakness measurable: a player's
   fork weakness is misses / opportunities, not how many fork errors happen
   to appear in their games. Without the denominator, players who simply
   meet more fork positions look worse at forks.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 REPRODUCIBILITY AND SPEED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Search is limited by NODES, not time. A time limit reaches a different depth
depending on machine load, so the same game could yield different errors on
two runs; a node limit with one thread and a fresh engine per game is
deterministic. When the player plays the best or second-best move, its value
is already known from the MultiPV search, so no second engine call is needed.
Parallelism is at game level, one Stockfish subprocess per thread.
"""

from __future__ import annotations

import io
import logging
import math
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import chess
import chess.engine
import chess.pgn

from src.data.pgn_parser import get_game_phase
from src.puzzles.labeller import label_line

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
INACCURACY_CP  = 50    # legacy centipawn thresholds, kept for reporting
MISTAKE_CP     = 100
BLUNDER_CP     = 200

INACCURACY_WP  = 5.0   # win-% points lost → inaccuracy
MISTAKE_WP     = 10.0  # → mistake
BLUNDER_WP     = 15.0  # → blunder
CRITICAL_WP_GAP = 10.0 # best move beats runner-up by this much → "only move"
HIT_WP_TOL     = 5.0   # player within this of best → found it

SKIP_PLIES     = 8     # ignore first 8 half-moves (opening book)
ANALYSIS_NODES = 75_000    # deterministic search budget per position (≈ depth 12–14,
                           # at least as deep as the old 50 ms limit, and reproducible)
ANALYSIS_TIME  = 0.05      # legacy time limit, used only if explicitly requested
MATE_CP        = 9000  # centipawn value assigned to forced mate
MAX_CP_LOSS_FOR_AVG = 1000  # stop a single mate-scale swing dominating ACPL
PV_PLIES_FOR_TAGGING = 7    # player-reply-player-… plies of the PV given to the tagger

_WIN_K = 0.00368208


def win_percent(cp: Optional[float]) -> Optional[float]:
    """Lichess's logistic map from centipawns (side to move) to win % (0-100)."""
    if cp is None:
        return None
    cp = max(-MATE_CP, min(MATE_CP, cp))
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-_WIN_K * cp)) - 1.0)


def time_class_of(time_control: str) -> str:
    """
    Classify a PGN TimeControl tag. Uses the estimated game duration
    base + 40 × increment (Lichess's convention): < 3 min bullet,
    < 10 min blitz, otherwise rapid. "1/86400"-style tags are daily.
    """
    tc = (time_control or "").strip()
    if not tc or tc == "-":
        return "unknown"
    if "/" in tc:
        return "daily"
    try:
        base, _, inc = tc.partition("+")
        est = float(base) + 40.0 * float(inc or 0)
    except ValueError:
        return "unknown"
    if est < 180:
        return "bullet"
    if est < 600:
        return "blitz"
    return "rapid"


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MoveError:
    """A single mistake/blunder by the player."""
    move_number:  int    # full move number in the game
    fen_before:   str    # FEN of the position BEFORE the bad move
    move_uci:     str    # the move the player played (UCI notation e.g. "e2e4")
    cp_loss:      int    # centipawn loss relative to best move
    severity:     str    # "inaccuracy" | "mistake" | "blunder"
    phase:        str    # "opening" | "middlegame" | "endgame"
    category:     str = "General"  # tactical theme of the MISSED best move, via
    # src.puzzles.generator._classify_tactic run on the engine's principal
    # variation from fen_before. "General" when the engine reported no PV
    # within the search budget, or the move didn't fit a specific pattern.
    wp_loss:      float = 0.0      # win-% points lost (severity is graded on this)


@dataclass
class Opportunity:
    """A critical position: exactly one clearly-best move existed."""
    category: str      # tactic of the best move (_classify_tactic)
    hit:      bool     # did the player find it (within HIT_WP_TOL)?
    wp_loss:  float    # win-% points the player's move cost
    phase:    str      # game phase before the move
    ply:      int      # half-move index in the game


@dataclass
class GameAnalysis:
    """Full analysis result for one game."""
    errors:        list[MoveError] = field(default_factory=list)
    opportunities: list[Opportunity] = field(default_factory=list)
    player_color:  str  = "white"
    player_won:    Optional[bool] = None    # True / False / None (draw)
    opening_eco:   str  = ""
    opening_family: str = ""
    num_moves:     int  = 0
    avg_cp_loss:   float = 0.0
    avg_wp_loss:   float = 0.0              # mean win-% points lost per move
    time_class:    str  = "unknown"         # bullet | blitz | rapid | daily
    end_time:      Optional[int] = None     # unix seconds, from PGN headers
    player_rating: Optional[int] = None
    opponent_rating: Optional[int] = None
    failed:        bool = False             # True if engine error or bad PGN

    @property
    def blunder_count(self)    -> int: return sum(1 for e in self.errors if e.severity == "blunder")
    @property
    def mistake_count(self)    -> int: return sum(1 for e in self.errors if e.severity == "mistake")
    @property
    def inaccuracy_count(self) -> int: return sum(1 for e in self.errors if e.severity == "inaccuracy")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def find_stockfish() -> Optional[str]:
    """
    Locate the Stockfish binary on this system.
    Returns the full path string, or None if not found.
    """
    candidates = [
        "stockfish",
        "stockfish.exe",
        # Project-local binary (checked first)
        str(Path(__file__).resolve().parents[2] / "stockfish" / "stockfish-windows-x86-64-avx2.exe"),
        str(Path(__file__).resolve().parents[2] / "stockfish" / "stockfish.exe"),
        r"C:\stockfish\stockfish.exe",
        r"C:\Users\frogo\stockfish\stockfish.exe",
        "/usr/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/opt/homebrew/bin/stockfish",
    ]
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        if Path(candidate).is_file():
            return candidate
    return None


def severity_of(wp_loss: float) -> Optional[str]:
    if wp_loss >= BLUNDER_WP:
        return "blunder"
    if wp_loss >= MISTAKE_WP:
        return "mistake"
    if wp_loss >= INACCURACY_WP:
        return "inaccuracy"
    return None


def _player_color(headers: chess.pgn.Headers, username: str) -> str:
    """Exact username match first; the old substring test mis-assigned the
    colour whenever the player's name was contained in the opponent's
    (e.g. "bob" vs "bobby_99")."""
    user = username.lower()
    white, black = headers.get("White", "").lower(), headers.get("Black", "").lower()
    if user == white:
        return "white"
    if user == black:
        return "black"
    return "white" if user in white else "black"


def _end_time(headers: chess.pgn.Headers) -> Optional[int]:
    for d_key, t_key in (("EndDate", "EndTime"), ("UTCDate", "UTCTime"), ("Date", None)):
        d = headers.get(d_key, "")
        if not d or "?" in d:
            continue
        t = headers.get(t_key, "00:00:00") if t_key else "00:00:00"
        try:
            dt = datetime.strptime(f"{d} {t}", "%Y.%m.%d %H:%M:%S").replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue
    return None


def _int_or_none(v: str) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def analyze_game(
    pgn_str: str,
    username: str,
    stockfish_path: str,
    *,
    time_limit: Optional[float] = None,
    nodes: int = ANALYSIS_NODES,
) -> GameAnalysis:
    """
    Analyse a single game PGN string with Stockfish.

    Opens its own engine subprocess — safe to call from multiple threads
    simultaneously (each thread has its own engine instance).

    Parameters
    ----------
    pgn_str        : raw PGN string from Chess.com API
    username       : Chess.com username (to identify which colour to track)
    stockfish_path : path to Stockfish binary
    time_limit     : seconds per position; if given, overrides `nodes`
                     (non-deterministic — kept only for backwards compatibility)
    nodes          : deterministic node budget per position (default)

    Returns
    -------
    GameAnalysis with errors, opportunities and game metadata.
    """
    result = GameAnalysis()
    limit = (chess.engine.Limit(time=time_limit) if time_limit
             else chess.engine.Limit(nodes=nodes))

    # ── Parse PGN ────────────────────────────────────────────────────────
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_str))
        if game is None:
            result.failed = True
            return result
    except Exception as e:
        logger.debug("PGN parse error: %s", e)
        result.failed = True
        return result

    headers = game.headers
    player_color = _player_color(headers, username)
    result.player_color = player_color

    # Game result from player's perspective
    game_result = headers.get("Result", "*")
    if player_color == "white":
        result.player_won = True if game_result == "1-0" else (False if game_result == "0-1" else None)
    else:
        result.player_won = True if game_result == "0-1" else (False if game_result == "1-0" else None)

    # Opening info from headers
    eco_url = headers.get("ECOUrl", "")
    result.opening_eco    = headers.get("ECO", "")
    result.opening_family = (
        eco_url.rstrip("/").split("/")[-1].replace("-", " ")
        if eco_url else headers.get("Opening", "Unknown")
    )
    result.time_class = time_class_of(headers.get("TimeControl", ""))
    result.end_time = _end_time(headers)
    own, opp = ("WhiteElo", "BlackElo") if player_color == "white" else ("BlackElo", "WhiteElo")
    result.player_rating = _int_or_none(headers.get(own))
    result.opponent_rating = _int_or_none(headers.get(opp))

    moves = list(game.mainline_moves())
    result.num_moves = len(moves) // 2

    if not moves:
        return result  # empty game

    player_turn = chess.WHITE if player_color == "white" else chess.BLACK

    # ── Stockfish analysis ────────────────────────────────────────────────
    try:
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            # Use 1 thread per engine — we parallelise at game level
            engine.configure({"Threads": 1, "Hash": 32})

            board = game.board()
            total_cp_loss = 0.0
            total_wp_loss = 0.0
            player_moves_n = 0
            last_capture_sq: Optional[int] = None

            for ply, move in enumerate(moves):
                if board.turn != player_turn or ply < SKIP_PLIES:
                    last_capture_sq = move.to_square if board.is_capture(move) else None
                    board.push(move)
                    continue

                infos = engine.analyse(board, limit, multipv=2)
                if isinstance(infos, dict):
                    infos = [infos]
                if not infos:
                    board.push(move)
                    continue

                best_score = _pov_cp(infos[0]["score"], board.turn)
                best_pv = infos[0].get("pv") or []
                best_move = best_pv[0] if best_pv else None
                second_score = second_move = None
                if len(infos) > 1:
                    second_score = _pov_cp(infos[1]["score"], board.turn)
                    second_pv = infos[1].get("pv") or []
                    second_move = second_pv[0] if second_pv else None

                # Snapshot the pre-move state before board.push() mutates it. One
                # move of history is kept: PuzzleNet reads the opponent's last move.
                pre_board = board.copy(stack=1)
                fen_before_move = pre_board.fen()
                phase = get_game_phase(pre_board)

                board.push(move)

                if best_move is not None and move == best_move:
                    after = best_score
                elif second_move is not None and move == second_move:
                    after = second_score
                else:
                    info_after = engine.analyse(board, limit)
                    s = _pov_cp(info_after["score"], board.turn)
                    after = -s if s is not None else None

                if best_score is None or after is None:
                    continue

                cp_loss = best_score - after
                wp_best = win_percent(best_score)
                wp_loss = max(0.0, wp_best - win_percent(after))
                total_cp_loss += min(max(0, cp_loss), MAX_CP_LOSS_FOR_AVG)
                total_wp_loss += wp_loss
                player_moves_n += 1

                # Taking back a piece the opponent just captured is usually the
                # only move, but it is not a tactic -- puzzle generators exclude
                # simple recaptures for the same reason.
                recapture = (best_move is not None and last_capture_sq is not None
                             and best_move.to_square == last_capture_sq
                             and pre_board.is_capture(best_move))
                critical = (second_score is not None and not recapture
                            and wp_best - win_percent(second_score) >= CRITICAL_WP_GAP)
                severity = severity_of(wp_loss)
                if not critical and severity is None:
                    continue

                category = "General"
                if best_move is not None and best_move in pre_board.legal_moves:
                    # Label the engine's whole line, not just its first move, and
                    # trust the engine's mate verdict. label_line uses PuzzleNet when
                    # a model is installed, else the rule-based tagger
                    # (src/puzzles/labeller.py; LABELLER=rules forces the rules).
                    category = label_line(pre_board, best_pv[:PV_PLIES_FOR_TAGGING],
                                          mate=best_score >= MATE_CP)

                if critical:
                    result.opportunities.append(Opportunity(
                        category=category,
                        hit=wp_loss < HIT_WP_TOL,
                        wp_loss=round(wp_loss, 2),
                        phase=phase,
                        ply=ply,
                    ))
                if severity is not None:
                    result.errors.append(MoveError(
                        move_number = pre_board.fullmove_number,
                        fen_before  = fen_before_move,
                        move_uci    = move.uci(),
                        cp_loss     = int(cp_loss),
                        severity    = severity,
                        phase       = phase,
                        category    = category,
                        wp_loss     = round(wp_loss, 2),
                    ))

            if player_moves_n > 0:
                result.avg_cp_loss = total_cp_loss / player_moves_n
                result.avg_wp_loss = total_wp_loss / player_moves_n

    except chess.engine.EngineTerminatedError:
        logger.warning("Stockfish process terminated unexpectedly on game.")
        result.failed = True
    except Exception as e:
        logger.warning("Stockfish analysis error: %s", e)
        result.failed = True

    return result


def analyze_games_parallel(
    pgn_strings: list[str],
    username: str,
    stockfish_path: str,
    *,
    workers: int = 4,
    time_limit: Optional[float] = None,
    nodes: int = ANALYSIS_NODES,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    use_processes: bool = False,
) -> list[GameAnalysis]:
    """
    Analyse a list of PGN strings in parallel.

    Each worker opens its own Stockfish subprocess. With threads (default)
    the Python side — UCI parsing, tactic tagging — shares one GIL;
    use_processes=True gives each worker its own interpreter. Measured on the
    development laptop (16 games, 8 workers, identical output): threads 127 s,
    processes 129 s — the engines, not the GIL, are the bottleneck there, so
    threads stay the default and processes are only an option.

    Parameters
    ----------
    pgn_strings       : list of raw PGN strings (from Chess.com API)
    username          : Chess.com username
    stockfish_path    : path to Stockfish binary
    workers           : number of parallel threads (default 4)
    time_limit        : optional seconds per position (overrides nodes)
    nodes             : deterministic node budget per position
    progress_callback : optional callable(games_done, games_total) for UI updates

    Returns
    -------
    List of GameAnalysis objects in the same order as pgn_strings.
    """
    n = len(pgn_strings)
    results: list[Optional[GameAnalysis]] = [None] * n

    pool = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    with pool(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(
                analyze_game, pgn, username, stockfish_path,
                time_limit=time_limit, nodes=nodes,
            ): i
            for i, pgn in enumerate(pgn_strings)
        }

        done = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error("Worker error on game %d: %s", idx, e)
                results[idx] = GameAnalysis(failed=True)

            done += 1
            if progress_callback:
                progress_callback(done, n)

    # Filter out None slots (shouldn't happen but defensive)
    return [r if r is not None else GameAnalysis(failed=True) for r in results]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pov_cp(score: chess.engine.PovScore, turn: chess.Color) -> Optional[int]:
    """
    Convert a Stockfish PovScore to centipawns from the perspective of `turn`.
    Forced mate is mapped to ±MATE_CP to keep arithmetic sane.
    """
    try:
        pov = score.pov(turn)
        if pov.is_mate():
            m = pov.mate()
            return MATE_CP if m > 0 else -MATE_CP
        return pov.score()
    except Exception:
        return None
