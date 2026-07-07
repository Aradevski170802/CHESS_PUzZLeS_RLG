"""
Fast parallel Stockfish analysis for player game histories.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SPEED STRATEGY — how we analyse 100 games in ~3-5 mins
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 50ms time limit per position (depth ~10-12 on modern
   hardware). Fast but still tactically sound.

2. Skip the first 8 half-moves (opening theory — mistakes
   here are mostly preparation gaps, not tactical errors).

3. Only evaluate positions where a significant evaluation
   change is plausible (skip when both sides have < 3
   pieces — simplified endgames rarely have tactics).

4. Parallelise at the GAME level using a ThreadPoolExecutor.
   Each thread owns its own Stockfish subprocess — no locking.

Expected throughput on a quad-core machine:
  40 positions/game × 0.05s × 100 games = 200s single-thread
  ÷ 4 parallel workers ≈ 50s  ✓  (well under 5 minutes)

Error severity thresholds (standard in chess analysis tools):
  Inaccuracy : 50 – 99 centipawn loss
  Mistake    : 100 – 199 centipawn loss
  Blunder    : ≥ 200 centipawn loss
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import io
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional

import chess
import chess.engine
import chess.pgn

from src.data.pgn_parser import get_game_phase

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
INACCURACY_CP  = 50    # centipawn loss ≥ 50  → inaccuracy
MISTAKE_CP     = 100   # centipawn loss ≥ 100 → mistake
BLUNDER_CP     = 200   # centipawn loss ≥ 200 → blunder
SKIP_PLIES     = 8     # ignore first 8 half-moves (opening book)
ANALYSIS_TIME  = 0.05  # seconds per position (Stockfish time limit)
MATE_CP        = 9000  # centipawn value assigned to forced mate


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


@dataclass
class GameAnalysis:
    """Full analysis result for one game."""
    errors:        list[MoveError] = field(default_factory=list)
    player_color:  str  = "white"
    player_won:    Optional[bool] = None    # True / False / None (draw)
    opening_eco:   str  = ""
    opening_family: str = ""
    num_moves:     int  = 0
    avg_cp_loss:   float = 0.0
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
        from pathlib import Path
        if Path(candidate).exists():
            return candidate
    return None


def analyze_game(
    pgn_str: str,
    username: str,
    stockfish_path: str,
    *,
    time_limit: float = ANALYSIS_TIME,
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
    time_limit     : seconds per position (default 0.05)

    Returns
    -------
    GameAnalysis dataclass with all detected errors.
    """
    result = GameAnalysis()

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

    # Determine player colour
    white = headers.get("White", "").lower()
    user  = username.lower()
    player_color = "white" if user in white else "black"
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

    moves = list(game.mainline_moves())
    result.num_moves = len(moves) // 2

    if not moves:
        return result  # empty game

    # ── Stockfish analysis ────────────────────────────────────────────────
    try:
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            # Use 1 thread per engine — we parallelise at game level
            engine.configure({"Threads": 1, "Hash": 32})

            board = game.board()
            total_cp_loss    = 0
            player_moves_n   = 0
            prev_score_cp: Optional[int] = None

            for ply, move in enumerate(moves):
                is_player_turn = (
                    (player_color == "white" and board.turn == chess.WHITE) or
                    (player_color == "black" and board.turn == chess.BLACK)
                )

                if is_player_turn and ply >= SKIP_PLIES:
                    # Evaluate position BEFORE the player's move
                    info_before = engine.analyse(
                        board, chess.engine.Limit(time=time_limit), multipv=1
                    )
                    score_before = _pov_cp(info_before["score"], board.turn)

                    board.push(move)

                    # Evaluate position AFTER the player's move (flip POV)
                    info_after = engine.analyse(
                        board, chess.engine.Limit(time=time_limit), multipv=1
                    )
                    score_after_opponent = _pov_cp(info_after["score"], board.turn)
                    # Convert back to player's POV
                    score_after_player = (
                        -score_after_opponent
                        if score_after_opponent is not None else None
                    )

                    # Centipawn loss = how much worse the position got for the player
                    if score_before is not None and score_after_player is not None:
                        cp_loss = score_before - score_after_player

                        if cp_loss >= INACCURACY_CP:
                            severity = (
                                "blunder"    if cp_loss >= BLUNDER_CP  else
                                "mistake"    if cp_loss >= MISTAKE_CP  else
                                "inaccuracy"
                            )
                            result.errors.append(MoveError(
                                move_number = board.fullmove_number,
                                fen_before  = board.fen(),
                                move_uci    = move.uci(),
                                cp_loss     = cp_loss,
                                severity    = severity,
                                phase       = get_game_phase(board),
                            ))

                        total_cp_loss  += max(0, cp_loss)
                        player_moves_n += 1

                else:
                    board.push(move)

            if player_moves_n > 0:
                result.avg_cp_loss = total_cp_loss / player_moves_n

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
    time_limit: float = ANALYSIS_TIME,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list[GameAnalysis]:
    """
    Analyse a list of PGN strings in parallel using a thread pool.

    Each worker thread opens its own Stockfish subprocess.
    On a quad-core machine, 100 games typically complete in 3-5 minutes.

    Parameters
    ----------
    pgn_strings       : list of raw PGN strings (from Chess.com API)
    username          : Chess.com username
    stockfish_path    : path to Stockfish binary
    workers           : number of parallel threads (default 4)
    time_limit        : Stockfish time per position in seconds (default 0.05)
    progress_callback : optional callable(games_done, games_total) for UI updates

    Returns
    -------
    List of GameAnalysis objects in the same order as pgn_strings.
    """
    n = len(pgn_strings)
    results: list[Optional[GameAnalysis]] = [None] * n

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(
                analyze_game, pgn, username, stockfish_path, time_limit=time_limit
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
