"""
PGN parser for Chess.com game records.

Chess.com includes ECO opening codes and names directly in PGN headers,
so we extract opening information without a separate opening book lookup.

Example PGN headers from Chess.com:
    [ECO "C54"]
    [ECOUrl "https://www.chess.com/openings/Italian-Game-Classical-Variation"]
    [TimeControl "600"]
    [WhiteElo "1450"]
    [BlackElo "1523"]
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import chess
import chess.pgn

logger = logging.getLogger(__name__)

# Piece count thresholds for game phase classification
# 32 pieces = full board, counting down as pieces are captured
_OPENING_THRESHOLD    = 28   # ≥ 28 pieces on board → opening
_MIDDLEGAME_THRESHOLD = 14   # 14–27 pieces → middlegame
# < 14 pieces → endgame


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def parse_game(pgn_str: str, username: str) -> Optional[dict]:
    """
    Parse a single PGN string from Chess.com into a structured dict.

    Returns
    -------
    {
        "pgn_game":        chess.pgn.Game,   # for Stockfish analysis
        "opening": {
            "eco":         str,   # e.g. "C54"
            "name":        str,   # e.g. "Italian Game Classical Variation"
            "family":      str,   # e.g. "Italian Game"
            "url":         str,
        },
        "player_color":    "white" | "black",
        "player_won":      True | False | None,   # None = draw
        "player_rating":   int,
        "opponent_rating": int,
        "time_control":    str,   # e.g. "600" (seconds)
        "termination":     str,   # e.g. "magnus_c won by checkmate"
        "num_moves":       int,   # full move count
    }

    Returns None if the player is not found in the game or PGN is malformed.
    """
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_str))
    except Exception as e:
        logger.debug("PGN parse error: %s", e)
        return None

    if game is None:
        return None

    headers = game.headers
    white = headers.get("White", "").lower()
    black = headers.get("Black", "").lower()
    user  = username.lower()

    if user in white:
        player_color    = "white"
        player_rating   = _safe_int(headers.get("WhiteElo"))
        opponent_rating = _safe_int(headers.get("BlackElo"))
        result          = headers.get("Result", "*")
        player_won      = True if result == "1-0" else (False if result == "0-1" else None)
    elif user in black:
        player_color    = "black"
        player_rating   = _safe_int(headers.get("BlackElo"))
        opponent_rating = _safe_int(headers.get("WhiteElo"))
        result          = headers.get("Result", "*")
        player_won      = True if result == "0-1" else (False if result == "1-0" else None)
    else:
        logger.debug("Username %s not found in game headers", username)
        return None

    opening = _extract_opening(headers)
    moves   = list(game.mainline_moves())

    return {
        "pgn_game":        game,
        "opening":         opening,
        "player_color":    player_color,
        "player_won":      player_won,
        "player_rating":   player_rating,
        "opponent_rating": opponent_rating,
        "time_control":    headers.get("TimeControl", ""),
        "termination":     headers.get("Termination", ""),
        "num_moves":       len(moves) // 2,
    }


def get_game_phase(board: chess.Board) -> str:
    """
    Classify the current position as 'opening', 'middlegame', or 'endgame'
    based on the number of pieces remaining on the board.
    """
    piece_count = len(board.piece_map())
    if piece_count >= _OPENING_THRESHOLD:
        return "opening"
    elif piece_count >= _MIDDLEGAME_THRESHOLD:
        return "middlegame"
    else:
        return "endgame"


def parse_games_bulk(pgn_strings: list[str], username: str) -> list[Optional[dict]]:
    """Parse a list of PGN strings and return structured dicts (None for failures)."""
    results = []
    for pgn in pgn_strings:
        results.append(parse_game(pgn, username))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_opening(headers: chess.pgn.Headers) -> dict:
    """
    Extract opening info from Chess.com PGN headers.

    Chess.com always includes ECO and ECOUrl when the opening is known.
    ECOUrl format: https://www.chess.com/openings/Italian-Game-Classical-Variation
    We parse the family name from the URL path segment.
    """
    eco     = headers.get("ECO", "")
    eco_url = headers.get("ECOUrl", "")

    if eco_url:
        # e.g. "Italian-Game-Classical-Variation"  →  "Italian Game Classical Variation"
        slug        = eco_url.rstrip("/").split("/")[-1]
        name        = slug.replace("-", " ")
        family      = _extract_family(name)
    else:
        # Fall back to the Opening tag (less common in Chess.com PGNs)
        name   = headers.get("Opening", "Unknown")
        family = _extract_family(name)

    return {
        "eco":    eco,
        "name":   name,
        "family": family,
        "url":    eco_url,
    }


_TWO_WORD_FAMILIES = {
    "Italian Game", "Ruy Lopez", "Sicilian Defense", "French Defense",
    "Caro Kann", "Kings Indian", "Queens Indian", "Queens Gambit",
    "Kings Gambit", "English Opening", "Nimzo Indian", "Dutch Defense",
    "Pirc Defense", "Modern Defense", "Grunfeld Defense", "Benoni Defense",
    "London System", "Catalan Opening", "Scotch Game", "Vienna Game",
    "Four Knights", "Three Knights", "Two Knights", "Giuoco Piano",
    "King Pawn", "Queens Pawn", "Bogo Indian", "Old Indian",
}


def _extract_family(opening_name: str) -> str:
    """
    Return the opening family (first distinctive 1-3 words).

    Examples:
        "Italian Game Classical Variation" → "Italian Game"
        "Sicilian Defense Najdorf Variation" → "Sicilian Defense"
        "King Indian Attack" → "King Indian"
    """
    lower = opening_name.lower()
    for family in sorted(_TWO_WORD_FAMILIES, key=len, reverse=True):
        if lower.startswith(family.lower()):
            return family

    words = opening_name.split()
    return " ".join(words[:2]) if len(words) >= 2 else opening_name


def _safe_int(value: Optional[str], default: int = 1200) -> int:
    try:
        return int(value or default)
    except (ValueError, TypeError):
        return default
