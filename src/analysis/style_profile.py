"""
Player style profile — derives tactical and stylistic metrics from a
player's game history without requiring a chess engine.

Metrics
-------
archetype        : string label (Tactical Attacker / Solid Positional / etc.)
win_rate         : overall win % across all provided games
white_win_rate   : win % when playing White
black_win_rate   : win % when playing Black
win_pattern      : how the player wins (checkmate / resignation / timeout %)
loss_pattern     : how the player loses
opening_breadth  : count of distinct ECO families in repertoire
avg_accuracy     : mean Chess.com accuracy score (if present in game data)
time_pressure    : fraction of moves played with < 30 s on clock
aggression_score : 0–100 proxy; high = plays sharp/sacrificial openings + mates
"""
from __future__ import annotations

import io
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import chess
import chess.pgn


# ── Regex for clock annotation ─────────────────────────────────────────────
_CLK_RE = re.compile(r"\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]")

OPENING_FAMILIES = {
    "A": "Flank / Indian",
    "B": "Sicilian / Caro-Kann",
    "C": "Open / Semi-Open",
    "D": "Closed / Semi-Closed",
    "E": "Indian Defences",
}


def compute_style_profile(
    username: str,
    pgn_strings: list[str],
    *,
    accuracy_data: Optional[list[Optional[float]]] = None,
) -> dict:
    """
    Compute a player style profile from a list of PGN strings.

    Parameters
    ----------
    username       : Chess.com username (case-insensitive)
    pgn_strings    : raw PGN strings from game history
    accuracy_data  : optional list of Chess.com accuracy scores (same order
                     as pgn_strings, None entries are skipped)

    Returns
    -------
    dict with all style metrics (safe to serialise to JSON)
    """
    if not pgn_strings:
        return _empty_profile()

    user = username.lower()

    # Running counters
    white_wins = white_losses = white_draws = white_total = 0
    black_wins = black_losses = black_draws = black_total = 0

    wins  = defaultdict(int)   # how player wins
    losses = defaultdict(int)  # how player loses

    eco_letters: set[str] = set()
    opening_names: set[str] = set()

    total_moves      = 0
    low_clock_moves  = 0
    sacrifice_count  = 0  # moves where player captures with a higher-value piece
    total_parsed     = 0

    acc_scores: list[float] = []

    for idx, pgn_str in enumerate(pgn_strings):
        game = chess.pgn.read_game(io.StringIO(pgn_str))
        if game is None:
            continue
        total_parsed += 1

        headers = game.headers
        is_white = user in headers.get("White", "").lower()
        result   = headers.get("Result", "*")
        term     = headers.get("Termination", "").lower()

        # ── Colour stats ─────────────────────────────────────────────────
        if is_white:
            white_total += 1
            if result == "1-0":
                white_wins += 1
                _tally_win(term, wins)
            elif result == "0-1":
                white_losses += 1
                _tally_loss(term, losses)
            else:
                white_draws += 1
        else:
            black_total += 1
            if result == "0-1":
                black_wins += 1
                _tally_win(term, wins)
            elif result == "1-0":
                black_losses += 1
                _tally_loss(term, losses)
            else:
                black_draws += 1

        # ── Opening ──────────────────────────────────────────────────────
        eco = headers.get("ECO", "")
        if eco:
            eco_letters.add(eco[0])

        eco_url = headers.get("ECOUrl", "")
        if eco_url:
            parts = eco_url.rstrip("/").split("/")
            if parts:
                name = parts[-1].split("-")[0].capitalize()
                if name:
                    opening_names.add(name)

        # ── Per-move stats (clock + sacrifices) ──────────────────────────
        board = game.board()
        player_side = chess.WHITE if is_white else chess.BLACK

        for node in game.mainline():
            if board.turn == player_side:
                total_moves += 1

                # Clock
                comment = node.comment or ""
                m = _CLK_RE.search(comment)
                if m:
                    h, mn, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
                    secs = h * 3600 + mn * 60 + s
                    if secs < 30:
                        low_clock_moves += 1

                # Sacrifice: player uses a piece worth ≥400cp more than what they capture
                # (e.g. rook takes pawn, queen takes bishop). Filters out normal knight-takes-pawn.
                mv = node.move
                target = board.piece_at(mv.to_square)
                attacker = board.piece_at(mv.from_square)
                if (target and attacker
                        and target.color != attacker.color
                        and _piece_val(attacker.piece_type) > _piece_val(target.piece_type) + 400):
                    sacrifice_count += 1

            try:
                board.push(node.move)
            except Exception:
                break

        # ── Chess.com accuracy ────────────────────────────────────────────
        if accuracy_data and idx < len(accuracy_data):
            a = accuracy_data[idx]
            if a is not None:
                acc_scores.append(float(a))

    if total_parsed == 0:
        return _empty_profile()

    total_games = white_total + black_total
    total_wins  = sum(wins.values())
    total_losses = sum(losses.values())
    total_draws = white_draws + black_draws

    win_rate  = total_wins  / total_games if total_games else 0
    draw_rate = total_draws / total_games if total_games else 0
    loss_rate = total_losses / total_games if total_games else 0

    # ── Derived metrics ───────────────────────────────────────────────────
    time_pressure = low_clock_moves / total_moves if total_moves else 0
    sacrifice_rate = sacrifice_count / total_games if total_games else 0

    checkmate_win_pct = wins["checkmate"] / total_wins if total_wins else 0
    resignation_loss_pct = losses["resignation"] / total_losses if total_losses else 0

    # Aggression: weighted from checkmate tendency + sacrifice frequency
    # sacrifice_rate is per-game; typical range 0–0.5; cap contribution at 35 pts
    aggression_score = min(100, int(
        checkmate_win_pct * 55 +                 # 0–55: checkmate-win percentage
        min(sacrifice_rate * 70, 35) +           # 0–35: sacrifice frequency (capped)
        (1 - resignation_loss_pct) * 10          # 0–10: don't concede early
    ))

    # Opening breadth
    opening_breadth = len(eco_letters)
    top_openings = list(opening_names)[:5]

    avg_accuracy = round(sum(acc_scores) / len(acc_scores), 1) if acc_scores else None

    # ── Archetype ────────────────────────────────────────────────────────
    archetype, archetype_desc = _classify_archetype(
        checkmate_win_pct, aggression_score, time_pressure,
        win_rate, opening_breadth, sacrifice_rate
    )

    return {
        "archetype":         archetype,
        "archetype_desc":    archetype_desc,
        "total_games":       total_parsed,
        "win_rate":          round(win_rate  * 100, 1),
        "draw_rate":         round(draw_rate * 100, 1),
        "loss_rate":         round(loss_rate * 100, 1),
        "white_win_rate":    round(white_wins  / white_total * 100, 1) if white_total else 0,
        "black_win_rate":    round(black_wins  / black_total * 100, 1) if black_total else 0,
        "white_games":       white_total,
        "black_games":       black_total,
        "win_pattern": {
            "checkmate":   _pct(wins["checkmate"],   total_wins),
            "resignation": _pct(wins["resignation"],  total_wins),
            "timeout":     _pct(wins["timeout"],      total_wins),
        },
        "loss_pattern": {
            "checkmate":   _pct(losses["checkmate"],   total_losses),
            "resignation": _pct(losses["resignation"],  total_losses),
            "timeout":     _pct(losses["timeout"],      total_losses),
        },
        "opening_breadth":  opening_breadth,
        "top_openings":     top_openings,
        "avg_accuracy":     avg_accuracy,
        "time_pressure":    round(time_pressure * 100, 1),
        "aggression_score": aggression_score,
        "sacrifice_rate":   round(sacrifice_rate, 3),
        "eco_families":     [OPENING_FAMILIES.get(l, l) for l in sorted(eco_letters)],
    }


# ── Helpers ───────────────────────────────────────────────────────────────

def _piece_val(pt: chess.PieceType) -> int:
    return {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}.get(pt, 0)


def _tally_win(termination: str, wins: defaultdict) -> None:
    if "checkmate" in termination:
        wins["checkmate"] += 1
    elif "resign" in termination:
        wins["resignation"] += 1
    elif "time" in termination:
        wins["timeout"] += 1
    else:
        wins["other"] += 1


def _tally_loss(termination: str, losses: defaultdict) -> None:
    if "checkmate" in termination:
        losses["checkmate"] += 1
    elif "resign" in termination:
        losses["resignation"] += 1
    elif "time" in termination:
        losses["timeout"] += 1
    else:
        losses["other"] += 1


def _pct(part: int, total: int) -> int:
    return round(part / total * 100) if total else 0


def _classify_archetype(
    checkmate_win_pct: float,
    aggression_score:  int,
    time_pressure:     float,
    win_rate:          float,
    opening_breadth:   int,
    sacrifice_rate:    float,
) -> tuple[str, str]:
    if checkmate_win_pct >= 0.35 or aggression_score >= 65:
        return (
            "Tactical Attacker",
            "You play sharp, aggressive chess — always hunting for the checkmate.",
        )
    if time_pressure >= 0.15:
        return (
            "Time-Trouble Gambler",
            "You favour complex positions but risk running short on clock.",
        )
    if win_rate >= 0.55 and opening_breadth >= 3:
        return (
            "Opening Theorist",
            "Broad repertoire and solid results — you win in the preparation.",
        )
    if win_rate >= 0.5 and sacrifice_rate < 0.03:
        return (
            "Positional Grinder",
            "Patient and methodical — you outplay opponents move by move.",
        )
    return (
        "Balanced Player",
        "Versatile style — comfortable in tactical and positional positions alike.",
    )


def _empty_profile() -> dict:
    return {
        "archetype": "Unknown", "archetype_desc": "Not enough games analysed.",
        "total_games": 0, "win_rate": 0, "draw_rate": 0, "loss_rate": 0,
        "white_win_rate": 0, "black_win_rate": 0,
        "white_games": 0, "black_games": 0,
        "win_pattern": {"checkmate": 0, "resignation": 0, "timeout": 0},
        "loss_pattern": {"checkmate": 0, "resignation": 0, "timeout": 0},
        "opening_breadth": 0, "top_openings": [], "avg_accuracy": None,
        "time_pressure": 0, "aggression_score": 0, "sacrifice_rate": 0,
        "eco_families": [],
    }
