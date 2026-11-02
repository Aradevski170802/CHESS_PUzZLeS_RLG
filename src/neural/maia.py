"""
Maia: a model of human moves, run through the Leela Chess Zero engine (lc0).

Maia (McIlroy-Young et al., KDD 2020) is a set of nine policy networks, each trained
on Lichess games between players of one rating level (1100, 1200, ..., 1900). Given a
position, the network for level R gives every legal move a probability: how likely a
player of rating R is to play it. That is the question this project keeps asking of a
puzzle (will a player of this rating find the move?), answered by a model trained on
millions of human games rather than on this project's own data.

lc0 evaluates the network with a one-node search (`go nodes 1`), which expands only the
root, and VerboseMoveStats reports each move's prior P. PolicyTemperature is set to 1.0
so that P is the network's own softmax; lc0's default (1.359) flattens it for search.
lc0 prints P to 0.01 %, so a probability it shows as 0.00 % is floored at P_FLOOR.

Files (downloaded, not in git; see .gitignore):
    lc0/lc0.exe                   lc0 v0.32.1, CPU (DNNL) build
    lc0/maia/maia-<R>.pb.gz       Maia v1.0 weights, R = 1100 ... 1900
LC0_PATH and MAIA_DIR override the locations.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterable, Optional

import chess
import chess.engine

ROOT = Path(__file__).resolve().parents[2]
MAIA_LEVELS = tuple(range(1100, 2000, 100))
P_FLOOR = 5e-5

_STAT = re.compile(r"^(\S+)\s+\(\s*\d+\s*\)\s+N:.*?\(P:\s*([\d.]+)%\)")
# lc0 may print castling as king-takes-rook; map it to standard UCI.
_CASTLING = {"e1h1": "e1g1", "e1a1": "e1c1", "e8h8": "e8g8", "e8a8": "e8c8"}


def lc0_path() -> Optional[Path]:
    path = Path(os.environ.get("LC0_PATH", ROOT / "lc0" / "lc0.exe"))
    return path if path.exists() else None


def weights_path(level: int) -> Optional[Path]:
    path = Path(os.environ.get("MAIA_DIR", ROOT / "lc0" / "maia")) / f"maia-{level}.pb.gz"
    return path if path.exists() else None


def available() -> bool:
    return lc0_path() is not None and all(weights_path(lv) for lv in MAIA_LEVELS)


def nearest_level(rating: float) -> int:
    """The Maia level closest to a rating. Players below 1100 or above 1900 get the
    end levels, which is the model's main limitation for this project's cohort."""
    return min(MAIA_LEVELS, key=lambda lv: (abs(lv - rating), lv))


def parse_move_stats(lines: Iterable[str], board: chess.Board) -> dict[str, float]:
    """Move -> probability from lc0's VerboseMoveStats lines. The closing 'node'
    summary is skipped, and castling printed as king-takes-rook becomes standard UCI."""
    probs: dict[str, float] = {}
    for line in lines:
        match = _STAT.match(line)
        if not match or match.group(1) == "node":
            continue
        uci = match.group(1)
        if uci in _CASTLING and board.is_castling(chess.Move.from_uci(uci)):
            uci = _CASTLING[uci]
        probs[uci] = float(match.group(2)) / 100
    return probs


class Maia:
    """One lc0 process running the Maia network of one rating level."""

    def __init__(self, level: int):
        engine, weights = lc0_path(), weights_path(level)
        if engine is None or weights is None:
            raise FileNotFoundError(f"lc0 or the Maia-{level} weights are missing "
                                    "(see src/neural/maia.py)")
        logging.getLogger("chess.engine").setLevel(logging.ERROR)   # lc0's banner
        self.level = level
        self.engine = chess.engine.SimpleEngine.popen_uci([str(engine), f"--weights={weights}"])
        self.engine.configure({"VerboseMoveStats": True, "PolicyTemperature": 1.0,
                               "Threads": 1})

    def move_probabilities(self, board: chess.Board) -> dict[str, float]:
        """Probability of every legal move for a player of this level (sums to ~1)."""
        with self.engine.analysis(board, chess.engine.Limit(nodes=1)) as analysis:
            lines = [info["string"] for info in analysis if "string" in info]
        return parse_move_stats(lines, board)

    def probability(self, board: chess.Board, move: chess.Move) -> float:
        """How likely a player of this level is to play `move` here."""
        return max(P_FLOOR, self.move_probabilities(board).get(move.uci(), 0.0))

    def close(self) -> None:
        self.engine.quit()

    def __enter__(self) -> "Maia":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
