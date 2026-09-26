"""
Which tactic labeller the system uses.

    neural   PuzzleNet (src/neural), the default whenever a trained model is installed
    rules    the hand-written line tagger (src/puzzles/tactic_tagger.py)

LABELLER=rules forces the rule-based tagger. So does PUZZLENET=off, or a missing
model file. Both labellers take the same input: a position with the solver to move
and the principal line that follows it.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import chess

from src.puzzles.tactic_tagger import tag_line

# How many plies of an engine line PuzzleNet is shown. Lichess solutions stop when
# the tactic is complete, but an engine principal variation runs on into the won
# position that follows, and those trailing moves mislead the network.
# scripts/neural/engine_pv_check.py measures the effect on held-out puzzles:
# kappa 0.60 (1 ply), 0.77 (3), 0.78 (5), 0.73 (7). The rule-based tagger keeps the
# full line it was validated on, because it restricts motifs by depth itself.
NEURAL_PV_PLIES = 5


def active_labeller() -> str:
    if os.environ.get("LABELLER", "neural").lower() == "rules":
        return "rules"
    from src.neural.predictor import get_predictor
    return "neural" if get_predictor() is not None else "rules"


def label_line(board: chess.Board, line: Sequence[chess.Move], *,
               mate: Optional[bool] = None) -> str:
    """Tactical category of `line` played from `board`."""
    if active_labeller() == "neural":
        from src.neural.predictor import get_predictor
        return get_predictor().predict_line(board, list(line)[:NEURAL_PV_PLIES], mate=mate).category
    return tag_line(board, line, mate=mate)
