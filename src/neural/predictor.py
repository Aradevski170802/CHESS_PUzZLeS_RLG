"""
The trained PuzzleNet as the rest of the system uses it.

    from src.neural.predictor import get_predictor
    net = get_predictor()          # None when no model is installed or it is disabled
    p = net.predict_line(board, line, mate=None)
    p.category, p.confidence, p.rating, p.rating_sd, p.themes

Where it is used
────────────────
  * stockfish_analyzer.analyze_game labels each critical position's best line
    (LABELLER=neural, the default whenever a model is installed; LABELLER=rules keeps
    the hand-written tactic_tagger)
  * generator._extract_stockfish gives a mined puzzle its category, its rating and
    a per-puzzle rating deviation, replacing the fixed-formula _estimate_rating
  * app.session_puzzle passes that rating deviation to the IRT learner, so an
    uncertain difficulty estimate moves the player's ratings less
    (irt_model: kappa = 1/sqrt(1 + 3 var_b / pi^2))

Model file: src/data/models/puzzlenet.npz (float16), written by
scripts/neural/train_puzzlenet.py --export. PUZZLENET_MODEL overrides the path;
PUZZLENET=off disables the network everywhere.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import chess
import numpy as np

from src.neural import encoding as E
from src.neural.dataset import feature_columns, make_input
from src.neural.network import PuzzleNet

logger = logging.getLogger(__name__)

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "data" / "models" / "puzzlenet.npz"
THEME_THRESHOLD = 0.5


@dataclass
class LinePrediction:
    category: str
    confidence: float                 # calibrated probability of `category`
    probs: dict[str, float]           # every category, calibrated
    rating: float                     # predicted Lichess-scale difficulty
    rating_sd: float                  # uncertainty about that difficulty (content only)
    themes: dict[str, float]          # themes with probability >= THEME_THRESHOLD


class PuzzleNetPredictor:
    def __init__(self, net: PuzzleNet, path: Optional[Path] = None):
        meta = net.meta
        if meta.get("encoding_version") != E.ENCODING_VERSION:
            raise ValueError(f"model encoding v{meta.get('encoding_version')} "
                             f"!= code v{E.ENCODING_VERSION}")
        self.net = net
        self.path = path
        self.categories: list[str] = meta["categories"]
        self.themes: list[str] = meta["themes"]
        self.cont_mean = np.asarray(meta["cont_mean"], dtype=np.float32)
        self.cont_std = np.asarray(meta["cont_std"], dtype=np.float32)
        self.rating_mean = float(meta["rating_mean"])
        self.rating_sd = float(meta["rating_sd"])
        self.columns = feature_columns(meta.get("features", "all"))

    @classmethod
    def load(cls, path: Path) -> "PuzzleNetPredictor":
        return cls(PuzzleNet.load(path), path)

    # ── Batch interface ────────────────────────────────────────────────────
    def predict_encoded(self, bits: np.ndarray, cont: np.ndarray) -> dict[str, np.ndarray]:
        """Raw arrays for pre-encoded examples: bits (n, N_BITS) 0/1, cont (n, N_CONT)."""
        x = make_input(bits, cont, self.cont_mean, self.cont_std, self.columns)
        out = self.net.predict(x)
        out["rating"] = out["rating_mean"] * self.rating_sd + self.rating_mean
        out["rating_sd_points"] = out["rating_sd"] * self.rating_sd
        return out

    def predict_many(self, examples: Sequence[tuple[chess.Board, Sequence[chess.Move], Optional[bool]]]
                     ) -> list[LinePrediction]:
        if not examples:
            return []
        enc = [E.encode_line(b, line, mate=mate) for b, line, mate in examples]
        out = self.predict_encoded(np.stack([e[0] for e in enc]), np.stack([e[1] for e in enc]))
        preds = []
        for i in range(len(examples)):
            probs = out["cat_probs"][i]
            k = int(probs.argmax())
            preds.append(LinePrediction(
                category=self.categories[k],
                confidence=float(probs[k]),
                probs={c: float(p) for c, p in zip(self.categories, probs)},
                rating=float(out["rating"][i]),
                rating_sd=float(out["rating_sd_points"][i]),
                themes={t: round(float(p), 3) for t, p in zip(self.themes, out["theme_probs"][i])
                        if p >= THEME_THRESHOLD},
            ))
        return preds

    # ── Single-example conveniences ────────────────────────────────────────
    def predict_line(self, board: chess.Board, line: Sequence[chess.Move], *,
                     mate: Optional[bool] = None) -> LinePrediction:
        """`board` has the solver to move; the opponent's previous move is read from
        its move stack when present."""
        return self.predict_many([(board, line, mate)])[0]

    def predict_puzzle(self, fen: str, moves: str | Sequence[str]) -> LinePrediction:
        """A puzzle in Lichess convention: FEN before the opponent's move,
        Moves = [opponent move, solution ...]."""
        ms = moves.split() if isinstance(moves, str) else list(moves)
        board = chess.Board(fen)
        board.push_uci(ms[0])
        return self.predict_line(board, [chess.Move.from_uci(u) for u in ms[1:]])


_lock = threading.Lock()
_cached: dict[str, Optional[PuzzleNetPredictor]] = {}


def get_predictor() -> Optional[PuzzleNetPredictor]:
    """The installed model, loaded once per process; None when it is missing,
    disabled (PUZZLENET=off) or fails to load (the caller then uses the rule-based
    tagger and heuristic rating)."""
    if os.environ.get("PUZZLENET", "on").lower() in ("off", "0", "false", "no"):
        return None
    path = Path(os.environ.get("PUZZLENET_MODEL", DEFAULT_MODEL))
    key = str(path)
    with _lock:
        if key not in _cached:
            predictor = None
            if path.exists():
                try:
                    predictor = PuzzleNetPredictor.load(path)
                    logger.info("PuzzleNet loaded from %s (%s parameters)", path,
                                f"{predictor.net.n_parameters():,}")
                except Exception as exc:          # a broken model must not break analysis
                    logger.warning("PuzzleNet at %s could not be loaded: %s", path, exc)
            _cached[key] = predictor
        return _cached[key]


def reset_cache() -> None:
    with _lock:
        _cached.clear()
