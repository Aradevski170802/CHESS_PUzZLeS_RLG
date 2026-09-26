"""
Targets, splits and loading for the PuzzleNet dataset.

Targets
───────
  category  the puzzle's primary category, resolved from its Lichess themes by
            MOTIF_PRIORITY (resolve_primary_category): the same label the
            labeller harness uses as reference. 24 classes (23 weakness
            categories + General).
  themes    multi-hot over THEMES: every Lichess theme except the ones that say
            where the game came from (master, masterVsMaster, superGM) and the
            length tags (oneMove/short/long/veryLong). The length tags are an exact
            function of the line length, which is an input, so predicting them
            would inflate the scores.
  rating    the puzzle's Lichess Glicko-2 rating, observed with its own rating
            deviation (RD), which the loss treats as known label noise.

Splits
──────
  HARNESS  every puzzle in the HELD-OUT labeller-validation sample (seed 11:
           30,000 uniform + up to 2,000 per class stratified), the sample on which
           the hand-written tagger scored kappa = 0.52. These puzzles never enter
           training or model selection, so the network is scored on exactly the
           puzzles the tagger was scored on. The seed-7 sample was only the tagger's
           development set; holding it out as well would have removed almost every
           training example of the rarest classes (the stratified draw takes up to
           2,000 per class, and En Passant has about 2,100 puzzles in total).
  VAL      2 % of the rest, by a hash of the PuzzleId (model selection,
           temperature scaling)
  TEST     3 % of the rest, by the same hash (every other reported metric)
  TRAIN    the remaining ~95 %
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.data.puzzle_loader import WEAKNESS_CATEGORIES, resolve_primary_category

DATA_DIR = Path("data/neural")
CATEGORIES: list[str] = WEAKNESS_CATEGORIES + ["General"]
CAT_INDEX = {c: i for i, c in enumerate(CATEGORIES)}

_EXCLUDED_THEMES = {"master", "masterVsMaster", "superGM", "oneMove", "short", "long", "veryLong"}
THEMES: list[str] = sorted({
    "advancedPawn", "advantage", "anastasiaMate", "arabianMate", "attackingF2F7", "attraction",
    "backRankMate", "balestraMate", "bishopEndgame", "blindSwineMate", "bodenMate",
    "capturingDefender", "castling", "clearance", "collinearMove", "cornerMate", "crushing",
    "defensiveMove", "deflection", "discoveredAttack", "discoveredCheck", "doubleBishopMate",
    "doubleCheck", "dovetailMate", "enPassant", "endgame", "epauletteMate", "equality",
    "exposedKing", "fork", "hangingPiece", "hookMate", "interference", "intermezzo",
    "killBoxMate", "kingsideAttack", "knightEndgame", "mate", "mateIn1", "mateIn2", "mateIn3",
    "mateIn4", "mateIn5", "middlegame", "morphysMate", "opening", "operaMate", "pawnEndgame",
    "pillsburysMate", "pin", "promotion", "queenEndgame", "queenRookEndgame",
    "queensideAttack", "quietMove", "rookEndgame", "sacrifice", "skewer", "smotheredMate",
    "swallowstailMate", "trappedPiece", "triangleMate", "underPromotion", "vukovicMate",
    "xRayAttack", "zugzwang",
})
THEME_INDEX = {t: i for i, t in enumerate(THEMES)}
N_THEME_BYTES = (len(THEMES) + 7) // 8

TRAIN, VAL, TEST, HARNESS = 0, 1, 2, 3
SPLIT_NAMES = {TRAIN: "train", VAL: "val", TEST: "test", HARNESS: "harness"}
HARNESS_SEEDS = (11,)


def hash_split(puzzle_id: str) -> int:
    h = int(hashlib.md5(puzzle_id.encode()).hexdigest()[:8], 16) % 100
    return VAL if h < 2 else TEST if h < 5 else TRAIN


def category_of(themes: str) -> int:
    return CAT_INDEX[resolve_primary_category(themes)]


def theme_vector(themes: str) -> np.ndarray:
    v = np.zeros(len(THEMES), dtype=np.uint8)
    for t in themes.split():
        i = THEME_INDEX.get(t)
        if i is not None:
            v[i] = 1
    return v


@dataclass
class PuzzleData:
    """The built dataset, loaded into memory (bits stay bit-packed)."""
    bits: np.ndarray        # (N, N_PACKED) uint8, bit-packed binary features
    cont: np.ndarray        # (N, N_CONT) float16
    cat: np.ndarray         # (N,) uint8
    themes: np.ndarray      # (N, N_THEME_BYTES) uint8, bit-packed
    rating: np.ndarray      # (N,) int16
    rd: np.ndarray          # (N,) int16
    nb_plays: np.ndarray    # (N,) int32
    n_moves: np.ndarray     # (N,) uint8, solution length incl. the opponent's first move
    split: np.ndarray       # (N,) uint8
    puzzle_id: np.ndarray   # (N,) str
    info: dict

    def indices(self, split: int) -> np.ndarray:
        return np.flatnonzero(self.split == split)


def load_dataset(data_dir: Path = DATA_DIR, *, mmap: bool = False,
                 with_ids: bool = True) -> PuzzleData:
    """The built dataset. with_ids=False skips the puzzle-id column (179 MB, and
    training never needs it), which matters on a memory-constrained machine."""
    mode = "r" if mmap else None
    meta = np.load(data_dir / "meta.npz", allow_pickle=False)
    return PuzzleData(
        bits=np.load(data_dir / "X_bits.npy", mmap_mode=mode),
        cont=np.load(data_dir / "X_cont.npy", mmap_mode=mode),
        cat=meta["cat"], themes=meta["themes"], rating=meta["rating"], rd=meta["rd"],
        nb_plays=meta["nb_plays"], n_moves=meta["n_moves"], split=meta["split"],
        puzzle_id=meta["puzzle_id"] if with_ids else np.empty(0, dtype="U8"),
        info=json.loads((data_dir / "dataset_info.json").read_text(encoding="utf-8")),
    )


# ── Feature subsets (for the ablation study) ──────────────────────────────────

def feature_columns(subset: str = "all") -> np.ndarray:
    """Column indices of the dense input used by a feature subset.

    all              every feature
    raw              piece-square planes and move blocks only (no hand-built
                     tactical relations, position facts or continuous features)
    engineered       the hand-built tactical, reply, global and continuous
                     features only (no boards, no move blocks)
    """
    from src.neural import encoding as E
    raw_end = E.OFFSETS["tactic_m1"]
    if subset == "all":
        return np.arange(E.N_FEATURES)
    if subset == "raw":
        return np.arange(raw_end)
    if subset == "engineered":
        return np.arange(raw_end, E.N_FEATURES)
    raise ValueError(f"unknown feature subset {subset!r}")


def make_input(bits: np.ndarray, cont: np.ndarray, cont_mean: np.ndarray, cont_std: np.ndarray,
               columns: Optional[np.ndarray] = None, out: Optional[np.ndarray] = None) -> np.ndarray:
    """Dense float32 network input from unpacked bits (0/1) and raw continuous
    features: bits pass through, continuous features are standardised with the
    training-split mean and SD, then the feature subset is selected."""
    from src.neural import encoding as E
    n = bits.shape[0]
    x = out if out is not None and out.shape == (n, E.N_FEATURES) else \
        np.empty((n, E.N_FEATURES), dtype=np.float32)
    x[:, :E.N_BITS] = bits
    np.subtract(cont, cont_mean, out=x[:, E.N_BITS:], casting="unsafe")
    x[:, E.N_BITS:] /= cont_std
    return x if columns is None or len(columns) == E.N_FEATURES else x[:, columns]


class BlockLoader:
    """Mini-batches over a (memory-mapped) dataset with bounded memory.

    Rows are stored in PuzzleId order, which is random with respect to content, so
    reading contiguous blocks and shuffling both the block order and the rows
    inside each block gives well-mixed batches while touching only one block of
    the 2.4 GB feature file at a time.
    """

    def __init__(self, data: PuzzleData, index: np.ndarray, batch_size: int,
                 cont_mean: np.ndarray, cont_std: np.ndarray, rating_mean: float,
                 rating_sd: float, columns: Optional[np.ndarray] = None,
                 block_rows: int = 262_144, seed: int = 0,
                 augment: Optional[PuzzleData] = None, augment_prob: float = 0.0):
        """augment: a second encoding of the SAME puzzles in the same row order (the
        truncated-line build). Each row is then drawn from it with probability
        augment_prob, so the network sees both whole solutions and the cut lines that
        game analysis actually supplies. Targets always come from `data`."""
        from src.neural import encoding as E
        self.d, self.batch_size = data, batch_size
        self.index = np.sort(index)
        self.blocks = [self.index[i:i + block_rows] for i in range(0, len(self.index), block_rows)]
        self.cont_mean, self.cont_std = cont_mean, cont_std
        self.rating_mean, self.rating_sd = rating_mean, rating_sd
        self.columns = columns
        self.rng = np.random.default_rng(seed)
        self.n_bits, self.n_theme = E.N_BITS, len(THEMES)
        self.augment, self.augment_prob = augment, augment_prob
        if augment is not None and len(augment.cat) != len(data.cat):
            raise ValueError("the augmented dataset must have the same rows as the main one")

    def __len__(self) -> int:
        return int(np.ceil(len(self.index) / self.batch_size))

    def batch(self, idx: np.ndarray, packed: np.ndarray, cont: np.ndarray):
        from src.neural.network import Batch
        bits = np.unpackbits(packed, axis=1, count=self.n_bits)
        x = make_input(bits, cont, self.cont_mean, self.cont_std, self.columns)
        themes = np.unpackbits(self.d.themes[idx], axis=1, count=self.n_theme).astype(np.float32)
        rating = (self.d.rating[idx].astype(np.float32) - self.rating_mean) / self.rating_sd
        rho2 = (self.d.rd[idx].astype(np.float32) / self.rating_sd) ** 2
        return Batch(x=x, cat=self.d.cat[idx].astype(np.int64), theme=themes,
                     rating=rating, rho2=rho2)

    @staticmethod
    def _read_from(data: PuzzleData, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Features of sorted `rows`: one contiguous read when the rows are dense
        in their range (the training split), row-by-row otherwise (val/test are
        spread over the whole file, and a contiguous read would load all of it)."""
        lo, hi = int(rows[0]), int(rows[-1]) + 1
        if hi - lo <= 2 * len(rows):
            return (np.asarray(data.bits[lo:hi])[rows - lo],
                    np.asarray(data.cont[lo:hi])[rows - lo])
        return np.asarray(data.bits[rows]), np.asarray(data.cont[rows])

    def _read(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        packed, cont = self._read_from(self.d, rows)
        if self.augment is not None and self.augment_prob > 0:
            take = self.rng.random(len(rows)) < self.augment_prob
            if take.any():
                a_packed, a_cont = self._read_from(self.augment, rows)
                packed = packed.copy()
                cont = cont.copy()
                packed[take] = a_packed[take]
                cont[take] = a_cont[take]
        return packed, cont

    def epoch(self):
        for b in self.rng.permutation(len(self.blocks)):
            rows = self.blocks[b]
            packed, cont = self._read(rows)
            order = self.rng.permutation(len(rows))
            for i in range(0, len(rows), self.batch_size):
                sel = order[i:i + self.batch_size]
                yield self.batch(rows[sel], packed[sel], cont[sel])

    def all_batches(self, batch_size: Optional[int] = None):
        """Every row once, in storage order (evaluation)."""
        bs = batch_size or self.batch_size
        for i in range(0, len(self.index), bs):
            rows = self.index[i:i + bs]
            packed, cont = self._read(rows)
            yield rows, self.batch(rows, packed, cont)
