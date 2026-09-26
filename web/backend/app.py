"""
Flask micro-service — serves the puzzle frontend and all APIs.

Routes
------
GET  /                              → index.html
GET  /api/stats                     → pool stats
GET  /api/puzzle/random             → random puzzle (legacy / guest mode)
GET  /api/puzzle/<id>               → puzzle by ID
GET  /api/player/lookup             → Chess.com profile card (fast, no analysis)
POST /api/analysis/start            → launch background game-analysis thread
GET  /api/analysis/status/<user>    → poll analysis progress
POST /api/session/start             → create Thompson-Sampling bandit for user
GET  /api/session/puzzle            → adaptive puzzle (bandit-selected, rating-matched, never repeats)
GET  /api/eval                      → Stockfish evaluation for a FEN position (eval bar)
POST /api/session/result            → record solve/fail/skip, update bandit, persist to disk
GET  /api/session/seen              → puzzle IDs already finished by this user
DEL  /api/session/seen              → forget played puzzles so the pool can be replayed
GET  /api/session/stats             → session accuracy, streak, weakness map
GET  /api/session/model/<user>      → per-category skill ratings ± RD (IRT learner)
GET  /api/user/stats/<user>         → all-time stats from persisted history (auth required)
POST /api/auth/challenge            → issue Chess.com ownership verification code
POST /api/auth/verify-chess         → check Chess.com location field contains code
POST /api/auth/register             → create account (requires Chess.com verification)
POST /api/auth/login                → verify password, return 30-day session token
GET  /api/auth/check                → validate a stored token (for auto-login)
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import random
import secrets
import sys
import threading
import urllib.error
import urllib.request as _urllib_req
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS

try:
    import chess as _pychess
    _PYCHESS_OK = True
except ImportError:
    _pychess = None
    _PYCHESS_OK = False

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.puzzle_loader import (
    LOW_QUALITY_THEMES,
    WEAKNESS_CATEGORIES,
    resolve_primary_category,
)
from src.recommender.bandit import ThompsonBandit
from src.recommender.irt_model import (
    DEFAULT_PUZZLE_RD,
    MINED_PUZZLE_RD,
    IRTLearner,
)

FRONTEND_DIR      = ROOT / "web" / "frontend"
PROCESSED_PARQUET = ROOT / "data" / "processed" / "puzzles_full.parquet"
RAW_CSV           = ROOT / "DataSets" / "lichess_db_puzzle.csv"
SESSIONS_DIR      = ROOT / "data" / "sessions"

# Which weakness-scoring model analysis_start() should try first when Stockfish
# analysis succeeds. "rule-based" (default) = player_profiler.build_profile(),
# the hand-tuned-but-Stockfish-verified scorer. "ml" = ml_weakness_model's
# trained RandomForest classifier -- opt-in, since it is currently trained and
# cross-validated on simulated players (see dissertation_documentation.md §7.2),
# not yet on enough real players to trust as the default for every user. Either
# way, _heuristic_profile() remains the last-resort fallback if Stockfish can't
# be found at all or the chosen model errors out.
WEAKNESS_MODEL = os.environ.get("WEAKNESS_MODEL", "rule-based").strip().lower()

# Which policy chooses the next category and difficulty.
#   "irt"  — difficulty-aware IRT learner (src/recommender/irt_model.py):
#            Thompson Sampling on per-category ability offsets, and puzzle
#            difficulty pitched so the predicted solve rate is ~65 %.
#   "beta" — the original Beta-Bernoulli Thompson bandit with the player's
#            Elo ± 300 band. Kept for A/B comparison.
# Both models are updated on every attempt regardless of which one serves,
# so either can be switched on at any time with a warm posterior.
RECOMMENDER = os.environ.get("RECOMMENDER", "irt").strip().lower()
# Discount for the Beta bandit (1.0 = stationary, the original behaviour).
BANDIT_DISCOUNT = float(os.environ.get("BANDIT_DISCOUNT", "1.0"))
# Half-width of the rating window around the IRT target difficulty.
IRT_WINDOW = 150
# Game-analysis threads: the node-limited MultiPV analysis is heavier per
# position than the old 50 ms search, so use more of the machine.
ANALYSIS_WORKERS = max(2, min(8, (os.cpu_count() or 4) // 2))

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)


# ── JSON safety net ───────────────────────────────────────────────────────────
# Python's json module happily writes bare `NaN` / `Infinity` literals, which
# JavaScript's JSON.parse rejects.  Any NaN leaking out of pandas therefore made
# an entire response unparseable in the browser.  Convert them to null instead.

def _json_safe(obj):
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


class _SafeJSONProvider(DefaultJSONProvider):
    def dumps(self, obj, **kwargs):
        return super().dumps(_json_safe(obj), **kwargs)


app.json = _SafeJSONProvider(app)

# ── Data pools ────────────────────────────────────────────────────────────────
PUZZLE_POOL:   list[dict] = []
PUZZLE_BY_CAT: dict[str, list[dict]] = {}   # PrimaryCategory → [puzzle dicts]

# ── In-memory state ───────────────────────────────────────────────────────────
# username (lowercase) → {status, progress, message, priors, profile}
ANALYSIS_STORE: dict[str, dict] = {}
# username (lowercase) → {status, progress, message, count}
GENERATE_STORE: dict[str, dict] = {}
# username (lowercase) → ThompsonBandit
SESSION_STORE: dict[str, ThompsonBandit] = {}
# username (lowercase) → IRTLearner
IRT_STORE: dict[str, IRTLearner] = {}
# username → {puzzleId: {policy, propensity, predicted, source}} for the puzzles
# currently on screen, so /api/session/result can log how each was chosen.
LAST_SERVED: dict[str, dict[str, dict]] = defaultdict(dict)
_LAST_SERVED_CAP = 50
# Guests have no state file, so their "already seen" set lives here for the
# lifetime of the process.  Capped so a long-running server cannot grow it
# without bound.
GUEST_SEEN: set[str] = set()
_GUEST_SEEN_CAP = 2_000
# Upper bound on how many puzzle IDs we persist per user.  ~5k covers years of
# daily training while keeping the state file small.
_SEEN_HISTORY_CAP = 5_000

# ── Demo fallback puzzles ─────────────────────────────────────────────────────
DEMO_PUZZLES = [
    # Verified fallback set — every FEN parses and every move in every
    # sequence is legal (checked against python-chess).  Used only when no
    # dataset is present.  Do not hand-edit without re-validating.
    {"PuzzleId": "5NS8U", "FEN": "3r4/1k3p2/1p2p3/1PP2p2/2K2P2/R4RP1/8/7r b - - 2 42",
     "Moves": "d8c8 c5c6 c8c6 b5c6", "Rating": 906, "RatingDeviation": 76,
     "Popularity": 98, "NbPlays": 18815, "Themes": "advantage endgame rookEndgame short",
     "GameUrl": "https://lichess.org/KMFX0yYa/black#84", "OpeningTags": None,
     "DifficultyTier": "Beginner", "PrimaryCategory": "Rook Endgame", "Categories": ['Rook Endgame']},
    {"PuzzleId": "7K7yL", "FEN": "r3rn1k/pp4R1/2pq3p/4p2Q/2BP4/2P1P2P/PP4P1/R5K1 b - - 0 21",
     "Moves": "h8g7 h5f7 g7h8 f7g8", "Rating": 908, "RatingDeviation": 77,
     "Popularity": 98, "NbPlays": 18352, "Themes": "mate mateIn2 middlegame short",
     "GameUrl": "https://lichess.org/kwUM6zSi/black#42", "OpeningTags": None,
     "DifficultyTier": "Beginner", "PrimaryCategory": "Mating Pattern", "Categories": ['Mating Pattern']},
    {"PuzzleId": "7wpkn", "FEN": "1k6/pp2bp2/2p3rp/1q1p1Q2/3P1P2/1P2P1P1/P4K1P/R1N1n3 b - - 1 31",
     "Moves": "g6f6 f5e5 b8c8 e5e7", "Rating": 1169, "RatingDeviation": 78,
     "Popularity": 95, "NbPlays": 32206, "Themes": "advantage fork middlegame short",
     "GameUrl": "https://lichess.org/hNwDfB33/black#62", "OpeningTags": None,
     "DifficultyTier": "Easy", "PrimaryCategory": "Fork", "Categories": ['Fork']},
    {"PuzzleId": "ALDCT", "FEN": "r1b1k1nr/pp3ppp/2n5/q1bQ4/4N3/6P1/PP2PP1P/R1B1KBNR w KQkq - 3 9",
     "Moves": "c1d2 c5f2 e4f2 a5d5", "Rating": 1154, "RatingDeviation": 77,
     "Popularity": 95, "NbPlays": 32058, "Themes": "attackingF2F7 crushing discoveredAttack opening short",
     "GameUrl": "https://lichess.org/JEeJ9pO0#17", "OpeningTags": None,
     "DifficultyTier": "Easy", "PrimaryCategory": "Discovered Attack", "Categories": ['Discovered Attack']},
    {"PuzzleId": "83wOc", "FEN": "r1b2k2/pp3PR1/2p2n1P/3p4/3P4/1PN5/P1PK4/8 b - - 2 28",
     "Moves": "b7b5 h6h7 f6h7 g7h7", "Rating": 1404, "RatingDeviation": 79,
     "Popularity": 95, "NbPlays": 47594, "Themes": "advancedPawn advantage endgame short",
     "GameUrl": "https://lichess.org/2umPGLGl/black#56", "OpeningTags": None,
     "DifficultyTier": "Intermediate", "PrimaryCategory": "Endgame", "Categories": ['Endgame']},
    {"PuzzleId": "1C0l6", "FEN": "2k2r1r/pppq2p1/1bn1p3/4p1Bp/Q7/3P2P1/PP2PPBP/2R2RK1 b - - 5 16",
     "Moves": "c6d4 g2b7 c8b7 a4d7", "Rating": 1488, "RatingDeviation": 75,
     "Popularity": 96, "NbPlays": 47039, "Themes": "crushing deflection middlegame queensideAttack short",
     "GameUrl": "https://lichess.org/Fxp9fbmG/black#32", "OpeningTags": None,
     "DifficultyTier": "Intermediate", "PrimaryCategory": "Deflection", "Categories": ['Deflection']},
    {"PuzzleId": "98cVb", "FEN": "4k2r/1q2ppb1/2bp2p1/p1p5/1rPnP3/1P1QBPPp/P4R1P/1RN2NK1 w k - 1 24",
     "Moves": "c1e2 d4f3 f2f3 c6e4", "Rating": 1543, "RatingDeviation": 75,
     "Popularity": 95, "NbPlays": 45508, "Themes": "crushing kingsideAttack middlegame short",
     "GameUrl": "https://lichess.org/2OxvKBTv#47", "OpeningTags": None,
     "DifficultyTier": "Advanced", "PrimaryCategory": "King Safety", "Categories": ['King Safety']},
    {"PuzzleId": "79NI0", "FEN": "r3k2r/p1qp1pp1/1pn1p3/1Bb1P2p/5B2/2N2P2/PPPQ3P/2KR4 b kq - 4 15",
     "Moves": "e8c8 b5a6 c8b8 c3b5 d7d6 b5c7", "Rating": 1557, "RatingDeviation": 77,
     "Popularity": 94, "NbPlays": 45492, "Themes": "crushing long middlegame queensideAttack trappedPiece",
     "GameUrl": "https://lichess.org/NRu3IZPv/black#30", "OpeningTags": None,
     "DifficultyTier": "Advanced", "PrimaryCategory": "Hanging Piece", "Categories": ['Hanging Piece']},
    {"PuzzleId": "5uDSr", "FEN": "4r3/pR6/6pk/3P4/1P4R1/5p2/r2P1P2/3K4 w - - 1 34",
     "Moves": "g4f4 e8e1 d1e1 a2a1", "Rating": 1892, "RatingDeviation": 79,
     "Popularity": 93, "NbPlays": 154083, "Themes": "attraction endgame mate mateIn2 rookEndgame sacrifice short",
     "GameUrl": "https://lichess.org/lHnU7Ldg#67", "OpeningTags": None,
     "DifficultyTier": "Hard", "PrimaryCategory": "Mating Pattern", "Categories": ['Mating Pattern']},
    {"PuzzleId": "2vVE7", "FEN": "5r1k/1pp3p1/pb1p1q1p/5r2/3PQn1N/6BP/PP3PP1/2R1R1K1 b - - 9 25",
     "Moves": "f5h5 e4f4 f6f4 h4g6 h8h7 g6f4", "Rating": 1941, "RatingDeviation": 75,
     "Popularity": 92, "NbPlays": 148713, "Themes": "attraction crushing fork long middlegame sacrifice",
     "GameUrl": "https://lichess.org/JqItJ6B1/black#50", "OpeningTags": None,
     "DifficultyTier": "Hard", "PrimaryCategory": "Fork", "Categories": ['Fork']},
    {"PuzzleId": "6Sz3s", "FEN": "4k2r/p5pp/3bp3/4n3/1r5q/4Q3/PP2B1PP/R1B2R1K w k - 3 21",
     "Moves": "e3a7 h4h2 h1h2 e5f3 h2h3 b4h4", "Rating": 2078, "RatingDeviation": 78,
     "Popularity": 93, "NbPlays": 195444, "Themes": "attraction discoveredCheck doubleCheck long mate mateIn3 middlegame pillsburysMate sacrifice",
     "GameUrl": "https://lichess.org/zgBwsXLr#41", "OpeningTags": None,
     "DifficultyTier": "Expert", "PrimaryCategory": "Mating Pattern", "Categories": ['Mating Pattern']},
    {"PuzzleId": "9nPpJ", "FEN": "2Q5/8/1p1k4/2p3P1/2K2P2/r7/2P5/r7 w - - 0 50",
     "Moves": "g5g6 b6b5 c4b5 a1b1 b5c4 b1b4", "Rating": 2035, "RatingDeviation": 73,
     "Popularity": 94, "NbPlays": 177645, "Themes": "attraction endgame long mate mateIn3 queenRookEndgame",
     "GameUrl": "https://lichess.org/uynEluFS#99", "OpeningTags": None,
     "DifficultyTier": "Expert", "PrimaryCategory": "Mating Pattern", "Categories": ['Mating Pattern']},
]


# ── Puzzle loading ─────────────────────────────────────────────────────────────

_PARQUET_COLS = [
    "PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation",
    "Popularity", "NbPlays", "Themes", "GameUrl", "OpeningTags",
    "PrimaryCategory", "DifficultyTier", "Categories",
]
_POOL_CAP = 200_000   # puzzles kept in RAM

# ── Pool quality gate ─────────────────────────────────────────────────────────
# The Lichess dump contains a lot of material that is technically a valid puzzle
# but reads as nonsense in a tactics trainer.  Every threshold below removes a
# specific failure mode users reported as "this puzzle makes no sense":
#
#   MIN_POPULARITY   Lichess community up/down-vote ratio.  Below ~85 the
#                    community itself flags the puzzle as poor.
#   MIN_NB_PLAYS     Rating is meaningless on a puzzle nobody has solved.
#   MAX_RATING_DEV   High deviation = the difficulty label is a guess.
#   MIN_HALF_MOVES   3 = opponent setup move + player move + a reply.  Two-move
#                    puzzles are single recaptures with nothing to find.
#   General category The puzzle carries no recognisable motif (`advantage
#                    middlegame short`) — nothing to learn, no honest label.
#   LOW_QUALITY_THEMES  `equality` (hold the draw) and `defensiveMove` (retreat)
#                    — correct answers that feel wrong when drilling tactics.
MIN_POPULARITY = 85
MIN_NB_PLAYS   = 250
MAX_RATING_DEV = 90
MIN_HALF_MOVES = 3
MIN_RATING     = 600
MAX_RATING     = 2400


def _apply_quality_gate(df: pd.DataFrame) -> pd.DataFrame:
    """Filter a raw puzzle frame down to the pool we are willing to serve.

    Also recomputes PrimaryCategory with the priority-ordered resolver so a
    stale parquet (built with the old tag-order rule) is corrected in memory —
    no dataset rebuild required.
    """
    themes = df["Themes"].fillna("")
    n_moves = df["Moves"].fillna("").str.split().str.len()
    rating_dev = (
        df["RatingDeviation"] if "RatingDeviation" in df.columns
        else pd.Series(0, index=df.index)
    )

    mask = (
        df["Rating"].between(MIN_RATING, MAX_RATING)
        & (df["Popularity"] >= MIN_POPULARITY)
        & (df["NbPlays"] >= MIN_NB_PLAYS)
        & (rating_dev <= MAX_RATING_DEV)
        & (n_moves >= MIN_HALF_MOVES)
        & ~themes.apply(lambda t: bool(set(t.split()) & LOW_QUALITY_THEMES))
    )
    out = df[mask].copy()
    out["PrimaryCategory"] = out["Themes"].fillna("").apply(resolve_primary_category)
    return out[out["PrimaryCategory"] != "General"]


def _load_puzzles() -> None:
    global PUZZLE_POOL
    if PROCESSED_PARQUET.exists():
        print(f"Loading from parquet: {PROCESSED_PARQUET}")
        # Read one row-group at a time so we never hold more than ~1 M rows in
        # RAM simultaneously.  Filter and proportionally sample each chunk, then
        # concatenate — peak memory ≈ one row-group + the growing sample list.
        import pyarrow.parquet as _pq
        pf   = _pq.ParquetFile(str(PROCESSED_PARQUET))
        n_rg = pf.metadata.num_row_groups
        target_per_rg = max(1, _POOL_CAP // n_rg)
        avail_cols = pf.schema_arrow.names
        cols = [c for c in _PARQUET_COLS if c in avail_cols]

        chunks: list[pd.DataFrame] = []
        kept = dropped = 0
        for i in range(n_rg):
            rg = pf.read_row_group(i, columns=cols).to_pandas()
            raw_n = len(rg)
            rg = _apply_quality_gate(rg)
            kept    += len(rg)
            dropped += raw_n - len(rg)
            if len(rg) > target_per_rg:
                rg = rg.sample(target_per_rg, random_state=42 + i)
            chunks.append(rg)

        df = pd.concat(chunks, ignore_index=True)
        if len(df) > _POOL_CAP:
            df = df.sample(_POOL_CAP, random_state=42)
        PUZZLE_POOL = df.to_dict("records")
        print(f"Quality gate: kept {kept:,} / {kept + dropped:,} "
              f"({kept / max(1, kept + dropped) * 100:.1f}%)")
    elif RAW_CSV.exists():
        print(f"Loading sample from CSV: {RAW_CSV}")
        df = pd.read_csv(RAW_CSV, nrows=400_000)
        PUZZLE_POOL = _apply_quality_gate(df).to_dict("records")
    else:
        print("WARNING: No data found — demo mode (12 puzzles).")
        PUZZLE_POOL = DEMO_PUZZLES

    _build_category_index()
    print(f"Puzzle pool: {len(PUZZLE_POOL):,} puzzles, {len(PUZZLE_BY_CAT)} categories")


def _build_category_index() -> None:
    global PUZZLE_BY_CAT
    idx: dict[str, list[dict]] = defaultdict(list)
    for p in PUZZLE_POOL:
        cat = p.get("PrimaryCategory") or "General"
        idx[cat].append(p)
    PUZZLE_BY_CAT = dict(idx)


def _validate_moves(fen: str, moves) -> bool:
    """Return True iff every UCI move in the sequence is legal on the given FEN."""
    if not _PYCHESS_OK:
        return True
    if isinstance(moves, str):
        moves = moves.split()
    try:
        board = _pychess.Board(fen)
        for uci in moves:
            m = _pychess.Move.from_uci(uci)
            if m not in board.legal_moves:
                return False
            board.push(m)
        return True
    except Exception:
        return False


def _pick_valid(pool: list, tries: int = 20):
    """Sample up to *tries* puzzles and return the first one with valid moves."""
    if not pool:
        return None
    candidates = random.sample(pool, min(tries, len(pool)))
    for p in candidates:
        if _validate_moves(p.get("FEN", ""), p.get("Moves", "")):
            return p
    return None


def _is_missing(v) -> bool:
    """True for None and for pandas' float('nan') placeholder."""
    return v is None or (isinstance(v, float) and v != v)


def _text(v) -> str:
    """Coerce a possibly-NaN dataframe cell to a JSON-safe string.

    Empty CSV cells arrive from pandas as float('nan'), which is *truthy*, so
    `value or ""` let it through — and Flask serialises it as a bare `NaN`
    literal.  That is not valid JSON, so `response.json()` threw in the browser
    and every affected puzzle silently fell back to the built-in demo set.
    """
    return "" if _is_missing(v) else str(v)


def _num(v, default: int = 0) -> int:
    return default if _is_missing(v) else int(v)


def _serialise(puzzle: dict) -> dict:
    themes_raw = puzzle.get("Themes", "") or ""
    themes = themes_raw.split() if isinstance(themes_raw, str) else []
    _meta = {"short", "long", "veryLong", "oneMove", "crushing", "advantage", "equality"}
    display_themes = [t for t in themes if t not in _meta]

    categories = puzzle.get("Categories", [])
    if hasattr(categories, "tolist"):
        categories = categories.tolist()
    elif isinstance(categories, str):
        import ast
        try:
            categories = ast.literal_eval(categories)
        except Exception:
            categories = []

    game_url = _text(puzzle.get("GameUrl"))
    if not game_url.startswith("http"):
        game_url = ""   # some generated rows stored the literal "Chess.com"

    return {
        "id":             _text(puzzle["PuzzleId"]),
        "fen":            _text(puzzle["FEN"]),
        "moves":          puzzle["Moves"].split() if isinstance(puzzle["Moves"], str) else list(puzzle["Moves"]),
        "rating":         _num(puzzle.get("Rating"), 1200),
        "ratingDeviation": _num(puzzle.get("RatingDeviation"), 80),
        "popularity":     _num(puzzle.get("Popularity"), 80),
        "nbPlays":        _num(puzzle.get("NbPlays"), 0),
        "themes":         display_themes,
        "categories":     [_text(c) for c in categories],
        "primaryCategory": _text(puzzle.get("PrimaryCategory")) or "General",
        "difficultyTier": _text(puzzle.get("DifficultyTier")),
        "gameUrl":        game_url,
        "openingTags":    _text(puzzle.get("OpeningTags")),
    }


def _heuristic_profile(parsed_games: list[dict], username: str):
    """
    Build a PlayerProfile from Chess.com game results without Stockfish.
    Uses win/loss patterns and game length to estimate weakness scores.
    """
    from src.classifier.player_profiler import PlayerProfile

    if not parsed_games:
        p = PlayerProfile(username=username, estimated_elo=1200)
        p.weakness_scores = {cat: 0.5 for cat in WEAKNESS_CATEGORIES}
        return p

    n = len(parsed_games)
    won  = sum(1 for g in parsed_games if g and g.get("player_won") is True)
    lost = sum(1 for g in parsed_games if g and g.get("player_won") is False)
    win_rate = won / n

    ratings = [g["player_rating"] for g in parsed_games if g and g.get("player_rating", 0) > 0]
    elo = int(sum(ratings) / len(ratings)) if ratings else 1200

    short_losses = sum(1 for g in parsed_games
                       if g and g.get("player_won") is False and g.get("num_moves", 30) < 25)
    long_losses  = sum(1 for g in parsed_games
                       if g and g.get("player_won") is False and g.get("num_moves", 30) > 40)

    short_loss_rate = short_losses / max(1, lost)
    long_loss_rate  = long_losses  / max(1, lost)

    tactical_w = max(0.30, min(0.85, 0.75 - win_rate * 0.50 + short_loss_rate * 0.20))
    endgame_w  = max(0.25, min(0.80, 0.60 - win_rate * 0.30 + long_loss_rate  * 0.30))

    TACTICAL = {"Fork", "Pin", "Hanging Piece", "Discovered Attack", "Skewer",
                "Deflection", "King Safety", "Mating Pattern"}
    ENDGAME  = {"Endgame", "Rook Endgame", "Pawn Endgame",
                "Queen Endgame", "Knight Endgame", "Bishop Endgame"}

    scores = {}
    for cat in WEAKNESS_CATEGORIES:
        if cat in TACTICAL:
            scores[cat] = tactical_w
        elif cat in ENDGAME:
            scores[cat] = endgame_w
        else:
            scores[cat] = 0.5

    # extract opening stats from parsed games
    from collections import defaultdict
    from src.classifier.player_profiler import OpeningStat

    _white: dict = defaultdict(lambda: {"eco": "", "games": 0, "wins": 0, "losses": 0, "draws": 0})
    _black: dict = defaultdict(lambda: {"eco": "", "games": 0, "wins": 0, "losses": 0, "draws": 0})
    for g in parsed_games:
        if not g:
            continue
        op     = g.get("opening") or {}
        family = op.get("family") or "Unknown"
        eco    = op.get("eco") or ""
        color  = g.get("player_color", "white")
        won_g  = g.get("player_won")
        target = _white if color == "white" else _black
        target[family]["eco"]    = eco
        target[family]["games"] += 1
        if won_g is True:    target[family]["wins"]   += 1
        elif won_g is False: target[family]["losses"] += 1
        else:                target[family]["draws"]  += 1

    def _openings_h(sd) -> list:
        return sorted(
            [OpeningStat(eco=v["eco"], family=k, games_played=v["games"],
                         wins=v["wins"], losses=v["losses"], draws=v["draws"])
             for k, v in sd.items()],
            key=lambda s: s.games_played, reverse=True,
        )[:5]

    profile = PlayerProfile(username=username, estimated_elo=elo)
    profile.games_analysed     = n
    profile.games_won          = won
    profile.games_lost         = lost
    profile.games_drawn        = n - won - lost
    profile.weakness_scores    = scores
    profile.top_openings_white = _openings_h(_white)
    profile.top_openings_black = _openings_h(_black)
    return profile


def _serialise_profile(profile) -> dict:
    # Prefer evidence-based scores (misses / opportunities) when the analysis
    # produced enough critical positions; the ML path keeps its own scores.
    scores, source = profile.weakness_scores, "rules"
    if getattr(profile, "has_opportunity_data", False) and WEAKNESS_MODEL != "ml":
        scores, source = profile.empirical_weakness_scores(), "opportunities"
    elif WEAKNESS_MODEL == "ml":
        source = "ml"
    opp_stats = getattr(profile, "opportunity_stats", {}) or {}
    return {
        "username":       profile.username,
        "estimatedElo":   profile.estimated_elo,
        "gamesAnalysed":  profile.games_analysed,
        "gamesWon":       profile.games_won,
        "gamesLost":      profile.games_lost,
        "gamesDrawn":     profile.games_drawn,
        "winRate":        round(profile.win_rate * 100, 1),
        "accuracy":       round(getattr(profile, "accuracy_estimate", 0.0), 1),
        "weaknessScores": {k: round(v, 3) for k, v in scores.items()},
        "weaknessSource": source,
        "opportunities":  {k: {"n": v["raw_n"], "hitRate": v["rate"]} for k, v in opp_stats.items()},
        "overallHitRate": getattr(profile, "overall_hit_rate", None),
        "topOpeningsWhite": [
            {"family": o.family, "games": o.games_played,
             "winRate": round(o.win_rate * 100, 1)}
            for o in getattr(profile, "top_openings_white", [])[:5]
        ],
        "topOpeningsBlack": [
            {"family": o.family, "games": o.games_played,
             "winRate": round(o.win_rate * 100, 1)}
            for o in getattr(profile, "top_openings_black", [])[:5]
        ],
    }


# ── User-state persistence ────────────────────────────────────────────────────
# Each non-guest user gets data/sessions/<username>.json containing:
#   {username, estimatedElo, profile, bandit, history: [...], bestStreak, lastUpdated}

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_user_state(username: str) -> dict | None:
    path = SESSIONS_DIR / f"{username}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_user_state(username: str, state: dict) -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = SESSIONS_DIR / f"{username}.json"
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# ── "Already seen" tracking ───────────────────────────────────────────────────
# A puzzle is *seen* once the player has finished with it — solved, failed, or
# revealed the solution.  Seen puzzles are never served again while unseen ones
# remain, so a finished puzzle cannot come back the next time you press Next.

def _seen_ids(username: str) -> set[str]:
    """Every puzzle ID this user has already finished (solved or not)."""
    if username == "guest":
        return set(GUEST_SEEN)
    saved = _load_user_state(username)
    if not saved:
        return set()
    return set(saved.get("attemptedPuzzleIds", [])) | set(saved.get("solvedPuzzleIds", []))


def _mark_seen(state: dict, puzzle_id: str, solved: bool) -> None:
    """Record a finished puzzle on a user-state dict (caller persists it)."""
    if not puzzle_id:
        return
    attempted = list(dict.fromkeys(state.get("attemptedPuzzleIds", []) + [puzzle_id]))
    state["attemptedPuzzleIds"] = attempted[-_SEEN_HISTORY_CAP:]
    if solved:
        solved_ids = list(dict.fromkeys(state.get("solvedPuzzleIds", []) + [puzzle_id]))
        state["solvedPuzzleIds"] = solved_ids[-_SEEN_HISTORY_CAP:]


def _mark_guest_seen(puzzle_id: str) -> None:
    if not puzzle_id:
        return
    if len(GUEST_SEEN) >= _GUEST_SEEN_CAP:
        GUEST_SEEN.clear()
    GUEST_SEEN.add(puzzle_id)


# ── Auth helpers ─────────────────────────────────────────────────────────────

_TOKEN_DAYS = 30

# Pending Chess.com ownership challenges: {username: {"code": str, "expires": datetime}}
_VERIFY_CHALLENGES: dict = {}
_VERIFY_TTL = 600  # seconds


def _cleanup_challenges() -> None:
    now = datetime.now(timezone.utc)
    stale = [u for u, v in _VERIFY_CHALLENGES.items() if v["expires"] < now]
    for u in stale:
        del _VERIFY_CHALLENGES[u]


def _chess_com_location(username: str) -> str | None:
    """Return the Chess.com location field for username, or None on any error."""
    url = f"https://api.chess.com/pub/player/{username.lower()}"
    req = _urllib_req.Request(url, headers={"User-Agent": "PuzzleAdvisor/1.0"})
    try:
        with _urllib_req.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        return data.get("location") or ""
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None           # account doesn't exist
        return None
    except Exception:
        return None

def _hash_password(password: str) -> tuple[str, str]:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000)
    return salt, h.hex()


def _verify_password(password: str, salt: str, stored_hash: str) -> bool:
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000)
    return secrets.compare_digest(h.hex(), stored_hash)


def _new_token() -> tuple[str, str]:
    token  = secrets.token_urlsafe(32)
    expiry = (datetime.now(timezone.utc) + timedelta(days=_TOKEN_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return token, expiry


def _check_token(username: str) -> bool:
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not token:
        return False
    saved = _load_user_state(username)
    if not saved or saved.get("token") != token:
        return False
    expiry = saved.get("tokenExpiry", "")
    if expiry:
        try:
            exp_dt = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
            if exp_dt < datetime.now(timezone.utc):
                return False
        except ValueError:
            return False
    return True


# ── Auth endpoints ────────────────────────────────────────────────────────────

@app.route("/api/auth/challenge", methods=["POST"])
def auth_challenge():
    """Step 1: issue a verification code the user must paste into Chess.com location."""
    data     = request.get_json(silent=True) or {}
    username = (data.get("username") or "").lower().strip()
    if not username or username == "guest":
        return jsonify({"error": "Invalid username"}), 400

    existing = _load_user_state(username)
    if existing and existing.get("passwordHash"):
        return jsonify({"error": "Account already exists — log in instead"}), 409

    location = _chess_com_location(username)
    if location is None:
        return jsonify({"error": f"Chess.com account '{username}' not found"}), 404

    _cleanup_challenges()
    code = "PZLADV-" + secrets.token_hex(3).upper()
    _VERIFY_CHALLENGES[username] = {
        "code": code,
        "expires": datetime.now(timezone.utc) + timedelta(seconds=_VERIFY_TTL),
    }
    return jsonify({"ok": True, "code": code, "expiresIn": _VERIFY_TTL})


@app.route("/api/auth/verify-chess", methods=["POST"])
def auth_verify_chess():
    """Step 2: check whether the Chess.com location field contains the challenge code."""
    data              = request.get_json(silent=True) or {}
    username          = (data.get("username") or "").lower().strip()
    verification_code = (data.get("verificationCode") or "").strip()

    if not username or not verification_code:
        return jsonify({"error": "Missing fields"}), 400

    challenge = _VERIFY_CHALLENGES.get(username)
    if not challenge:
        return jsonify({"error": "No challenge found — request a new code"}), 400
    if datetime.now(timezone.utc) > challenge["expires"]:
        _VERIFY_CHALLENGES.pop(username, None)
        return jsonify({"error": "Code expired — request a new code"}), 400
    if challenge["code"] != verification_code:
        return jsonify({"error": "Code mismatch"}), 400

    location = _chess_com_location(username)
    if location is None:
        return jsonify({"error": f"Could not reach Chess.com for '{username}'"}), 503
    if verification_code not in location:
        return jsonify({"error": "Code not found in your Chess.com location field. Make sure you saved the profile."}), 400

    return jsonify({"ok": True, "username": username})


@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    data              = request.get_json(silent=True) or {}
    username          = (data.get("username") or "").lower().strip()
    password          = data.get("password") or ""
    verification_code = (data.get("verificationCode") or "").strip()

    if not username or not password or not verification_code:
        return jsonify({"error": "Missing required fields"}), 400
    if username == "guest":
        return jsonify({"error": "Reserved username"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    existing = _load_user_state(username)
    if existing and existing.get("passwordHash"):
        return jsonify({"error": "Account already exists — log in instead"}), 409

    challenge = _VERIFY_CHALLENGES.get(username)
    if not challenge:
        return jsonify({"error": "No verification challenge found — request a new code"}), 400
    if datetime.now(timezone.utc) > challenge["expires"]:
        _VERIFY_CHALLENGES.pop(username, None)
        return jsonify({"error": "Verification code expired — request a new code"}), 400
    if challenge["code"] != verification_code:
        return jsonify({"error": "Verification code mismatch"}), 400

    location = _chess_com_location(username)
    if location is None:
        return jsonify({"error": f"Could not reach Chess.com to verify '{username}'"}), 503
    if verification_code not in location:
        return jsonify({"error": "Verification code not found in your Chess.com location field. Make sure you saved the profile."}), 400

    salt, pw_hash = _hash_password(password)
    token, expiry = _new_token()
    state = existing or {"username": username, "history": [], "bestStreak": 0}
    state.update({"passwordSalt": salt, "passwordHash": pw_hash,
                  "token": token, "tokenExpiry": expiry,
                  "chessComVerified": True})
    _save_user_state(username, state)
    _VERIFY_CHALLENGES.pop(username, None)

    return jsonify({"ok": True, "token": token, "username": username})


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data     = request.get_json(silent=True) or {}
    username = (data.get("username") or "").lower().strip()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400

    saved = _load_user_state(username)
    if not saved or not saved.get("passwordHash"):
        return jsonify({"error": "No account found — create one first"}), 404

    if not _verify_password(password, saved["passwordSalt"], saved["passwordHash"]):
        return jsonify({"error": "Wrong password"}), 401

    token, expiry    = _new_token()
    saved["token"]       = token
    saved["tokenExpiry"] = expiry
    _save_user_state(username, saved)

    return jsonify({"ok": True, "token": token, "username": username})


@app.route("/api/auth/check")
def auth_check():
    username = request.args.get("username", "").lower().strip()
    if not username or not _check_token(username):
        return jsonify({"valid": False}), 401
    return jsonify({"valid": True, "username": username})


# ── Static serving ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


# ── Puzzle API (legacy / guest mode) ──────────────────────────────────────────

@app.route("/api/puzzle/random")
def puzzle_random():
    rating_min = request.args.get("ratingMin", 600,  type=int)
    rating_max = request.args.get("ratingMax", 2400, type=int)
    theme      = request.args.get("theme", None)

    pool = [
        p for p in PUZZLE_POOL
        if rating_min <= p["Rating"] <= rating_max
        and (theme is None or theme in (p.get("Themes") or ""))
    ]
    if not pool:
        return jsonify({"error": "No puzzles found"}), 404
    chosen = _pick_valid(pool) or random.choice(pool)
    return jsonify(_serialise(chosen))


@app.route("/api/puzzle/<puzzle_id>")
def puzzle_by_id(puzzle_id: str):
    for p in PUZZLE_POOL:
        if p["PuzzleId"] == puzzle_id:
            return jsonify(_serialise(p))
    return jsonify({"error": "Puzzle not found"}), 404


@app.route("/api/stats")
def stats():
    return jsonify({
        "totalPuzzles": len(PUZZLE_POOL),
        "categories":   len(PUZZLE_BY_CAT),
        "source": "parquet" if PROCESSED_PARQUET.exists() else
                  "csv_sample" if RAW_CSV.exists() else "demo",
    })


# ── Player lookup ─────────────────────────────────────────────────────────────

@app.route("/api/player/lookup")
def player_lookup():
    username = (request.args.get("username") or "").strip()
    if not username:
        return jsonify({"error": "username required"}), 400
    try:
        from src.api.chess_com_fetcher import get_player_profile
        profile = get_player_profile(username)
        return jsonify(profile)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 404


# ── Player style profile ──────────────────────────────────────────────────────

@app.route("/api/player/style/<username>")
def player_style(username: str):
    """
    Derive a tactical/stylistic profile from the player's cached game history.
    Returned without auth — style data is not sensitive.
    """
    username = username.strip().lower()
    if not username or username == "guest":
        return jsonify({"error": "username required"}), 400
    try:
        from src.api.chess_com_fetcher import get_recent_games
        from src.analysis.style_profile import compute_style_profile

        # Load all cached months — up to last 6 months, fast (no network)
        import json as _json
        from pathlib import Path as _Path

        cache_dir = ROOT / "data" / "cache" / "chess_com"
        pgns: list[str] = []
        acc_data: list[float | None] = []

        # Collect all game objects from cache files for this user
        for cache_file in sorted(cache_dir.glob(f"games_{username}_*.json"), reverse=True)[:6]:
            try:
                obj = _json.loads(cache_file.read_text(encoding="utf-8"))
                for g in obj.get("games", []):
                    if g.get("pgn"):
                        pgns.append(g["pgn"])
                        accs = g.get("accuracies", {})
                        # Pick the player's accuracy
                        white_name = (g.get("white", {}).get("username") or "").lower()
                        if username in white_name:
                            acc_data.append(accs.get("white"))
                        else:
                            acc_data.append(accs.get("black"))
            except Exception:
                continue

        if not pgns:
            # Try fetching from API (might be slow; fallback to empty)
            try:
                pgns = get_recent_games(username, n=30)
                acc_data = [None] * len(pgns)
            except Exception:
                return jsonify({"error": "No games found. Run game analysis first."}), 404

        profile = compute_style_profile(username, pgns, accuracy_data=acc_data)

        # Enrich with missed tactic types from generated puzzles
        from src.puzzles.generator import load_user_puzzles
        user_puzzles = load_user_puzzles(username)
        if user_puzzles:
            from collections import Counter
            tactic_counts = Counter(p.get("PrimaryCategory", "General") for p in user_puzzles)
            total_puz = len(user_puzzles)
            profile["missed_tactics"] = {
                cat: round(cnt / total_puz * 100)
                for cat, cnt in tactic_counts.most_common(5)
            }
        else:
            profile["missed_tactics"] = {}

        # Enrich with bandit weakness data if session exists
        bandit = SESSION_STORE.get(username)
        if bandit:
            weaknesses = {}
            for cat in bandit.arms:
                sr = bandit.solve_rate(cat)
                weaknesses[cat] = round(sr * 100)
            # Top 3 weakest categories
            profile["bandit_weaknesses"] = dict(
                sorted(weaknesses.items(), key=lambda x: x[1])[:5]
            )
        else:
            profile["bandit_weaknesses"] = {}

        return jsonify(profile)

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Background game analysis ───────────────────────────────────────────────────

@app.route("/api/analysis/start", methods=["POST"])
def analysis_start():
    data     = request.get_json(force=True) or {}
    username = (data.get("username") or "").strip().lower()
    if not username:
        return jsonify({"error": "username required"}), 400

    ANALYSIS_STORE[username] = {
        "status": "running", "progress": 0,
        "message": "Starting…", "priors": None, "profile": None,
    }

    def _run():
        try:
            from src.api.chess_com_fetcher import get_player_profile, get_recent_games, get_best_rating
            from src.classifier.stockfish_analyzer import find_stockfish
            from src.data.pgn_parser import parse_games_bulk
            from src.classifier.player_profiler import (
                build_profile, profile_to_bandit_priors, profile_to_irt_prior,
            )

            ANALYSIS_STORE[username]["message"] = "Fetching profile from Chess.com…"
            ch_profile = get_player_profile(username)
            elo = get_best_rating(ch_profile)
            ANALYSIS_STORE[username]["progress"] = 10

            ANALYSIS_STORE[username]["message"] = "Fetching recent games…"
            pgns = get_recent_games(username, n=50)
            ANALYSIS_STORE[username]["progress"] = 35

            profile     = None
            model_used  = None
            # Was shutil.which("stockfish") -- only matched a binary literally
            # named "stockfish" on PATH, so it never found the bundled engine at
            # stockfish/stockfish-windows-x86-64-avx2.exe and this analysis path
            # silently fell back to the heuristic scorer below on every run.
            # find_stockfish() checks that bundled path too (see §6.1 of
            # dissertation_documentation.md for how this was found).
            sf_path = find_stockfish()

            if sf_path:
                try:
                    from src.classifier.stockfish_analyzer import analyze_games_parallel
                    ANALYSIS_STORE[username]["message"] = f"Running Stockfish on {len(pgns)} games…"
                    analyses = analyze_games_parallel(
                        pgns, username, sf_path, workers=ANALYSIS_WORKERS,
                        progress_callback=lambda done, total: ANALYSIS_STORE[username].update(
                            progress=35 + int(50 * done / max(1, total))),
                    )
                    ANALYSIS_STORE[username]["progress"] = 85
                    successful = [a for a in analyses if not getattr(a, "failed", True)]
                    if successful:
                        if WEAKNESS_MODEL == "ml":
                            try:
                                from src.classifier.ml_weakness_model import (
                                    build_profile_ml, DEFAULT_MODEL_PATH,
                                )
                                if DEFAULT_MODEL_PATH.exists():
                                    profile = build_profile_ml(analyses, username, estimated_elo=elo)
                                    model_used = "ml (RandomForest, trained on simulated players)"
                            except Exception:
                                profile = None  # fall through to the rule-based scorer below
                        if profile is None:
                            profile    = build_profile(analyses, username, estimated_elo=elo)
                            model_used = model_used or "rule-based (Stockfish-verified)"
                except Exception:
                    pass  # fall through to heuristic

            if profile is None:
                ANALYSIS_STORE[username]["message"] = "Building weakness profile (heuristic)…"
                parsed = [g for g in parse_games_bulk(pgns, username) if g]
                ANALYSIS_STORE[username]["progress"] = 60
                profile    = _heuristic_profile(parsed, username)
                model_used = "heuristic (no Stockfish found, or analysis failed)"

            ANALYSIS_STORE[username]["progress"] = 95
            priors = profile_to_bandit_priors(profile)
            irt_prior = profile_to_irt_prior(profile)
            serialised = _serialise_profile(profile)
            ANALYSIS_STORE[username].update({
                "status":    "done",
                "progress":  100,
                "message":   "Analysis complete!",
                "priors":    {k: list(v) for k, v in priors.items()},
                "irtPrior":  {k: list(v) for k, v in irt_prior.items()},
                "profile":   serialised,
                "modelUsed": model_used,
                "evidence":  "opportunities" if getattr(profile, "has_opportunity_data", False)
                             else "weakness-scores",
            })
            # Persist profile + ELO so the dashboard reloads on next visit
            if username != "guest":
                _ustate = _load_user_state(username) or {
                    "username": username, "history": [], "bestStreak": 0,
                }
                _ustate["profile"]      = serialised
                _ustate["estimatedElo"] = profile.estimated_elo
                _ustate["irtPrior"]     = {k: list(v) for k, v in irt_prior.items()}
                _ustate["lastUpdated"]  = _now_iso()
                _save_user_state(username, _ustate)

        except Exception as exc:
            ANALYSIS_STORE[username].update({
                "status":  "error",
                "message": str(exc),
            })

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/analysis/status/<username>")
def analysis_status(username: str):
    username = username.lower()
    state = ANALYSIS_STORE.get(username, {
        "status": "not_started", "progress": 0, "message": "",
    })
    return jsonify({k: v for k, v in state.items() if k != "history"})


# ── Puzzle generation from player games ───────────────────────────────────────

@app.route("/api/generate/puzzles", methods=["POST"])
def generate_puzzles():
    data              = request.get_json(force=True) or {}
    username          = (data.get("username") or "").strip().lower()
    target_categories = data.get("targetCategories") or []
    if not username:
        return jsonify({"error": "username required"}), 400

    if GENERATE_STORE.get(username, {}).get("status") == "running":
        return jsonify({"status": "already_running"}), 409

    from src.classifier.stockfish_analyzer import find_stockfish
    sf_path = find_stockfish()
    if not sf_path:
        return jsonify({
            "error": "Stockfish not found. Place the binary at "
                     "stockfish/stockfish-windows-x86-64-avx2.exe or ensure it is in PATH.",
            "status": "error",
        }), 503

    GENERATE_STORE[username] = {
        "status": "running", "progress": 0, "message": "Starting…",
        "count": 0, "targetCategories": target_categories,
    }

    def _run():
        try:
            from src.api.chess_com_fetcher import get_recent_games
            from src.puzzles.generator import generate_from_games, save_user_puzzles

            target_label = (
                f" (targeting: {', '.join(target_categories)})"
                if target_categories else ""
            )
            GENERATE_STORE[username]["message"] = f"Fetching recent games from Chess.com{target_label}…"
            pgns = get_recent_games(username, n=60)
            if not pgns:
                GENERATE_STORE[username].update({
                    "status": "error",
                    "message": "No games found on Chess.com for this account. "
                               "Make sure your game history is public.",
                })
                return
            GENERATE_STORE[username]["progress"] = 10
            GENERATE_STORE[username]["message"] = (
                f"Analysing {len(pgns)} games with Stockfish{target_label}…"
            )

            def _cb(done, total, found=0):
                pct = 10 + int(done / total * 85)
                GENERATE_STORE[username]["progress"] = pct
                found_str = f" — {found} found" if found else ""
                GENERATE_STORE[username]["message"] = (
                    f"Game {done}/{total}{found_str}{target_label}…"
                )

            puzzles = generate_from_games(pgns, username, sf_path,
                                          max_total=30, progress_callback=_cb,
                                          target_categories=target_categories or None)

            if puzzles:
                save_user_puzzles(username, puzzles)

            msg = (
                f"Generated {len(puzzles)} verified puzzles from your games!"
                if puzzles else
                "No qualifying puzzles found in recent games. "
                "Try playing more games and regenerate."
            )
            GENERATE_STORE[username].update({
                "status":   "done",
                "progress": 100,
                "message":  msg,
                "count":    len(puzzles),
            })
        except Exception as exc:
            GENERATE_STORE[username].update({"status": "error", "message": str(exc)})

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "started", "stockfishPath": sf_path})


@app.route("/api/generate/status/<username>")
def generate_status(username: str):
    return jsonify(GENERATE_STORE.get(username.lower(), {
        "status": "not_started", "progress": 0, "message": "", "count": 0,
    }))


@app.route("/api/generate/list/<username>")
def generate_list(username: str):
    username = username.lower()
    if not _check_token(username):
        return jsonify({"error": "Unauthorized"}), 401
    from src.puzzles.generator import load_user_puzzles
    puzzles = load_user_puzzles(username)
    # Return raw dicts so all custom fields (userFeedback, generatedBy…) are preserved.
    # The frontend handles both PascalCase raw and camelCase serialised formats.
    return jsonify({"count": len(puzzles), "puzzles": puzzles})


@app.route("/api/generate/quality/<username>")
def generate_quality(username: str):
    """Return quality metrics computed from stored puzzle metadata (no Stockfish needed)."""
    username = username.lower()
    if not _check_token(username):
        return jsonify({"error": "Unauthorized"}), 401

    from collections import Counter
    from src.puzzles.generator import load_user_puzzles
    puzzles = load_user_puzzles(username)
    n = len(puzzles)
    if not n:
        return jsonify({"error": "No puzzles found"}), 404

    def _moves_len(p):
        m = p.get("Moves", "")
        return len(m.split()) if isinstance(m, str) else len(m or [])

    stockfish_n   = sum(1 for p in puzzles if p.get("generatedBy") == "stockfish")
    clarity_vals  = [p["clarityCp"]   for p in puzzles if p.get("clarityCp")  is not None]
    drop_vals     = [p["evalDrop"]     for p in puzzles if p.get("evalDrop")   is not None]
    depths        = [_moves_len(p)     for p in puzzles]
    player_moves  = [p.get("playerMoves", 1) for p in puzzles]

    clear_n       = sum(1 for v in clarity_vals if v >= 150)
    deep_n        = sum(1 for d in depths if d >= 3)
    multi_n       = sum(1 for v in player_moves if v >= 2)
    tactical_cats = {"Fork","Pin","Skewer","Discovered Attack","Mating Pattern",
                     "Sacrifice","Promotion","Hanging Piece","X-Ray Attack","King Safety"}
    tactical_n    = sum(1 for p in puzzles if p.get("PrimaryCategory","") in tactical_cats)

    avg_clarity   = round(sum(clarity_vals) / len(clarity_vals)) if clarity_vals else None
    avg_drop      = round(sum(drop_vals)    / len(drop_vals))    if drop_vals    else None
    avg_depth     = round(sum(depths) / n, 1)

    # Weighted overall score (mirrors eval/puzzle_evaluator.py)
    engine_score  = stockfish_n  / n
    clarity_score = (clear_n / n) if clarity_vals else 0.5
    depth_score   = deep_n       / n
    tactical_score= tactical_n   / n
    multi_score   = multi_n      / n

    overall = round(
        0.30 * engine_score  +
        0.25 * clarity_score +
        0.20 * depth_score   +
        0.15 * tactical_score +
        0.10 * multi_score,
        3,
    )
    grade = "GOOD" if overall >= 0.70 else "FAIR" if overall >= 0.50 else "POOR"

    categories = dict(Counter(p.get("PrimaryCategory","General") for p in puzzles))

    per_puzzle = [
        {
            "id":          p.get("PuzzleId", p.get("id", "")),
            "category":    p.get("PrimaryCategory", "General"),
            "rating":      p.get("Rating", 0),
            "clarityCp":   p.get("clarityCp"),
            "evalDrop":    p.get("evalDrop"),
            "playerMoves": p.get("playerMoves", 1),
            "depth":       _moves_len(p),
            "generatedBy": p.get("generatedBy", "?"),
        }
        for p in puzzles
    ]

    return jsonify({
        "n":            n,
        "overall":      overall,
        "grade":        grade,
        "metrics": {
            "engineVerified": {"n": stockfish_n,  "pct": round(stockfish_n /n*100)},
            "clarityOk":      {"n": clear_n,      "pct": round(clear_n/n*100) if clarity_vals else None, "avgCp": avg_clarity},
            "sufficientDepth":{"n": deep_n,        "pct": round(deep_n/n*100)},
            "multiMove":      {"n": multi_n,       "pct": round(multi_n/n*100)},
            "tactical":       {"n": tactical_n,    "pct": round(tactical_n/n*100)},
            "avgEvalDrop":    avg_drop,
            "avgDepth":       avg_depth,
        },
        "categories":   categories,
        "perPuzzle":    per_puzzle,
        "solvedIds":    list((_load_user_state(username) or {}).get("solvedPuzzleIds", [])),
        "seenIds":      sorted(_seen_ids(username)),
    })


@app.route("/api/generate/puzzle/<username>/<puzzle_id>", methods=["PATCH"])
def update_puzzle_feedback(username: str, puzzle_id: str):
    """Update a puzzle's userFeedback field: liked | disliked | null."""
    username = username.lower()
    if not _check_token(username):
        return jsonify({"error": "Unauthorized"}), 401

    data     = request.get_json(silent=True) or {}
    feedback = data.get("feedback")   # "liked", "disliked", or None/absent = clear
    if feedback not in ("liked", "disliked", None):
        return jsonify({"error": "feedback must be 'liked', 'disliked', or null"}), 400

    from src.puzzles.generator import load_user_puzzles, USER_PUZZLES_DIR
    puzzles = load_user_puzzles(username)
    updated = False
    for p in puzzles:
        if p.get("PuzzleId") == puzzle_id:
            if feedback is None:
                p.pop("userFeedback", None)
            else:
                p["userFeedback"] = feedback
            updated = True
            break

    if not updated:
        return jsonify({"error": "Puzzle not found"}), 404

    USER_PUZZLES_DIR.mkdir(parents=True, exist_ok=True)
    (USER_PUZZLES_DIR / f"{username}.json").write_text(
        json.dumps(puzzles, indent=2), encoding="utf-8"
    )
    return jsonify({"ok": True, "puzzleId": puzzle_id, "feedback": feedback})


@app.route("/api/generate/puzzle/<username>/<puzzle_id>", methods=["DELETE"])
def delete_puzzle(username: str, puzzle_id: str):
    """Remove a single generated puzzle by ID."""
    username = username.lower()
    if not _check_token(username):
        return jsonify({"error": "Unauthorized"}), 401

    from src.puzzles.generator import load_user_puzzles, USER_PUZZLES_DIR
    puzzles  = load_user_puzzles(username)
    filtered = [p for p in puzzles if p.get("PuzzleId") != puzzle_id]

    if len(filtered) == len(puzzles):
        return jsonify({"error": "Puzzle not found"}), 404

    USER_PUZZLES_DIR.mkdir(parents=True, exist_ok=True)
    (USER_PUZZLES_DIR / f"{username}.json").write_text(
        json.dumps(filtered, indent=2), encoding="utf-8"
    )
    return jsonify({"ok": True, "remaining": len(filtered)})


@app.route("/api/generate/puzzles/<username>", methods=["DELETE"])
def delete_all_puzzles(username: str):
    """Delete the entire puzzle file for a user (full reset before regeneration)."""
    username = username.lower()
    if not _check_token(username):
        return jsonify({"error": "Unauthorized"}), 401

    from src.puzzles.generator import USER_PUZZLES_DIR
    path = USER_PUZZLES_DIR / f"{username}.json"
    count = 0
    if path.exists():
        try:
            count = len(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
        path.unlink()

    return jsonify({"ok": True, "deleted": count})


# ── Adaptive session ──────────────────────────────────────────────────────────

@app.route("/api/session/start", methods=["POST"])
def session_start():
    data          = request.get_json(force=True) or {}
    username      = (data.get("username") or "guest").strip().lower()
    priors_raw    = data.get("priors")        # {category: [alpha, beta]} or None
    estimated_elo = data.get("estimatedElo")  # int or None

    saved     = _load_user_state(username) if username != "guest" else None
    returning = saved is not None and bool(saved.get("bandit"))

    if returning:
        bandit = ThompsonBandit.from_dict(saved["bandit"])
    else:
        priors = None
        if priors_raw:
            # Floats, not int(): evidence-scaled priors are fractional, and
            # truncating them threw information away.
            priors = {cat: (float(v[0]), float(v[1])) for cat, v in priors_raw.items()
                      if isinstance(v, (list, tuple)) and len(v) == 2}
        bandit = ThompsonBandit(priors=priors, discount=BANDIT_DISCOUNT)

    SESSION_STORE[username] = bandit
    learner = _init_learner(username, saved, data.get("irtPrior"), estimated_elo)

    # Persist ELO and initialise state file for first-time users
    if username != "guest" and (estimated_elo or saved):
        _s = saved or {"username": username, "history": [], "bestStreak": 0}
        if estimated_elo:
            _s["estimatedElo"] = int(estimated_elo)
        _s["lastUpdated"]  = _now_iso()
        if not returning:
            _s["bandit"] = bandit.to_dict()
        _s["irt"] = learner.to_dict()
        _save_user_state(username, _s)

    return jsonify({
        "status":        "ok",
        "returning":     returning,
        "policy":        RECOMMENDER,
        "weaknessMap":   _active_weakness_map(username),
        "topWeaknesses": _active_top_weaknesses(username),
        "categoryRatings": _category_ratings(learner),
    })


def _init_learner(username: str, saved: dict | None, irt_prior_raw, elo) -> IRTLearner:
    """Load, rebuild, or create the player's IRT learner, and cache it."""
    prior = None
    raw = irt_prior_raw or (saved or {}).get("irtPrior")
    if raw:
        prior = {c: (float(v[0]), float(v[1])) for c, v in raw.items()
                 if isinstance(v, (list, tuple)) and len(v) == 2}
    elo = elo or (saved or {}).get("estimatedElo")
    if saved and saved.get("irt"):
        learner = IRTLearner.from_dict(saved["irt"])
    elif saved and saved.get("history"):
        # A player who trained before this model existed: replay their real
        # attempts so the posterior starts from evidence, not from scratch.
        learner = IRTLearner.from_history(saved["history"], elo=elo, delta_prior=prior)
    else:
        learner = IRTLearner.new(elo=elo, delta_prior=prior)
    IRT_STORE[username] = learner
    return learner


def _learner(username: str) -> IRTLearner:
    learner = IRT_STORE.get(username)
    if learner is None:
        saved = _load_user_state(username) if username != "guest" else None
        learner = _init_learner(username, saved, None, None)
    return learner


def _bandit(username: str) -> ThompsonBandit:
    bandit = SESSION_STORE.get(username)
    if bandit is None:
        bandit = ThompsonBandit(discount=BANDIT_DISCOUNT)
        SESSION_STORE[username] = bandit
    return bandit


def _active_weakness_map(username: str) -> dict[str, float]:
    if RECOMMENDER == "irt":
        return _learner(username).weakness_map()
    return _bandit(username).weakness_map()


def _active_top_weaknesses(username: str, n: int = 5) -> list[dict]:
    if RECOMMENDER == "irt":
        return _learner(username).top_weaknesses(n)
    return _bandit(username).top_weaknesses(n)


def _category_ratings(learner: IRTLearner) -> dict[str, dict]:
    out = {}
    for cat in WEAKNESS_CATEGORIES:
        r, rd = learner.category_rating(cat)
        out[cat] = {"rating": round(r), "rd": round(rd)}
    return out


# ── Eval-bar engine ───────────────────────────────────────────────────────────
# The eval bar fires on every move of every puzzle.  Spawning a fresh Stockfish
# process per request exhausted process handles as soon as the player clicked
# through puzzles quickly — dozens of engines racing at depth 14, and the
# endpoint started returning 500s.  One long-lived engine behind a lock costs a
# few megabytes and removes both the spawn latency and the failure mode.

_EVAL_ENGINE = None
_EVAL_LOCK   = threading.Lock()
_EVAL_DEPTH  = 12


def _eval_engine():
    """Return the shared analysis engine, starting it on first use."""
    global _EVAL_ENGINE
    if _EVAL_ENGINE is None:
        from src.classifier.stockfish_analyzer import find_stockfish
        sf_path = find_stockfish()
        if not sf_path:
            return None
        import chess.engine
        _EVAL_ENGINE = chess.engine.SimpleEngine.popen_uci(sf_path)
        _EVAL_ENGINE.configure({"Threads": 1, "Hash": 32})
    return _EVAL_ENGINE


def _close_eval_engine() -> None:
    global _EVAL_ENGINE
    if _EVAL_ENGINE is not None:
        try:
            _EVAL_ENGINE.quit()
        except Exception:
            pass
        _EVAL_ENGINE = None


atexit.register(_close_eval_engine)


@app.route("/api/eval")
def position_eval():
    """Quick Stockfish evaluation of a FEN position for the eval bar."""
    fen = request.args.get("fen", "").strip()
    if not fen:
        return jsonify({"error": "Missing fen"}), 400

    import chess
    import chess.engine

    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({"error": "Invalid FEN"}), 400

    with _EVAL_LOCK:
        for attempt in (1, 2):
            engine = _eval_engine()
            if engine is None:
                return jsonify({"error": "Stockfish not available"}), 503
            try:
                info = engine.analyse(board, chess.engine.Limit(depth=_EVAL_DEPTH))
                break
            except chess.engine.EngineError:
                return jsonify({"error": "Position could not be analysed"}), 422
            except Exception:
                # Engine died (crash, broken pipe).  Drop it and let the next
                # attempt start a fresh one; give up if that also fails.
                _close_eval_engine()
                if attempt == 2:
                    return jsonify({"error": "Engine unavailable"}), 503

    score = info["score"].white()
    if score.is_mate():
        m = score.mate()
        return jsonify({"mate": m, "cp": None,
                        "advantage": "white" if (m or 0) > 0 else "black"})
    cp = score.score()
    return jsonify({"cp": cp, "mate": None,
                    "advantage": "white" if (cp or 0) > 0 else "black" if (cp or 0) < 0 else "equal"})


@app.route("/api/session/puzzle")
def session_puzzle():
    username   = (request.args.get("username") or "guest").strip().lower()
    rating_min = request.args.get("ratingMin", 600,  type=int)
    rating_max = request.args.get("ratingMax", 2400, type=int)
    elo        = request.args.get("elo", type=int)

    bandit = _bandit(username)
    learner = _learner(username)
    target_rating = None

    if RECOMMENDER == "irt":
        # Category: Thompson Sampling on difficulty-adjusted ability offsets,
        # restricted to categories that actually have puzzles to serve.
        allowed = [c for c in WEAKNESS_CATEGORIES if PUZZLE_BY_CAT.get(c)]
        target = learner.select_category(allowed=allowed or None)
        propensity = learner.selection_probabilities(n_samples=1000).get(target, 0.0)
        # Difficulty: pitched so the predicted solve rate is p_target (~65 %),
        # inside any band the player explicitly chose.
        target_rating = learner.target_rating(target)
        lo = max(rating_min, int(target_rating - IRT_WINDOW))
        hi = min(rating_max, int(target_rating + IRT_WINDOW))
        if lo <= hi:
            rating_min, rating_max = lo, hi
    else:
        # Original policy: the player's known ELO ± 300, Beta-bandit category.
        if elo:
            rating_min = max(rating_min, elo - 300)
            rating_max = min(rating_max, elo + 300)
        target = bandit.select_one()
        propensity = bandit.selection_probabilities(n_samples=1000).get(target, 0.0)
    selected = target
    solve_rate = bandit.solve_rate(target)

    # Everything this player has already finished — solved *or* failed.  A
    # puzzle you have seen the answer to teaches nothing the second time.
    seen_ids = _seen_ids(username)
    # The client may also pass IDs it served locally (generated-puzzle mode),
    # which the server never observed.
    for extra in request.args.getlist("exclude"):
        seen_ids.update(i for i in extra.split(",") if i)

    def _unseen(p):
        return p.get("PuzzleId") not in seen_ids

    def _in_range(p, lo, hi):
        return lo <= p.get("Rating", 1200) <= hi

    # Blend user-generated puzzles (priority) with Lichess pool
    from src.puzzles.generator import load_user_puzzles
    user_puzzles = load_user_puzzles(username, rerate=True) if username != "guest" else []

    chosen = None
    exhausted = False

    if user_puzzles:
        user_cat = [p for p in user_puzzles
                    if p.get("PrimaryCategory") == target
                    and _in_range(p, rating_min, rating_max) and _unseen(p)]
        if user_cat:
            chosen = dict(_pick_valid(user_cat) or random.choice(user_cat), source="generated")
        elif random.random() < 0.4:
            user_range = [p for p in user_puzzles
                          if _in_range(p, rating_min, rating_max) and _unseen(p)]
            if user_range:
                chosen = dict(_pick_valid(user_range) or random.choice(user_range),
                              source="generated")
                target = chosen.get("PrimaryCategory", target)

    # Fall back to the Lichess pool, relaxing one constraint at a time.  Order
    # matters: we would rather widen the rating band than abandon the category
    # the bandit asked for, and we would rather change category than repeat a
    # puzzle the player has already finished.
    if chosen is None:
        wide_min, wide_max = max(MIN_RATING, rating_min - 200), min(MAX_RATING, rating_max + 200)
        attempts = [
            # (pool, lo, hi, keeps_target)
            (PUZZLE_BY_CAT.get(target, []), rating_min, rating_max, True),
            (PUZZLE_BY_CAT.get(target, []), wide_min,   wide_max,   True),
            (PUZZLE_POOL,                   rating_min, rating_max, False),
            (PUZZLE_POOL,                   MIN_RATING, MAX_RATING, False),
        ]
        for pool, lo, hi, keeps_target in attempts:
            candidates = [p for p in pool if _in_range(p, lo, hi) and _unseen(p)]
            if candidates:
                chosen = _pick_valid(candidates) or random.choice(candidates)
                if not keeps_target:
                    target = chosen.get("PrimaryCategory", target)
                break

        if chosen is None:
            # Genuinely nothing left unseen — allow repeats and say so, rather
            # than silently serving a puzzle the player already solved.
            repeat_pool = [p for p in PUZZLE_POOL if _in_range(p, rating_min, rating_max)]
            if not repeat_pool:
                return jsonify({"error": "No puzzles found in this rating range"}), 404
            chosen = _pick_valid(repeat_pool) or random.choice(repeat_pool)
            target = chosen.get("PrimaryCategory", target)
            exhausted = True

    result = _serialise(chosen)
    source = chosen.get("source", "lichess")
    puzzle_rd = _puzzle_rd(chosen)
    predicted = learner.predict(target, chosen.get("Rating", 1500), puzzle_rd)
    cat_rating, cat_rd = learner.category_rating(target)

    result["targetCategory"]  = target
    # Under the IRT policy this is the model's prediction for THIS puzzle
    # (difficulty-aware); under the Beta policy, the category's solve rate.
    result["targetSolveRate"] = round(predicted if RECOMMENDER == "irt" else solve_rate, 3)
    result["predictedSolveProb"] = round(predicted, 3)
    result["policy"]          = RECOMMENDER
    result["targetRating"]    = round(target_rating) if target_rating else None
    result["categoryRating"]  = round(cat_rating)
    result["categoryRd"]      = round(cat_rd)
    result["source"]          = source
    result["poolExhausted"]   = exhausted

    served = LAST_SERVED[username]
    served[str(chosen.get("PuzzleId", ""))] = {
        "policy": RECOMMENDER,
        "selected": selected,
        "relabelled": target != selected,
        "propensity": round(propensity, 4),
        "predicted": round(predicted, 4),
        "source": source,
        "rd": puzzle_rd,
    }
    while len(served) > _LAST_SERVED_CAP:
        served.pop(next(iter(served)))
    return jsonify(result)


def _puzzle_rd(puzzle: dict) -> float:
    """Uncertainty about a puzzle's difficulty, in rating points, for the IRT
    update. Lichess puzzles use DEFAULT_PUZZLE_RD. A mined puzzle rated by
    PuzzleNet uses the network's own per-puzzle deviation. A mined puzzle that only
    has the fixed-formula rating keeps the wide MINED_PUZZLE_RD."""
    if puzzle.get("source") != "generated":
        return DEFAULT_PUZZLE_RD
    if puzzle.get("ratingModel") == "puzzlenet" and puzzle.get("RatingDeviation"):
        return float(puzzle["RatingDeviation"])
    return MINED_PUZZLE_RD


@app.route("/api/session/result", methods=["POST"])
def session_result():
    data      = request.get_json(force=True) or {}
    username  = (data.get("username") or "guest").strip().lower()
    category  = data.get("category", "")
    solved    = bool(data.get("solved", False))
    puzzle_id = data.get("puzzleId", "")
    rating    = int(data.get("rating", 0)) if data.get("rating") else 0

    # A skipped puzzle counts as seen but must not move the bandit or the
    # accuracy stats — the player never claimed to have tried it.
    skipped = bool(data.get("skipped", False))

    bandit = _bandit(username)
    learner = _learner(username)
    served = LAST_SERVED[username].pop(str(puzzle_id), {})

    predicted = None
    if category and not skipped:
        bandit.update(category, solved)
        if rating:
            # Both models learn from every attempt, whichever one served it.
            predicted = learner.update(category, rating, solved,
                                       served.get("rd", DEFAULT_PUZZLE_RD))

    if username == "guest":
        if not skipped:
            _mark_guest_seen(puzzle_id)
    else:
        _s = _load_user_state(username) or {
            "username": username, "history": [], "bestStreak": 0,
        }
        _s["lastUpdated"] = _now_iso()
        _s["bandit"]      = bandit.to_dict()
        _s["irt"]         = learner.to_dict()
        _s["bestStreak"]  = max(_s.get("bestStreak", 0), bandit.best_streak)
        if not skipped:
            entry = {
                "ts":       _now_iso(),
                "puzzleId": puzzle_id,
                "category": category,
                "rating":   rating,
                "solved":   solved,
            }
            # How this puzzle was chosen, and what the model predicted before
            # seeing the outcome: enables calibration checks on real data and
            # off-policy (inverse-propensity) evaluation of other policies.
            if served:
                entry.update({
                    "policy":     served.get("policy"),
                    "selected":   served.get("selected"),
                    "relabelled": served.get("relabelled"),
                    "propensity": served.get("propensity"),
                    "source":     served.get("source"),
                })
            if predicted is not None:
                entry["predicted"] = round(predicted, 4)
            _s.setdefault("history", []).append(entry)
            # Only a genuine attempt burns the puzzle.  A skip leaves it in the
            # pool so the player can meet it again another day.
            _mark_seen(_s, puzzle_id, solved)
        _save_user_state(username, _s)

    return jsonify({
        "streak":        bandit.streak,
        "bestStreak":    bandit.best_streak,
        "accuracy":      round(bandit.session_accuracy() * 100, 1),
        "puzzlesPlayed": bandit.puzzles_played(),
        "weaknessMap":   _active_weakness_map(username),
        "topWeaknesses": _active_top_weaknesses(username),
        "categoryRatings": _category_ratings(learner),
    })


@app.route("/api/session/model/<username>")
def session_model(username: str):
    """Per-category skill estimates (rating ± RD) from the IRT learner."""
    username = username.lower()
    learner = _learner(username)
    return jsonify({
        "policy":          RECOMMENDER,
        "attempts":        learner.n_updates,
        "overallRating":   round(learner.category_rating("")[0]),
        "categoryRatings": _category_ratings(learner),
        "weaknessMap":     learner.weakness_map(),
        "topWeaknesses":   learner.top_weaknesses(5),
    })


@app.route("/api/session/seen")
def session_seen():
    """IDs the player has already finished — lets the client dedupe locally."""
    username = (request.args.get("username") or "guest").strip().lower()
    if username != "guest" and not _check_token(username):
        return jsonify({"error": "Unauthorized"}), 401
    ids = sorted(_seen_ids(username))
    return jsonify({"count": len(ids), "seenIds": ids})


@app.route("/api/session/seen", methods=["DELETE"])
def session_seen_reset():
    """Forget which puzzles have been played so the pool can be replayed."""
    username = (request.args.get("username") or "guest").strip().lower()
    if username == "guest":
        GUEST_SEEN.clear()
        return jsonify({"ok": True, "cleared": True})

    if not _check_token(username):
        return jsonify({"error": "Unauthorized"}), 401

    state = _load_user_state(username)
    if not state:
        return jsonify({"ok": True, "cleared": False})
    state["attemptedPuzzleIds"] = []
    state["solvedPuzzleIds"]    = []
    state["lastUpdated"]        = _now_iso()
    _save_user_state(username, state)
    return jsonify({"ok": True, "cleared": True})


@app.route("/api/session/stats")
def session_stats():
    username = (request.args.get("username") or "guest").strip().lower()
    bandit   = SESSION_STORE.get(username)
    if bandit is None:
        return jsonify({"error": "No active session"}), 404

    return jsonify({
        "streak":        bandit.streak,
        "bestStreak":    bandit.best_streak,
        "accuracy":      round(bandit.session_accuracy() * 100, 1),
        "puzzlesPlayed": bandit.puzzles_played(),
        "weaknessMap":   _active_weakness_map(username),
        "topWeaknesses": _active_top_weaknesses(username),
        "categoryRatings": _category_ratings(_learner(username)),
    })


# ── All-time user statistics ──────────────────────────────────────────────────

@app.route("/api/user/stats/<username>")
def user_history(username: str):
    username = username.lower()
    if username == "guest":
        return jsonify({"hasHistory": False})

    if not _check_token(username):
        return jsonify({"error": "Unauthorized"}), 401

    saved = _load_user_state(username)
    if not saved:
        return jsonify({"hasHistory": False})

    history = saved.get("history", [])
    total   = len(history)
    if not total:
        return jsonify({"hasHistory": False})

    solved   = sum(1 for h in history if h.get("solved"))
    accuracy = round(solved / total * 100, 1)

    cat_stats: dict = defaultdict(lambda: {"total": 0, "solved": 0})
    for h in history:
        cat = h.get("category", "")
        if cat:
            cat_stats[cat]["total"]  += 1
            if h.get("solved"):
                cat_stats[cat]["solved"] += 1

    cat_accuracy = {
        cat: round(v["solved"] / v["total"] * 100, 1)
        for cat, v in cat_stats.items() if v["total"] >= 3
    }

    recent          = history[-20:]
    recent_accuracy = round(sum(1 for h in recent if h.get("solved")) / max(1, len(recent)) * 100, 1)

    recent_history = [
        {
            "ts":       h.get("ts", ""),
            "rating":   h.get("rating", 0),
            "solved":   bool(h.get("solved", False)),
            "category": h.get("category", ""),
        }
        for h in history[-60:]
    ]

    return jsonify({
        "hasHistory":       True,
        "totalPuzzles":     total,
        "totalSolved":      solved,
        "accuracy":         accuracy,
        "recentAccuracy":   recent_accuracy,
        "bestStreak":       saved.get("bestStreak", 0),
        "lastUpdated":      saved.get("lastUpdated", ""),
        "categoryAccuracy": cat_accuracy,
        "recentHistory":    recent_history,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _load_puzzles()
    print(f"\n  Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000, use_reloader=False)
