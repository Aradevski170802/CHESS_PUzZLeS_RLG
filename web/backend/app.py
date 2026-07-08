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
GET  /api/session/puzzle            → adaptive puzzle (bandit-selected, rating-matched, skips solved)
GET  /api/eval                      → Stockfish evaluation for a FEN position (eval bar)
POST /api/session/result            → record solve/fail, update bandit, persist to disk
GET  /api/session/stats             → session accuracy, streak, weakness map
GET  /api/user/stats/<user>         → all-time stats from persisted history (auth required)
POST /api/auth/challenge            → issue Chess.com ownership verification code
POST /api/auth/verify-chess         → check Chess.com location field contains code
POST /api/auth/register             → create account (requires Chess.com verification)
POST /api/auth/login                → verify password, return 30-day session token
GET  /api/auth/check                → validate a stored token (for auto-login)
"""
from __future__ import annotations

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
from flask_cors import CORS

try:
    import chess as _pychess
    _PYCHESS_OK = True
except ImportError:
    _pychess = None
    _PYCHESS_OK = False

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.puzzle_loader import WEAKNESS_CATEGORIES
from src.recommender.bandit import ThompsonBandit

FRONTEND_DIR      = ROOT / "web" / "frontend"
PROCESSED_PARQUET = ROOT / "data" / "processed" / "puzzles_full.parquet"
RAW_CSV           = ROOT / "DataSets" / "lichess_db_puzzle.csv"
SESSIONS_DIR      = ROOT / "data" / "sessions"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

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

# ── Demo fallback puzzles ─────────────────────────────────────────────────────
DEMO_PUZZLES = [
    {"PuzzleId": "00008", "FEN": "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 1",
     "Moves": "f2g3 e6e7 b2b1 b3c1 b1c1 h6c1", "Rating": 1862, "RatingDeviation": 76,
     "Popularity": 95, "NbPlays": 9697, "Themes": "crushing hangingPiece long middlegame",
     "GameUrl": "https://lichess.org/787zsVup/black#48", "OpeningTags": None,
     "DifficultyTier": "Hard", "PrimaryCategory": "Hanging Piece", "Categories": ["Hanging Piece"]},
    {"PuzzleId": "0000D", "FEN": "5rk1/1p3ppp/pq3b2/8/8/1P1Q1N2/P4PPP/3R2K1 w - - 1 26",
     "Moves": "d3d6 f8d8 d6d8 f6d8", "Rating": 1579, "RatingDeviation": 73,
     "Popularity": 96, "NbPlays": 36672, "Themes": "advantage endgame short",
     "GameUrl": "https://lichess.org/F8M8OS71#53", "OpeningTags": None,
     "DifficultyTier": "Advanced", "PrimaryCategory": "Endgame", "Categories": ["Endgame"]},
    {"PuzzleId": "000Pw", "FEN": "6k1/5p1p/4p3/4q3/3nN3/2Q3P1/PP3P1P/6K1 w - - 2 37",
     "Moves": "e4d2 d4e2 g1f1 e2c3", "Rating": 1550, "RatingDeviation": 75,
     "Popularity": 92, "NbPlays": 626, "Themes": "crushing endgame fork short",
     "GameUrl": "https://lichess.org/au2lCK5o#73", "OpeningTags": None,
     "DifficultyTier": "Advanced", "PrimaryCategory": "Fork", "Categories": ["Fork", "Endgame"]},
    {"PuzzleId": "001aB", "FEN": "2rq1rk1/pb1nbppp/1p2p3/3pP3/3P1P2/Q1PB1N2/P4PPP/R1B1R1K1 b - - 0 1",
     "Moves": "d7c5 d3h7 g8h7 q3h3 h7g8 h3h8", "Rating": 1900, "RatingDeviation": 80,
     "Popularity": 88, "NbPlays": 1200, "Themes": "crushing sacrifice middlegame long",
     "GameUrl": "https://lichess.org/example1#40", "OpeningTags": None,
     "DifficultyTier": "Hard", "PrimaryCategory": "Sacrifice", "Categories": ["Sacrifice"]},
    {"PuzzleId": "002mP", "FEN": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
     "Moves": "f3e5 f6e4 d1f3 e4f2 f3f7", "Rating": 1320, "RatingDeviation": 72,
     "Popularity": 93, "NbPlays": 15000, "Themes": "mateIn2 middlegame short",
     "GameUrl": "https://lichess.org/example2#8", "OpeningTags": "Italian_Game",
     "DifficultyTier": "Intermediate", "PrimaryCategory": "Mating Pattern", "Categories": ["Mating Pattern"]},
    {"PuzzleId": "003xY", "FEN": "6k1/pp3p1p/2p3p1/2b5/2Bn4/1P4P1/P4P1P/3R2K1 b - - 0 25",
     "Moves": "d4f3 g1f1 c5e3 d1d8", "Rating": 1450, "RatingDeviation": 78,
     "Popularity": 85, "NbPlays": 3400, "Themes": "crushing fork endgame short",
     "GameUrl": "https://lichess.org/example3#50", "OpeningTags": None,
     "DifficultyTier": "Intermediate", "PrimaryCategory": "Fork", "Categories": ["Fork", "Endgame"]},
    {"PuzzleId": "004rZ", "FEN": "r4rk1/1pp2ppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP2PPP/R2QR1K1 w - - 0 10",
     "Moves": "g5f6 g7f6 c4f7 g8h8 d1d3", "Rating": 1700, "RatingDeviation": 82,
     "Popularity": 90, "NbPlays": 5600, "Themes": "crushing sacrifice middlegame long",
     "GameUrl": "https://lichess.org/example4#20", "OpeningTags": None,
     "DifficultyTier": "Hard", "PrimaryCategory": "Sacrifice", "Categories": ["Sacrifice"]},
    {"PuzzleId": "007hK", "FEN": "8/8/8/8/3k4/8/4R1K1/8 w - - 0 1",
     "Moves": "e2e4 d4d3 e4e3", "Rating": 950, "RatingDeviation": 80,
     "Popularity": 85, "NbPlays": 2100, "Themes": "rookEndgame endgame",
     "GameUrl": "", "OpeningTags": None,
     "DifficultyTier": "Easy", "PrimaryCategory": "Rook Endgame", "Categories": ["Rook Endgame"]},
    {"PuzzleId": "008aB", "FEN": "r2qkb1r/pp3ppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP3PPP/RN1QK2R b KQkq - 0 8",
     "Moves": "c5d4 c3d4 f6e4 d4e5 d8a5", "Rating": 1620, "RatingDeviation": 76,
     "Popularity": 89, "NbPlays": 5300, "Themes": "hangingPiece middlegame",
     "GameUrl": "", "OpeningTags": None,
     "DifficultyTier": "Advanced", "PrimaryCategory": "Hanging Piece", "Categories": ["Hanging Piece"]},
    {"PuzzleId": "009kT", "FEN": "r1b2rk1/pp2ppbp/2np1np1/q7/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 3 10",
     "Moves": "d4c6 b7c6 d2a5 d8a5", "Rating": 1250, "RatingDeviation": 74,
     "Popularity": 88, "NbPlays": 7200, "Themes": "hangingPiece middlegame",
     "GameUrl": "", "OpeningTags": None,
     "DifficultyTier": "Intermediate", "PrimaryCategory": "Hanging Piece", "Categories": ["Hanging Piece"]},
    {"PuzzleId": "010vR", "FEN": "r2q1rk1/pp1bppbp/3p1np1/3P4/2P1PP2/2N5/PP1QB1PP/R3K2R b KQ - 0 13",
     "Moves": "f6d5 c3d5 g7d4 d2d4", "Rating": 1680, "RatingDeviation": 79,
     "Popularity": 86, "NbPlays": 3900, "Themes": "fork middlegame",
     "GameUrl": "", "OpeningTags": None,
     "DifficultyTier": "Advanced", "PrimaryCategory": "Fork", "Categories": ["Fork"]},
    {"PuzzleId": "011pK", "FEN": "r3k2r/ppp2ppp/2n1bn2/3qp3/3P4/2N1PN2/PPP1BPPP/R2QK2R b KQkq - 0 9",
     "Moves": "d5d4 c3b5 d4b2 b5c7 e8d8 c7a8", "Rating": 1780, "RatingDeviation": 77,
     "Popularity": 87, "NbPlays": 4200, "Themes": "pin middlegame long",
     "GameUrl": "", "OpeningTags": None,
     "DifficultyTier": "Hard", "PrimaryCategory": "Pin", "Categories": ["Pin"]},
]


# ── Puzzle loading ─────────────────────────────────────────────────────────────

_PARQUET_COLS = [
    "PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation",
    "Popularity", "NbPlays", "Themes", "GameUrl", "OpeningTags",
    "PrimaryCategory", "DifficultyTier", "Categories",
]
_POOL_CAP = 200_000   # puzzles kept in RAM


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
        # Guess target_fraction conservatively (upper-bound after filter ≈ 80 %)
        target_per_rg = max(1, _POOL_CAP // n_rg)
        avail_cols = pf.schema_arrow.names
        cols = [c for c in _PARQUET_COLS if c in avail_cols]

        chunks: list[pd.DataFrame] = []
        for i in range(n_rg):
            rg = pf.read_row_group(i, columns=cols).to_pandas()
            rg = rg[
                rg["Rating"].between(600, 2400)
                & (rg["Popularity"] >= 60)
                & (rg["NbPlays"] >= 100)
            ]
            if len(rg) > target_per_rg:
                rg = rg.sample(target_per_rg, random_state=42 + i)
            chunks.append(rg)

        df = pd.concat(chunks, ignore_index=True)
        if len(df) > _POOL_CAP:
            df = df.sample(_POOL_CAP, random_state=42)
        PUZZLE_POOL = df.to_dict("records")
    elif RAW_CSV.exists():
        print(f"Loading sample from CSV: {RAW_CSV}")
        df = pd.read_csv(RAW_CSV, nrows=100_000)
        df = df[
            (df["Rating"].between(600, 2400))
            & (df["Popularity"] >= 60)
            & (df["NbPlays"] >= 100)
        ].copy()
        PUZZLE_POOL = df.to_dict("records")
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

    return {
        "id":             puzzle["PuzzleId"],
        "fen":            puzzle["FEN"],
        "moves":          puzzle["Moves"].split() if isinstance(puzzle["Moves"], str) else list(puzzle["Moves"]),
        "rating":         int(puzzle["Rating"]),
        "ratingDeviation": int(puzzle.get("RatingDeviation", 80)),
        "popularity":     int(puzzle.get("Popularity", 80)),
        "nbPlays":        int(puzzle.get("NbPlays", 0)),
        "themes":         display_themes,
        "categories":     categories,
        "primaryCategory": puzzle.get("PrimaryCategory", "General"),
        "difficultyTier": puzzle.get("DifficultyTier", ""),
        "gameUrl":        puzzle.get("GameUrl", ""),
        "openingTags":    puzzle.get("OpeningTags") or "",
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
    return {
        "username":       profile.username,
        "estimatedElo":   profile.estimated_elo,
        "gamesAnalysed":  profile.games_analysed,
        "gamesWon":       profile.games_won,
        "gamesLost":      profile.games_lost,
        "gamesDrawn":     profile.games_drawn,
        "winRate":        round(profile.win_rate * 100, 1),
        "weaknessScores": {k: round(v, 3) for k, v in profile.weakness_scores.items()},
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
            import shutil
            from src.api.chess_com_fetcher import get_player_profile, get_recent_games, get_best_rating
            from src.data.pgn_parser import parse_games_bulk
            from src.classifier.player_profiler import build_profile, profile_to_bandit_priors

            ANALYSIS_STORE[username]["message"] = "Fetching profile from Chess.com…"
            ch_profile = get_player_profile(username)
            elo = get_best_rating(ch_profile)
            ANALYSIS_STORE[username]["progress"] = 10

            ANALYSIS_STORE[username]["message"] = "Fetching recent games…"
            pgns = get_recent_games(username, n=50)
            ANALYSIS_STORE[username]["progress"] = 35

            profile = None
            sf_path = shutil.which("stockfish")  # None if not in PATH

            if sf_path:
                try:
                    from src.classifier.stockfish_analyzer import analyze_games_parallel
                    ANALYSIS_STORE[username]["message"] = f"Running Stockfish on {len(pgns)} games…"
                    analyses = analyze_games_parallel(pgns, username, sf_path, workers=2)
                    ANALYSIS_STORE[username]["progress"] = 85
                    successful = [a for a in analyses if not getattr(a, "failed", True)]
                    if successful:
                        profile = build_profile(analyses, username, estimated_elo=elo)
                except Exception:
                    pass  # fall through to heuristic

            if profile is None:
                ANALYSIS_STORE[username]["message"] = "Building weakness profile (heuristic)…"
                parsed = [g for g in parse_games_bulk(pgns, username) if g]
                ANALYSIS_STORE[username]["progress"] = 60
                profile = _heuristic_profile(parsed, username)

            ANALYSIS_STORE[username]["progress"] = 95
            priors = profile_to_bandit_priors(profile)
            serialised = _serialise_profile(profile)
            ANALYSIS_STORE[username].update({
                "status":   "done",
                "progress": 100,
                "message":  "Analysis complete!",
                "priors":   {k: list(v) for k, v in priors.items()},
                "profile":  serialised,
            })
            # Persist profile + ELO so the dashboard reloads on next visit
            if username != "guest":
                _ustate = _load_user_state(username) or {
                    "username": username, "history": [], "bestStreak": 0,
                }
                _ustate["profile"]      = serialised
                _ustate["estimatedElo"] = profile.estimated_elo
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
    data     = request.get_json(force=True) or {}
    username = (data.get("username") or "").strip().lower()
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

    GENERATE_STORE[username] = {"status": "running", "progress": 0, "message": "Starting…", "count": 0}

    def _run():
        try:
            from src.api.chess_com_fetcher import get_recent_games
            from src.puzzles.generator import generate_from_games, save_user_puzzles

            GENERATE_STORE[username]["message"] = "Fetching recent games from Chess.com…"
            pgns = get_recent_games(username, n=40)
            if not pgns:
                GENERATE_STORE[username].update({
                    "status": "error",
                    "message": "No games found on Chess.com for this account. "
                               "Make sure your game history is public.",
                })
                return
            GENERATE_STORE[username]["progress"] = 10
            GENERATE_STORE[username]["message"] = (
                f"Analysing {len(pgns)} games with Stockfish (quality mode)…"
            )

            def _cb(done, total, found=0):
                pct = 10 + int(done / total * 85)
                GENERATE_STORE[username]["progress"] = pct
                found_str = f" — {found} found" if found else ""
                GENERATE_STORE[username]["message"] = (
                    f"Game {done}/{total}{found_str}…"
                )

            puzzles = generate_from_games(pgns, username, sf_path,
                                          max_total=30, progress_callback=_cb)

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
            priors = {cat: (int(v[0]), int(v[1])) for cat, v in priors_raw.items()
                      if isinstance(v, (list, tuple)) and len(v) == 2}
        bandit = ThompsonBandit(priors=priors)

    SESSION_STORE[username] = bandit

    # Persist ELO and initialise state file for first-time users
    if username != "guest" and estimated_elo:
        _s = saved or {"username": username, "history": [], "bestStreak": 0}
        _s["estimatedElo"] = int(estimated_elo)
        _s["lastUpdated"]  = _now_iso()
        if not returning:
            _s["bandit"] = bandit.to_dict()
        _save_user_state(username, _s)

    return jsonify({
        "status":        "ok",
        "returning":     returning,
        "weaknessMap":   bandit.weakness_map(),
        "topWeaknesses": bandit.top_weaknesses(5),
    })


@app.route("/api/eval")
def position_eval():
    """Quick Stockfish evaluation of a FEN position for the eval bar."""
    fen = request.args.get("fen", "").strip()
    if not fen:
        return jsonify({"error": "Missing fen"}), 400

    from src.classifier.stockfish_analyzer import find_stockfish
    sf_path = find_stockfish()
    if not sf_path:
        return jsonify({"error": "Stockfish not available"}), 503

    try:
        import chess
        import chess.engine
        board = chess.Board(fen)
        with chess.engine.SimpleEngine.popen_uci(sf_path) as engine:
            info = engine.analyse(board, chess.engine.Limit(depth=14))
        score = info["score"].white()
        if score.is_mate():
            m = score.mate()
            return jsonify({"mate": m, "cp": None,
                            "advantage": "white" if (m or 0) > 0 else "black"})
        cp = score.score()
        return jsonify({"cp": cp, "mate": None,
                        "advantage": "white" if (cp or 0) > 0 else "black" if (cp or 0) < 0 else "equal"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/session/puzzle")
def session_puzzle():
    username   = (request.args.get("username") or "guest").strip().lower()
    rating_min = request.args.get("ratingMin", 600,  type=int)
    rating_max = request.args.get("ratingMax", 2400, type=int)
    elo        = request.args.get("elo", type=int)

    # Narrow the rating window around the player's known ELO (±300)
    if elo:
        rating_min = max(rating_min, elo - 300)
        rating_max = min(rating_max, elo + 300)

    bandit = SESSION_STORE.get(username)
    if bandit is None:
        bandit = ThompsonBandit()
        SESSION_STORE[username] = bandit

    target = bandit.select_one()
    solve_rate = bandit.solve_rate(target)

    # Load solved puzzle IDs so we never repeat a solved puzzle
    saved = _load_user_state(username) if username != "guest" else None
    solved_ids = set((saved or {}).get("solvedPuzzleIds", []))

    # Blend user-generated puzzles (priority) with Lichess pool
    from src.puzzles.generator import load_user_puzzles
    user_puzzles = load_user_puzzles(username) if username != "guest" else []

    def _in_range(p):
        return rating_min <= p.get("Rating", 1200) <= rating_max

    def _not_solved(p):
        return p.get("PuzzleId") not in solved_ids

    chosen = None

    if user_puzzles:
        user_cat = [p for p in user_puzzles
                    if p.get("PrimaryCategory") == target and _in_range(p) and _not_solved(p)]
        if user_cat:
            chosen = _pick_valid(user_cat) or random.choice(user_cat)
            chosen = dict(chosen, source="generated")
        elif random.random() < 0.4:
            user_range = [p for p in user_puzzles if _in_range(p) and _not_solved(p)]
            if user_range:
                chosen = _pick_valid(user_range) or random.choice(user_range)
                chosen = dict(chosen, source="generated")
                target = chosen.get("PrimaryCategory", target)

    # Fall back to Lichess pool
    if chosen is None:
        cat_pool = [p for p in PUZZLE_BY_CAT.get(target, [])
                    if _in_range(p) and _not_solved(p)]
        if not cat_pool:
            fallback_pool = [p for p in PUZZLE_POOL if _in_range(p) and _not_solved(p)]
            if not fallback_pool:
                # All puzzles in range solved — broaden and allow repeats
                fallback_pool = [p for p in PUZZLE_POOL if rating_min <= p["Rating"] <= rating_max]
            if not fallback_pool:
                return jsonify({"error": "No puzzles found in this rating range"}), 404
            chosen = _pick_valid(fallback_pool) or random.choice(fallback_pool)
            target = chosen.get("PrimaryCategory", "General")
        else:
            chosen = _pick_valid(cat_pool) or random.choice(cat_pool)

    result = _serialise(chosen)
    result["targetCategory"] = target
    result["targetSolveRate"] = round(solve_rate, 3)
    return jsonify(result)


@app.route("/api/session/result", methods=["POST"])
def session_result():
    data      = request.get_json(force=True) or {}
    username  = (data.get("username") or "guest").strip().lower()
    category  = data.get("category", "")
    solved    = bool(data.get("solved", False))
    puzzle_id = data.get("puzzleId", "")
    rating    = int(data.get("rating", 0)) if data.get("rating") else 0

    bandit = SESSION_STORE.get(username)
    if bandit is None:
        bandit = ThompsonBandit()
        SESSION_STORE[username] = bandit

    if category:
        bandit.update(category, solved)

    # Persist after every result (skip guests)
    if username != "guest":
        _s = _load_user_state(username) or {
            "username": username, "history": [], "bestStreak": 0,
        }
        _s["lastUpdated"] = _now_iso()
        _s["bandit"]      = bandit.to_dict()
        _s["bestStreak"]  = max(_s.get("bestStreak", 0), bandit.best_streak)
        _s.setdefault("history", []).append({
            "ts":       _now_iso(),
            "puzzleId": puzzle_id,
            "category": category,
            "rating":   rating,
            "solved":   solved,
        })
        if solved and puzzle_id:
            ids = set(_s.get("solvedPuzzleIds", []))
            ids.add(puzzle_id)
            _s["solvedPuzzleIds"] = list(ids)
        _save_user_state(username, _s)

    return jsonify({
        "streak":        bandit.streak,
        "bestStreak":    bandit.best_streak,
        "accuracy":      round(bandit.session_accuracy() * 100, 1),
        "puzzlesPlayed": bandit.puzzles_played(),
        "weaknessMap":   bandit.weakness_map(),
        "topWeaknesses": bandit.top_weaknesses(5),
    })


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
        "weaknessMap":   bandit.weakness_map(),
        "topWeaknesses": bandit.top_weaknesses(5),
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
