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
GET  /api/session/puzzle            → adaptive puzzle (bandit-selected category)
POST /api/session/result            → record solve/fail, update bandit
GET  /api/session/stats             → session accuracy, streak, weakness map
"""
from __future__ import annotations

import random
import sys
import threading
from collections import defaultdict
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.puzzle_loader import WEAKNESS_CATEGORIES
from src.recommender.bandit import ThompsonBandit

FRONTEND_DIR      = ROOT / "web" / "frontend"
PROCESSED_PARQUET = ROOT / "data" / "processed" / "puzzles_full.parquet"
RAW_CSV           = ROOT / "DataSets" / "lichess_db_puzzle.csv"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

# ── Data pools ────────────────────────────────────────────────────────────────
PUZZLE_POOL:   list[dict] = []
PUZZLE_BY_CAT: dict[str, list[dict]] = {}   # PrimaryCategory → [puzzle dicts]

# ── In-memory state ───────────────────────────────────────────────────────────
# username (lowercase) → {status, progress, message, priors, profile}
ANALYSIS_STORE: dict[str, dict] = {}
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
    {"PuzzleId": "007hK", "FEN": "8/8/8/8/3k4/8/3KR3/8 w - - 0 1",
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

def _load_puzzles() -> None:
    global PUZZLE_POOL
    if PROCESSED_PARQUET.exists():
        print(f"Loading from parquet: {PROCESSED_PARQUET}")
        df = pd.read_parquet(PROCESSED_PARQUET)
        df = df[
            (df["Rating"].between(600, 2400))
            & (df["Popularity"] >= 60)
            & (df["NbPlays"] >= 100)
        ].copy()
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
    return jsonify(_serialise(random.choice(pool)))


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
            ANALYSIS_STORE[username].update({
                "status":   "done",
                "progress": 100,
                "message":  "Analysis complete!",
                "priors":   {k: list(v) for k, v in priors.items()},
                "profile":  _serialise_profile(profile),
            })

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
    # Don't serialise full puzzle history in the response
    return jsonify({k: v for k, v in state.items() if k != "history"})


# ── Adaptive session ──────────────────────────────────────────────────────────

@app.route("/api/session/start", methods=["POST"])
def session_start():
    data     = request.get_json(force=True) or {}
    username = (data.get("username") or "guest").strip().lower()
    priors_raw = data.get("priors")  # {category: [alpha, beta]} or None

    priors = None
    if priors_raw:
        priors = {cat: (int(v[0]), int(v[1])) for cat, v in priors_raw.items()
                  if isinstance(v, (list, tuple)) and len(v) == 2}

    bandit = ThompsonBandit(priors=priors)
    SESSION_STORE[username] = bandit

    return jsonify({
        "status":       "ok",
        "weaknessMap":  bandit.weakness_map(),
        "topWeaknesses": bandit.top_weaknesses(5),
    })


@app.route("/api/session/puzzle")
def session_puzzle():
    username   = (request.args.get("username") or "guest").strip().lower()
    rating_min = request.args.get("ratingMin", 600,  type=int)
    rating_max = request.args.get("ratingMax", 2400, type=int)

    bandit = SESSION_STORE.get(username)
    if bandit is None:
        bandit = ThompsonBandit()
        SESSION_STORE[username] = bandit

    target = bandit.select_one()
    solve_rate = bandit.solve_rate(target)

    # Try target category first, then fall back to any puzzle in range
    cat_pool = [p for p in PUZZLE_BY_CAT.get(target, [])
                if rating_min <= p["Rating"] <= rating_max]

    if not cat_pool:
        fallback_pool = [p for p in PUZZLE_POOL if rating_min <= p["Rating"] <= rating_max]
        if not fallback_pool:
            return jsonify({"error": "No puzzles found in this rating range"}), 404
        chosen = random.choice(fallback_pool)
        target = chosen.get("PrimaryCategory", "General")
    else:
        chosen = random.choice(cat_pool)

    result = _serialise(chosen)
    result["targetCategory"] = target
    result["targetSolveRate"] = round(solve_rate, 3)
    return jsonify(result)


@app.route("/api/session/result", methods=["POST"])
def session_result():
    data     = request.get_json(force=True) or {}
    username = (data.get("username") or "guest").strip().lower()
    category = data.get("category", "")
    solved   = bool(data.get("solved", False))

    bandit = SESSION_STORE.get(username)
    if bandit is None:
        bandit = ThompsonBandit()
        SESSION_STORE[username] = bandit

    if category:
        bandit.update(category, solved)

    return jsonify({
        "streak":       bandit.streak,
        "bestStreak":   bandit.best_streak,
        "accuracy":     round(bandit.session_accuracy() * 100, 1),
        "puzzlesPlayed": bandit.puzzles_played(),
        "weaknessMap":  bandit.weakness_map(),
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


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _load_puzzles()
    print(f"\n  Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000, use_reloader=False)
