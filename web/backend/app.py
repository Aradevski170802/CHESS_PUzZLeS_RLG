"""
Flask micro-service — serves the puzzle frontend and provides the puzzle API.

Run from project root:
    python web/backend/app.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

FRONTEND_DIR = ROOT / "web" / "frontend"
PROCESSED_PARQUET = ROOT / "data" / "processed" / "puzzles_full.parquet"
RAW_CSV = ROOT / "DataSets" / "lichess_db_puzzle.csv"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

PUZZLE_POOL: list[dict] = []


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
    {"PuzzleId": "0009B", "FEN": "r2qr1k1/b1p2ppp/pp4n1/P1P1p3/4P1n1/B2P2Pb/3NBPPP/R2QR1K1 b - - 1 17",
     "Moves": "b6c5 e2g4 h3g4 d1g4", "Rating": 1084, "RatingDeviation": 74,
     "Popularity": 88, "NbPlays": 606, "Themes": "advantage middlegame short",
     "GameUrl": "https://lichess.org/4MWQCxQ6/black#32", "OpeningTags": None,
     "DifficultyTier": "Easy", "PrimaryCategory": "General", "Categories": []},
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
]


def _load_puzzles() -> None:
    global PUZZLE_POOL

    if PROCESSED_PARQUET.exists():
        print(f"Loading from processed parquet: {PROCESSED_PARQUET}")
        df = pd.read_parquet(PROCESSED_PARQUET)
        df = df[
            (df["Rating"].between(600, 2400))
            & (df["Popularity"] >= 60)
            & (df["NbPlays"] >= 100)
        ].copy()
        PUZZLE_POOL = df.to_dict("records")
    elif RAW_CSV.exists():
        print(f"Processed parquet not found — loading sample from CSV: {RAW_CSV}")
        df = pd.read_csv(RAW_CSV, nrows=100_000)
        df = df[
            (df["Rating"].between(600, 2400))
            & (df["Popularity"] >= 60)
            & (df["NbPlays"] >= 100)
        ].copy()
        PUZZLE_POOL = df.to_dict("records")
    else:
        print("WARNING: No puzzle data found — running in demo mode with 8 sample puzzles.")
        print("To load real puzzles: restore lichess_db_puzzle.csv to DataSets/ and re-run.")
        PUZZLE_POOL = DEMO_PUZZLES

    print(f"Puzzle pool ready: {len(PUZZLE_POOL):,} puzzles")


def _serialise(puzzle: dict) -> dict:
    themes_raw = puzzle.get("Themes", "") or ""
    themes = themes_raw.split() if isinstance(themes_raw, str) else []

    # Derive display themes (strip metadata tags)
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
        "id": puzzle["PuzzleId"],
        "fen": puzzle["FEN"],
        "moves": puzzle["Moves"].split() if isinstance(puzzle["Moves"], str) else list(puzzle["Moves"]),
        "rating": int(puzzle["Rating"]),
        "ratingDeviation": int(puzzle.get("RatingDeviation", 80)),
        "popularity": int(puzzle.get("Popularity", 80)),
        "nbPlays": int(puzzle.get("NbPlays", 0)),
        "themes": display_themes,
        "categories": categories,
        "primaryCategory": puzzle.get("PrimaryCategory", "General"),
        "difficultyTier": puzzle.get("DifficultyTier", ""),
        "gameUrl": puzzle.get("GameUrl", ""),
        "openingTags": puzzle.get("OpeningTags") or "",
    }


# ---------------------------------------------------------------------------
# Static serving
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


# ---------------------------------------------------------------------------
# Puzzle API
# ---------------------------------------------------------------------------

@app.route("/api/puzzle/random")
def puzzle_random():
    rating_min = request.args.get("ratingMin", 600, type=int)
    rating_max = request.args.get("ratingMax", 2400, type=int)
    theme = request.args.get("theme", None)

    pool = [
        p for p in PUZZLE_POOL
        if rating_min <= p["Rating"] <= rating_max
        and (theme is None or theme in (p.get("Themes") or ""))
    ]

    if not pool:
        return jsonify({"error": "No puzzles found matching filters"}), 404

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
        "source": "parquet" if PROCESSED_PARQUET.exists() else "csv_sample",
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _load_puzzles()
    print(f"\n  Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000, use_reloader=False)
