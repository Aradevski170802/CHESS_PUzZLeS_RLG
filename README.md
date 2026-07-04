# Adaptive Chess Puzzle Advisor

MSc AI & Data Science dissertation — City College, University of York  
**Aleksandar Radevski** | July–October 2026

---

## Overview

A system that analyses a player's Chess.com game history, classifies their tactical weaknesses, and then adaptively recommends puzzles to target those weaknesses using a Thompson Sampling multi-armed bandit.

**Pipeline:** Chess.com API → PGN feature extraction → Style classifier → Puzzle pool (Lichess DB + own-game miner) → Thompson Sampling recommender → Interactive web UI

---

## Project Structure

```
├── data/
│   ├── raw/              # gitignored — place lichess_db_puzzle.csv in DataSets/
│   ├── processed/        # Parquet puzzle index (gitignored)
│   ├── cache/            # Chess.com API response cache (gitignored)
│   ├── sessions/         # per-user auth tokens (gitignored)
│   └── user_puzzles/     # generated puzzles per user (gitignored)
├── eval/
│   ├── puzzle_evaluator.py  # per-puzzle quality metrics
│   ├── run_evaluation.py    # CLI evaluation harness
│   └── reports/             # saved evaluation JSON reports
├── src/
│   ├── data/             # puzzle_loader, difficulty_mapper, pgn_parser
│   ├── classifier/       # stockfish_analyzer, player_profiler
│   ├── analysis/         # style_profile — player archetype + metrics
│   ├── recommender/      # bandit.py — Thompson Sampling engine
│   ├── puzzles/          # generator.py — own-game puzzle miner
│   └── api/              # chess_com_fetcher
├── web/
│   ├── backend/          # app.py — Flask API (port 5000)
│   └── frontend/         # index.html — vanilla JS SPA
├── scripts/              # one-shot data processing
├── notebooks/            # EDA and experiments
└── tests/                # pytest test suite
```

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build the processed puzzle dataset (requires DataSets/lichess_db_puzzle.csv)
python scripts/build_puzzle_dataset.py

# 4. Start the Flask backend
python web/backend/app.py
# → running on http://localhost:5000

# 5. Serve the frontend (in a second terminal)
python -m http.server 3001 --directory web/frontend
# → open http://localhost:3001

# 6. Run tests
pytest tests/ -v
```

---

## Puzzle Quality Evaluation

After generating puzzles for a user, score their quality with Stockfish:

```bash
python eval/run_evaluation.py --username chescam_sakcs --elo 1050
```

This runs Stockfish (depth 18) on every generated puzzle and reports:

| Metric | Criterion |
|--------|-----------|
| Engine agrees | Solution matches engine's best move |
| Clarity | Eval gap between PV1 and PV2 ≥ 150 cp |
| Depth | Move sequence ≥ 3 ply (non-trivial) |
| No dual solution | Unique best move (gap ≥ 50 cp) |
| Difficulty fit | Puzzle Elo within ±300 of player Elo |

Reports are saved to `eval/reports/` as JSON for iteration comparison.

**Target quality thresholds:** `avg_score ≥ 0.70`, `engine_agrees ≥ 75%`

**Iteration loop:** Tweak constants in `src/puzzles/generator.py` → regenerate → re-run evaluator → compare.

Key constants to tune:

| Constant | Default | Effect |
|----------|---------|--------|
| `PUZZLE_THRESHOLD` | 120 cp | Raise (150–200) for harder, cleaner puzzles |
| `DETECT_TIME` | 0.05 s | Raise (0.10) for more accurate detection |
| `MIN_CLARITY_CP` | 80 cp | Raise to reduce dual-solution puzzles |
| `MIN_SOLUTION_DEPTH` | 3 | Raise to require deeper solutions |

---

## Data

**Lichess puzzle DB** (~1.1 GB CSV): not tracked in git.  
Download from: https://database.lichess.org/#puzzles  
Place at: `DataSets/lichess_db_puzzle.csv`

**Chess.com games**: fetched live via the public API and cached in `data/cache/`.

**Stockfish 18** (Windows x86-64): place binary at `stockfish/stockfish-windows-x86-64-avx2.exe`.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Core ML | Python, python-chess, Stockfish 18 |
| Recommender | Thompson Sampling (Beta distribution), custom bandit |
| Backend | Flask (Python), JSON file storage |
| Frontend | Vanilla JS SPA, no framework, Inter font |
| Puzzle data | Lichess Open Puzzle Database (Parquet), own-game miner |
| Style analysis | python-chess (no engine required) |
| Evaluation | Stockfish multi-PV, custom scoring framework |

---

## Algorithm: Thompson Sampling Bandit

23 arms, one per tactical category (Fork, Pin, Mating Pattern, Skewer, …).  
Each arm maintains `Beta(α, β)` where `α = successes + 1`, `β = failures + 1`.

**Selection:** sample θ_i ~ Beta(α_i, β_i); recommend the category with the *lowest* θ — the arm most likely to be a genuine weakness.

**Update:** `α += 1` on solve; `β += 1` on failure or timeout.

Priors are seeded from the player's game analysis: blunders in tactical positions inflate the β prior for the relevant category.

---

## Puzzle Generation

Own-game puzzles are extracted from the player's Chess.com history in two modes:

**Heuristic** (fast, no engine): detects hanging pieces, forks, checkmate-in-one, promotions using python-chess board analysis. Milliseconds per game.

**Stockfish** (thorough): finds positions where the player's move is ≥120 cp below the engine's best. Continuation lines computed at depth 16.

Quality filters applied during generation:
- Hanging-piece puzzles require ≥3-ply solution (no trivial free captures)
- Stockfish mode: clarity gap PV1 − PV2 ≥ 80 cp (prevents dual solutions)
