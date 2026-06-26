# Adaptive Chess Puzzle Advisor

MSc AI & Data Science dissertation — City College, University of York  
**Aleksandar Radevski** | July–October 2026

---

## Overview

A system that classifies a player's chess style from their Chess.com game history, then adaptively recommends puzzles targeting their weak tactical themes using Thompson Sampling (multi-armed bandit RL).

**Pipeline:** Chess.com API → PGN feature extraction → Style classifier → Puzzle pool (Lichess DB + own-game miner) → Adaptive recommender → Interactive web UI

---

## Project Structure

```
├── data/
│   ├── raw/          # gitignored — place lichess_db_puzzle.csv in DataSets/
│   ├── processed/    # generated Parquet files (gitignored)
│   └── cache/        # Chess.com API cache (gitignored)
├── src/
│   ├── data/         # puzzle_loader, difficulty_mapper, data_cleaner
│   ├── classifier/   # feature_extractor, pgn_parser, player_profiler, style_classifier
│   ├── recommender/  # thompson sampling recommender, session_tracker
│   ├── miner/        # puzzle_miner (from own games), stockfish_annotator
│   ├── api/          # chess_com_fetcher, flask_service
│   └── utils/        # shared helpers
├── web/
│   ├── backend/      # Node.js Express API
│   └── frontend/     # chessboard.js puzzle UI
├── scripts/          # one-shot data processing scripts
├── notebooks/        # EDA and experiments
├── tests/            # pytest test suite
└── docs/             # architecture diagrams, design notes
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

# 4. Run tests
pytest tests/ -v
```

---

## Data

The raw Lichess puzzle database (~1.1 GB CSV) is not tracked in git.  
Download from: https://database.lichess.org/#puzzles  
Place at: `DataSets/lichess_db_puzzle.csv`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data & ML | Python, pandas, scikit-learn, python-chess, Stockfish |
| Backend | Node.js (Express) + Python (Flask micro-service) |
| Frontend | HTML/CSS, chess.js, chessboard.js, Chart.js |
| Storage | SQLite (sessions), Parquet (puzzle index) |
