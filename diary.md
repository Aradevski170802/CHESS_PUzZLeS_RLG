# Project Diary — Adaptive Chess Puzzle Advisor

**Dissertation:** MSc Artificial Intelligence & Data Science  
**Institution:** City College, University of York  
**Target Grade:** Distinction (70–84)  
**Timeline:** July 1 – October 31, 2026  

---

## Overview

This project builds an **Adaptive Chess Puzzle Advisor** — a system that learns which tactical weaknesses a player has and recommends puzzles that target those weaknesses. The core algorithm is a Thompson Sampling multi-armed bandit that tracks success rates per tactic category and adjusts recommendations over time.

---

## The Dataset

**Source:** Lichess Open Puzzle Database  
**File:** `DataSets/lichess_db_puzzle.csv`  
**Size:** ~1.1 GB, 6,014,381 puzzles (before cleaning)  
**After cleaning:** 5,877,641 puzzles  

### Columns

| Column | Type | Description |
|--------|------|-------------|
| PuzzleId | string | Unique identifier (e.g. `000Pw`) |
| FEN | string | Board position in Forsyth–Edwards Notation |
| Moves | string | Space-separated UCI move sequence (opponent's first move + player's solution) |
| Rating | int | Lichess Glicko-2 puzzle rating |
| RatingDeviation | int | Uncertainty of the rating — lower = more reliable |
| Popularity | int | Community vote score (−100 to 100) |
| NbPlays | int | Total number of times the puzzle has been attempted |
| Themes | string | Space-separated Lichess theme tags (e.g. `fork middlegame short`) |
| GameUrl | string | Link to the original game on Lichess |
| OpeningTags | string | ECO opening family the puzzle arose from (if applicable) |

### Enriched Columns (added by pipeline)

| Column | Description |
|--------|-------------|
| ThemeList | `Themes` split into a Python list |
| Categories | Raw Lichess tags mapped to our 23 weakness categories |
| PrimaryCategory | Single most specific weakness label for the puzzle |
| DifficultyTier | Named tier derived from Rating |
| MoveLength | Solution length class (`oneMove`, `short`, `long`, `veryLong`) |

### Difficulty Tiers

| Tier | Rating Range | Count | Share |
|------|-------------|-------|-------|
| Beginner | < 1000 | 1,319,202 | 22.4% |
| Easy | 1000–1199 | 830,080 | 14.1% |
| Intermediate | 1200–1499 | 1,048,911 | 17.8% |
| Advanced | 1500–1799 | 985,060 | 16.8% |
| Hard | 1800–1999 | 552,032 | 9.4% |
| Expert | 2000–2199 | 453,735 | 7.7% |
| Master | 2200+ | 688,621 | 11.7% |

### 23 Weakness Categories (Bandit Arms)

Fork, Pin, Skewer, Discovered Attack, Hanging Piece, Sacrifice, Deflection, Attraction, Interference, Clearance, Quiet Move, Zugzwang, X-Ray Attack, King Safety, Mating Pattern, Endgame, Rook Endgame, Queen Endgame, Pawn Endgame, Bishop Endgame, Knight Endgame, Promotion, En Passant

---

## Session Log

---

### 2026-06-25 — Project Kickoff & Data Pipeline

#### Starting point
- Single initial git commit with only `.gitignore` and a placeholder `README.md`
- `DataSets/lichess_db_puzzle.csv` present (~1.1 GB)
- Broken `.venv` — had `python.exe` but no `pyvenv.cfg` and no installed packages
- One exploratory notebook (`Notebooks/lichess_df.ipynb`) with basic EDA only
- Empty `WebApplication/` folder

#### What was built

**Project structure** — created the full folder layout:
- `src/data/`, `src/classifier/`, `src/recommender/`, `src/miner/`, `src/api/`, `src/utils/` — Python packages
- `web/backend/routes/` and `web/frontend/{css,js}/` — web application skeleton
- `data/raw/`, `data/processed/`, `data/cache/` — data storage (raw stays gitignored)
- `scripts/`, `tests/`, `docs/` — pipeline scripts, pytest tests, architecture docs

**Virtual environment** — deleted broken `.venv`, recreated with Python 3.13.14, installed and pinned all dependencies in `requirements.txt`:
- Data & ML: pandas 2.2.3, numpy 1.26.4, scikit-learn 1.5.2, scipy 1.13.1, pyarrow 16.1.0
- Chess: python-chess 1.999, stockfish 3.28.0
- Web/API: flask 3.0.3, flask-cors 4.0.1, requests 2.32.3
- Visualisation: matplotlib 3.9.2, seaborn 0.13.2, plotly 5.22.0
- Notebooks: jupyterlab 4.2.5, ipykernel 6.29.5
- Testing: pytest 8.3.2, pytest-cov 5.0.0
- Utilities: tqdm 4.66.5, python-dotenv 1.0.1

**`src/data/puzzle_loader.py`** — core data module:
1. Loads the raw CSV with optimised dtypes (`int32`/`int16`) and optional chunked reading
2. Enriches each puzzle with `ThemeList`, `Categories`, `PrimaryCategory`, `DifficultyTier`, `MoveLength`
3. Filters by rating range, popularity, play count, categories, tiers
4. Helpers: `save_processed()` / `load_processed()` (Parquet I/O), `split_by_category()`, `get_puzzles_for_player()`

**`src/data/puzzle_difficulty_mapper.py`** — maps player ELO to 1–2 appropriate difficulty tiers using the zone of proximal development principle (puzzles rated within [player_ELO, player_ELO + 200]).

**`src/data/data_cleaner.py`** — quality filter: removes missing FEN/Moves/Rating, duplicate PuzzleIds, RatingDeviation > 150, NbPlays = 0, negative popularity, empty move strings. 136,740 puzzles removed (2.3%).

**`scripts/build_puzzle_dataset.py`** — one-shot pipeline:
1. Load 6M CSV → clean → save `data/processed/puzzles_full.parquet`
2. Split into 23 per-category Parquets (`data/processed/puzzles_by_category/`)
3. Split into 7 per-tier Parquets (`data/processed/puzzles_by_tier/`)
4. Save `data/processed/category_summary.csv`

**`tests/test_puzzle_loader.py`** — 22 unit tests, all passing. Covers: `rating_to_tier()`, `moves_to_length()`, `_tags_to_categories()`, `_enrich()`, `get_puzzles_for_player()`, `WEAKNESS_CATEGORIES`.

#### Bug caught during testing
`PrimaryCategory` was picking the first item from `Categories`, which could be a phase tag like `"Middlegame"`. Fixed by filtering to only pick from `WEAKNESS_CATEGORIES`, falling back to `"General"`.

#### Decisions
- **Parquet over CSV** for processed data — 10–20× faster to load, preserves Python list types natively
- **23 weakness categories, not raw Lichess tags** — 50+ raw tags collapse to 23 meaningful bandit arms
- **ELO-based tiers, not Lichess length tags** — `short`/`long` describe solution length, not difficulty

---

### 2026-06-25 (Evening) — Web Application: Interactive Chess Puzzle UI

#### What was built

**`web/backend/app.py`** — Flask API on `http://localhost:5000`:
- Startup: loads `puzzles_full.parquet` → raw CSV fallback (100k rows) → 10 hardcoded demo puzzles
- `GET /api/puzzle/random?ratingMin=&ratingMax=` — random puzzle within rating band
- `GET /api/puzzle/<id>` — puzzle by ID
- `GET /api/stats` — pool size and data source
- `GET /` — serves the frontend HTML

**`web/frontend/index.html`** — fully self-contained single-file app (all CSS and JS inlined):

*Design:* Dark theme (`#0f1117` background, `#c9a84c` gold accent). Header with rating filter dropdown (Beginner → Expert) and Next puzzle button. Two-column layout: board on the left, info panel on the right.

*Chess board:* 8×8 CSS grid using Unicode glyphs (♔♕♖♗♘♙♚♛♜♝♞♟). Board colours: light `#f0d9b5` / dark `#b58863` (Lichess style). Supports both click-to-move and HTML5 drag-and-drop. Legal moves shown as grey dot overlays.

*Info panel:* Puzzle identity card (ID, difficulty badge, rating, plays, theme tags) · Turn indicator · Progress bar with move dot indicators · Scrollable move history · Hint / Flip / Next controls · Solved overlay on completion.

*Puzzle flow:*
1. Fetch puzzle from Flask API (2.5s timeout) or fall back to demo pool
2. Load FEN into chess.js, determine player colour (opposite of first mover)
3. Orient board from player's perspective
4. Auto-play opponent's first move after 500ms
5. Player finds correct response; wrong moves undo immediately
6. Alternate auto-moves and player moves until puzzle complete

**`web/frontend/js/chess.min.js`** — local copy of chess.js 0.10.3 (15 KB) for move validation. No CDN dependency.

#### Problems solved

| Problem | Root cause | Solution |
|---------|-----------|----------|
| External CDN images blocked | Preview CSP | Replaced chessboard.js with Unicode glyph renderer |
| SVG data URIs blocked | Preview CSP | Abandoned SVG approach |
| Preview caches JS files | Iframe HTTP cache | Inlined all JS into `index.html` — HTML is always freshly fetched |
| Flask API returns 500 | `ndarray` not JSON serialisable | Added `.tolist()` conversion in `_serialise()` |
| Auto-move freezes UI | Demo puzzle had pinned knight (illegal move) | Replaced broken demo puzzle; `autoMove()` now calls `loadPuzzle()` on invalid move |

#### Decisions
- **Unicode pieces over images** — no external deps, works offline, styled with CSS drop-shadow
- **All JS inlined** — eliminates caching problem, makes frontend truly single-file
- **Board from player's perspective** — if player is Black, rank 1 is at the top

---

### 2026-06-26 — Dataset Processing & EDA Notebook

#### Processing pipeline run
`scripts/build_puzzle_dataset.py` executed successfully on the full 6,014,381-row CSV. Duration: 169 seconds. Output:
- `data/processed/puzzles_full.parquet` — 5,877,641 puzzles
- `data/processed/puzzles_by_category/` — 23 Parquet files
- `data/processed/puzzles_by_tier/` — 7 Parquet files
- `data/processed/category_summary.csv`

#### EDA Notebook — `Notebooks/02_dataset_deep_analysis.ipynb`

16-section deep analysis of the processed dataset using pandas, seaborn, and plotly. Key findings:

**Rating distribution:** Near-normal, mean ≈ 1500, range 399–3327. Slight right skew — more very hard puzzles than very easy ones. Beginner tier (< 1000) accounts for 22.4% of puzzles.

**Category distribution:** Endgame dominates (41.8% of puzzles by primary category), followed by General (12.8%), Mating Pattern (12.5%), King Safety (6.7%), Fork (5.2%). This reflects how chess games actually resolve — most decisive moments are endgame or mating sequences.

**Popularity:** Median popularity score ≈ 85/100. Most puzzles are well-received. Harder puzzles (Expert/Master) tend to have slightly lower popularity — likely because fewer players reach those positions.

**Move length:** The majority of puzzles have 2–4 moves in the solution. Rating correlates positively with move count — longer puzzles are harder.

**Rating deviation:** Most puzzles have RD < 100, meaning ratings are well-calibrated. The cleaner already removed RD > 150.

**Opening coverage:** ~30% of puzzles have an opening tag, meaning they arise from a specific known opening. The rest occur in positions where the opening phase is over.

**Correlation highlights:** Rating and move count have a moderate positive correlation. NbPlays and Popularity are weakly correlated — highly played puzzles are not always the most liked (early puzzles accumulate plays simply by being older).

**Quality score:** A composite score (normalised popularity × log plays × reliability penalty) was computed per puzzle. This metric will inform puzzle selection quality in the recommender.

#### Technical notes
- PySpark was initially attempted but Java is not installed — Spark requires a JVM. Replaced with pandas/seaborn/plotly which are faster for single-machine analysis.
- Plotly requires `renderer='notebook'` in VS Code; `nbformat>=4.2.0` must be installed.
- The `.venv` kernel was registered as "Python (Chess Puzzles)" for VS Code notebook use.

---

### 2026-07-02 — Adaptive Engine + Complete UI Redesign

Branch: `feature/adaptive-engine` (off `develop`). Two commits:
- `84646c7 feat: add adaptive recommendation engine`
- `9b6acce feat: complete UI redesign with login and adaptive session flow`

#### What was built

##### `src/recommender/bandit.py` — Thompson Sampling bandit

- `ArmState` dataclass: `alpha`, `beta` integers with `.mean` and `.sample()` properties
- `ThompsonBandit`: 23 arms (one per `WEAKNESS_CATEGORY`), Beta(α, β) per arm
- **Selection rule:** sample θ_i ~ Beta(α_i, β_i) for each arm; pick the arm with the *lowest* θ_i (= most likely weakness = highest training value)
- `update(category, solved)`: α += 1 on solve; β += 1 on failure; maintains streak + best_streak
- `weakness_map()`, `top_weaknesses(n)`, `session_accuracy()`, `to_dict()` / `from_dict()` for persistence
- Priors seeded from `profile_to_bandit_priors()`: weakness_score 0.8 → Beta(2,8); 0.5 → Beta(5,5); 0.2 → Beta(8,2)

##### `src/api/chess_com_fetcher.py` — Chess.com public API client (committed)

(Written in previous session, now committed after `.gitignore` fix)
- `get_player_profile(username)` → profile dict with ratings per time control
- `get_recent_games(username, n=50)` → list of PGN strings, newest first
- Disk-caching at `data/cache/chess_com/`, 0.5s delay between archive requests

##### `src/data/pgn_parser.py` — PGN parser (committed)

- `parse_game(pgn_str, username)` → dict with opening (ECO/family/URL), player colour, result, rating, num_moves
- `get_game_phase(board)` → "opening" / "middlegame" / "endgame" by piece count
- Opening family extracted from Chess.com ECOUrl slug

##### `src/classifier/stockfish_analyzer.py` — Stockfish analysis (committed)

- Was invisible due to `.gitignore: stockfish*` accidentally matching the `.py` file
- Fixed `.gitignore` to use explicit binary names (`stockfish`, `stockfish.exe`, `stockfish-*`)
- `analyze_games_parallel(pgn_strings, username, stockfish_path, workers=4)` → list[GameAnalysis]
- 50ms per position; skips first 8 half-moves; inaccuracy/mistake/blunder thresholds

##### `src/classifier/player_profiler.py` — Player profiler (committed)

- `build_profile(analyses, username, estimated_elo)` → PlayerProfile
- `profile_to_bandit_priors(profile)` → dict[str, (alpha, beta)], α+β=10
- Heuristic weakness scoring from error phase distribution + blunder rate

##### `web/backend/app.py` — 6 new Flask routes

| Route | Description |
|-------|-------------|
| `GET /api/player/lookup` | Fast Chess.com profile fetch (no analysis) |
| `POST /api/analysis/start` | Launch background analysis thread |
| `GET /api/analysis/status/<username>` | Poll analysis progress (0–100%) |
| `POST /api/session/start` | Create ThompsonBandit with optional priors |
| `GET /api/session/puzzle` | Bandit-selected category → random puzzle in that category |
| `POST /api/session/result` | Record solve/fail, update bandit, return streak/accuracy |
| `GET /api/session/stats` | Current session statistics |

Also added `PUZZLE_BY_CAT` index built at startup for O(1) category lookups. Background analysis falls back to heuristic profile if Stockfish is not installed.

##### `web/frontend/index.html` — Complete redesign (4-view SPA)

No external CSS framework. Custom CSS only, ~350 lines. Dark theme: `#0d1117` background, `#6366f1` indigo accent, `#3fb950` green, `#f85149` red. Classic Lichess brown board (64px squares = 512px).

**View 1 — Login:** Chess.com username input with Enter-key support, "Load Profile" button, "Play as Guest" button. Gradient radial background.

**View 2 — Profile:** Avatar + FIDE title badge + display name. Rating grid (Rapid/Blitz/Bullet/Daily). "Analyse my games" (triggers View 3) or "Skip" (flat priors).

**View 3 — Analysis:** Animated progress bar polling `/api/analysis/status` every 2s. Four step indicators (fetch → download → parse → build) that transition pending → active → done. "Skip analysis" abort button.

**View 4 — Game:** Fixed header with session stats (🔥 streak, ✓ accuracy %, # count), rating filter, "Next →" button, "Guest / @username" label. Two-column layout: board left, panel right.

**Training Focus card** (new): Displays the bandit's currently targeted weakness category, a colour-coded solve-rate bar (red <40%, amber <70%, green ≥70%), and the estimated solve rate as text. Shows "Random Mode" when no adaptive session is active.

**All existing game mechanics preserved:** drag-and-drop, click-to-move, auto-move, legal move dots, move history log, progress dots, hint (indigo highlight), flip, solved overlay.

**Adaptive result reporting:** After every puzzle (solved or abandoned via Next), `reportResult(category, solved)` calls `/api/session/result` and updates the header stats.

#### Bug fixed: `.gitignore` over-matching

The rule `stockfish*` was causing `stockfish_analyzer.py` to be silently excluded from `git status`. Replaced with explicit patterns: `stockfish`, `stockfish.exe`, `stockfish-*`.

---

## Pending Tasks

### Immediate
- [ ] Start Flask backend before each web app session (`python web/backend/app.py`)
- [ ] Merge `feature/adaptive-engine` into `develop`

### Short Term
- [ ] Evaluation framework: compare random vs adaptive recommendation (puzzle accuracy improvement over N sessions)
- [ ] Persist bandit state to disk (JSON) between server restarts
- [ ] Rating auto-suggest: after player lookup, set the filter to the player's ELO tier automatically
- [ ] Weakness radar chart on profile/analysis view (show all 23 category scores visually)

### Dissertation
- [ ] Write Methods chapter: Thompson Sampling algorithm, Beta distribution priors, weakness scoring
- [ ] Evaluation: record sessions, compare solve rates over time (adaptive vs baseline)
- [ ] Write Results chapter
