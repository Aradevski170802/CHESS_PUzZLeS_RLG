# Adaptive Chess Puzzle Advisor

MSc AI & Data Science dissertation — City College, University of York
**Aleksandar Radevski** | July–October 2026

---

## Overview

The system analyses a player's Chess.com games and works out which tactical
motifs they actually miss, measured per critical position rather than per error.
It then trains them on those motifs, with each puzzle pitched at a difficulty
the player can realistically solve.

**Pipeline:** Chess.com API → Stockfish analysis (win-probability grading,
critical-position logging, line-level tactic tagging) → opportunity-normalised
weakness profile → difficulty-aware IRT learner (Thompson Sampling over
per-category skill) → puzzle pool (Lichess DB + own-game miner) → web UI.

---

## Algorithms

| Component | File | What it does |
|---|---|---|
| **Difficulty-aware IRT learner** (default policy) | `src/recommender/irt_model.py` | Online Bayesian model `P(solve) = σ(θ + δ_k − b)`: overall ability θ, per-category offset δ_k, puzzle difficulty b. Thompson Sampling on δ picks the category; the difficulty is pitched so the predicted solve rate is ~65%. With one parameter the update reduces exactly to Glicko's (tested). |
| Beta–Bernoulli Thompson bandit | `src/recommender/bandit.py` | The original policy (`RECOMMENDER=beta`). Fractional priors, optional discounting (`BANDIT_DISCOUNT`), selection-probability logging. |
| Game analyzer | `src/classifier/stockfish_analyzer.py` | Node-limited (deterministic) MultiPV analysis. Errors are graded on win-% loss. Every critical ("only-move") position is logged as an **opportunity**, found or missed. |
| Tactic tagger | `src/puzzles/tactic_tagger.py` | Tags the engine's whole line, using its mate verdict. Agreement with Lichess themes is **58% (κ = 0.52)**, vs 31% (κ = 0.23) for the old single-move labeller, measured on a held-out sample. |
| Weakness profiler | `src/classifier/player_profiler.py` | Per-category hit rate = found / opportunities. Recent games are weighted more (25-game half-life) and bullet less, then shrunk toward the **population norm for that category** at the player's level (`src/data/category_norms.json`, fitted on the cohort; cross-validated strength ≈ 256). Priors for both policies come from this. |
| Glicko-2 fitter | `src/analysis/difficulty_fitter.py` | From-scratch Glicko-2, validated against Glickman's published example; `freeze_pool` mode; prequential hook. |
| RandomForest weakness model | `src/classifier/ml_weakness_model.py` | Opt-in (`WEAKNESS_MODEL=ml`); trained on simulated players. |
| Puzzle miner | `src/puzzles/generator.py` | Four-gate Stockfish extraction of puzzles from the player's own games. |

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `RECOMMENDER` | `irt` | `irt` = difficulty-aware IRT policy; `beta` = original Beta bandit with an Elo ± 300 band |
| `BANDIT_DISCOUNT` | `1.0` | Beta-bandit discount γ. Left at 1.0: simulation E4 shows discounting *raises* regret with 23 arms |
| `WEAKNESS_MODEL` | `rule-based` | `ml` = use the RandomForest scorer for the displayed profile |

Both models are updated on every attempt, whichever one served the puzzle.
Every attempt is logged with the serving policy, the selection propensity,
and the model's prediction *before* the outcome. That supports calibration
checks and off-policy evaluation.

---

## Research evaluation (`scripts/research/`, results in `eval/research/`)

| Script | Question | Headline result |
|---|---|---|
| `validate_labeller.py` | Do tactic labels agree with Lichess themes? | Old 31% / κ 0.23 → new 58% / κ 0.52 (held-out) |
| `run_simulation.py` | Does adaptivity help, and which policy is best? (5 experiments, 200 paired runs) | Pre-registered: bandit fails C1 and C2, passes C3. IRT-TS cuts frustrating puzzles 44% → 3% and lowers regret. Learning gain depends on the assumed learning model. |
| `evaluate_difficulty_models.py` | Prequential calibration on the real solve logs | Static Elo is badly miscalibrated (predicts 45% solved; players solve 77%) |
| `fetch_cohort.py` → `analyze_cohort.py` → `evaluate_weakness_models.py` | Which weakness model predicts a real player's *future* misses? (300 players, 60 per rating band) | Shrinking toward **population category norms** (production) and a real-data RF tie for best. The original rule-based scorer has no predictive value (AUC 0.47). Per-category reliability from ~40 games is near zero, so personal weaknesses must be learned online from puzzles. |
| `make_figures.py` | Figures | `eval/research/figures/` |

The design for a live study is in `docs/USER_STUDY_PROTOCOL.md`: a
pre-registered crossover with a power analysis (34 paired players for d = 0.5).

---

## Project Structure

```
├── data/                    # all gitignored: raw, processed, cache, sessions,
│                            #   user_puzzles, research (pseudonymised cohort)
├── docs/                    # technical report, presentation, study protocol
├── eval/
│   ├── puzzle_evaluator.py  # per-puzzle quality metrics
│   ├── run_evaluation.py    # CLI evaluation harness
│   └── research/            # labeller, simulation, calibration, cohort results
├── scripts/
│   ├── build_puzzle_dataset.py
│   └── research/            # evaluation and data-collection scripts
├── src/
│   ├── api/                 # chess_com_fetcher
│   ├── analysis/            # difficulty_fitter (Glicko-2), style_profile
│   ├── classifier/          # stockfish_analyzer, player_profiler, ml_weakness_model
│   ├── data/                # puzzle_loader, difficulty_mapper, pgn_parser
│   ├── evaluation/          # simulation environment (policies, synthetic players)
│   ├── puzzles/             # generator (miner), tactic_tagger
│   └── recommender/         # irt_model (default), bandit
├── web/
│   ├── backend/app.py       # Flask API (port 5000)
│   └── frontend/index.html  # vanilla JS SPA
└── tests/                   # pytest suite, fully offline
```

---

## Quick Start

```bash
python -m venv .venv313
.venv313\Scripts\activate          # Windows
pip install -r requirements.txt

python scripts/build_puzzle_dataset.py   # needs DataSets/lichess_db_puzzle.csv
python web/backend/app.py                # → http://localhost:5000
pytest tests/ -v                         # offline test suite

# Research evaluation
python -m scripts.research.validate_labeller --seed 11 --tag _heldout
python -m scripts.research.run_simulation --runs 200
python -m scripts.research.evaluate_difficulty_models
python -m scripts.research.make_figures
```

---

## Puzzle Quality Evaluation

```bash
python eval/run_evaluation.py --username <name> --elo <elo>
```

This scores each mined puzzle on engine agreement, clarity (PV1 − PV2 gap),
solution depth, non-triviality and difficulty fit. Reports are saved to
`eval/reports/`. The miner's current constants, from `src/puzzles/generator.py`:

| Constant | Value | Role |
|---|---|---|
| `PUZZLE_THRESHOLD` | 150 cp | Minimum eval drop for a puzzle |
| `MIN_CLARITY_CP` | 100 cp | PV1 must beat PV2 by this (no dual solutions) |
| `FORCED_OPP_MARGIN` | 80 cp | Opponent replies must be forced |
| `DETECT_TIME` | 0.10 s | Detection pass time per position |
| `VERIFY_DEPTH` | 18 | Re-verification depth |
| `CONTINUATION_MOVES` | 6 | Max half-moves after the first player move |
| `MAX_PER_GAME` | 5 | Cap per game |

---

## Data

- **Lichess puzzle DB** (~1.1 GB CSV, CC0): https://database.lichess.org/#puzzles → `DataSets/lichess_db_puzzle.csv`
- **Chess.com games:** fetched from the public API and cached in `data/cache/`.
- **Stockfish 18** (Windows x86-64): `stockfish/stockfish-windows-x86-64-avx2.exe`
- **Research cohort:** publicly available Chess.com games collected through the
  published-data API, stored pseudonymised under `data/research/` (gitignored).
  Public availability is not consent, so check your institution's research-ethics
  requirements before publishing results derived from third-party players.
