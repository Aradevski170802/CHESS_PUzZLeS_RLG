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
| **PuzzleNet** (neural tactic + difficulty model) | `src/neural/` | Multi-task network trained from scratch (NumPy, no framework) on 5.5M Lichess puzzles. Reads a position and the line that follows it; predicts the tactic category, 66 Lichess theme tags, and the difficulty **with an uncertainty**. The rating loss treats each puzzle's Lichess rating deviation as known label noise, so the learned variance is content uncertainty — which the IRT learner then uses as a mined puzzle's rating deviation. |
| Tactic tagger | `src/puzzles/tactic_tagger.py` | Tags the engine's whole line, using its mate verdict. Agreement with Lichess themes is **58% (κ = 0.52)**, vs 31% (κ = 0.23) for the old single-move labeller, measured on a held-out sample. Still the fallback whenever no model is installed or `LABELLER=rules`. |
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
| `LABELLER` | `neural` | `rules` = label critical positions with the hand-written tagger instead of PuzzleNet |
| `PUZZLENET` | `on` | `off` disables the network everywhere (miner and analyzer fall back to the formula and the rule tagger) |
| `PUZZLENET_MODEL` | `src/data/models/puzzlenet.npz` | Path to the trained model |
| `MINED_RATING` | `heuristic` | `puzzlenet` = serve mined puzzles at the network's predicted difficulty. Off by default: on the real logs the network rates mined puzzles 493 points harder than the formula, yet players solve 86.7% of them (they are positions from their own games), and prequential log loss is 0.599 (formula) vs 0.754 (network). The estimate is still recorded on every mined puzzle as `puzzlenetRating`/`puzzlenetRd` |

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
| `evaluate_difficulty_models.py` | Prequential calibration on the real solve logs | Static Elo is badly miscalibrated (predicts 45% solved; players solve 84.6%) |
| `classify_rating_band.py` | Objective 1: predict a player's rating band from how they play (300 players, rating fields excluded) | 59.1% accuracy over 5 bands (chance 20%), 93% within one band, permutation p = 0.002; rating predicted within 206 points (r = 0.90) |
| `fetch_cohort.py` → `analyze_cohort.py` → `evaluate_weakness_models.py` | Which weakness model predicts a real player's *future* misses? (300 players, 60 per rating band) | Shrinking toward **population category norms** (production) and a real-data RF tie for best. The original rule-based scorer has no predictive value (AUC 0.47). Per-category reliability from ~40 games is near zero, so personal weaknesses must be learned online from puzzles. |
| `make_figures.py` | Figures | `eval/research/figures/` |

### PuzzleNet (`scripts/neural/`, results in `eval/neural/`)

| Script | Question | Headline result |
|---|---|---|
| `build_dataset.py` | Encode the puzzle database for training | 5,877,641 puzzles, 0 failures, ~50 min; the held-out labeller sample (59,668 puzzles) is excluded from training |
| `run_experiments.py` | Train the model, 9 ablations and a learning curve | see `eval/neural/training/` |
| `evaluate_puzzlenet.py` | Is the network a better labeller, and a better difficulty model? | On the **same held-out puzzles** as the rule tagger: agreement 58.4% -> **92.6%**, kappa 0.52 -> **0.91**, macro recall 0.32 -> **0.86**, and all 24 categories emitted (the rules can never emit 5). Difficulty: RMSE **268** (rating SD 550), R2 0.76, and calibrated intervals (50/80/95% cover 50/81/95%) |
| `engine_pv_check.py` | Does it hold up on Stockfish lines, as in game analysis? | On Stockfish lines, as in game analysis, kappa **0.78** vs **0.49** for the rules; 5 plies is the best cut, so that is what the analyzer now passes |
| `run_label_simulation.py` | Does better labelling help the recommender downstream? | Yes: early weak-category targeting 0.226 (rules) -> **0.306** (PuzzleNet) -> 0.321 (perfect labels); the gap to perfect is no longer significant (p = 0.31) |
| `relabel_cohort.py` | Do better labels make 300 real players' weaknesses measurable? | No: the labels change a lot (the two labellers agree on only 37% of real game positions) but split-half reliability stays ~0. Label noise was **not** why weaknesses looked like noise |

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
