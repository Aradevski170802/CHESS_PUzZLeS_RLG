"""
Fits player and puzzle difficulty ratings from this system's own solve logs
(`data/sessions/*.json`), using the Glicko-2 rating system — the same system
Lichess uses for its puzzle ratings (see EVALUATION_METHODOLOGY.md / references
[12][13]). This replaces reliance on the static Lichess `Rating` field with a
rating that reflects how puzzles and players actually perform *in this app*.

Design, stated plainly (see IMPROVEMENT_PLAN.md §3.2):
  - Every solved/failed puzzle attempt is treated as a single-game "rating
    period" between the player and the puzzle (the puzzle is the "opponent"),
    exactly mirroring how bandit.py's own docstring already describes puzzle
    attempts ("every attempt is a game between solver and puzzle").
  - Events are replayed in chronological order across ALL users, since puzzle
    ratings are shared/global while player ratings are per-player.
  - A puzzle's rating is seeded from its Lichess `Rating` field the first time
    it is attempted (an informative prior), then evolves from real outcomes.
  - A player's rating is seeded from their Chess.com `estimatedElo` if present
    in their session file, else a neutral 1500 default.

This module is intentionally NOT wired into web/backend/app.py yet — run it,
inspect data/processed/fitted_ratings.json, and confirm the output looks sane
before deciding how (or whether) to blend it into puzzle selection.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

GLICKO2_SCALE = 173.7178
DEFAULT_TAU = 0.5             # system constant controlling volatility change; 0.3-1.2 is the
                               # normal range recommended by Glickman, 0.5 is his own example value
DEFAULT_PLAYER_RATING = 1500.0
DEFAULT_PLAYER_RD = 350.0     # full uncertainty for a genuinely new, unseeded player
SEEDED_PLAYER_RD = 200.0      # moderate uncertainty when we do have a Chess.com Elo to seed from
PUZZLE_SEED_RD = 200.0        # moderate uncertainty: a Lichess Rating is real signal, not a guess
POOL_PUZZLE_RD = 75.0         # a Lichess rating fitted from thousands of attempts
MINED_PUZZLE_RD = 300.0       # own-game puzzle: heuristic rating only
DEFAULT_VOLATILITY = 0.06     # Glickman's own recommended default


def is_mined_puzzle(puzzle_id: str) -> bool:
    """Puzzles mined from the player's own games are id'd "gen_<hash>"."""
    return str(puzzle_id).startswith("gen_")


# ─────────────────────────────────────────────────────────────────────────────
# Core Glicko-2 rating type and math
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Glicko2Rating:
    rating: float = DEFAULT_PLAYER_RATING
    rd: float = DEFAULT_PLAYER_RD
    volatility: float = DEFAULT_VOLATILITY

    def to_internal(self) -> tuple[float, float]:
        """Convert to the Glicko-2 internal (mu, phi) scale."""
        mu = (self.rating - 1500.0) / GLICKO2_SCALE
        phi = self.rd / GLICKO2_SCALE
        return mu, phi

    @staticmethod
    def from_internal(mu: float, phi: float, sigma: float) -> "Glicko2Rating":
        rating = GLICKO2_SCALE * mu + 1500.0
        rd = GLICKO2_SCALE * phi
        return Glicko2Rating(rating=rating, rd=rd, volatility=sigma)

    def to_dict(self) -> dict:
        return {
            "rating": round(self.rating, 2),
            "rd": round(self.rd, 2),
            "volatility": round(self.volatility, 5),
        }


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi ** 2 / math.pi ** 2)


def _e(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def _volatility_objective(x: float, delta: float, phi: float, v: float, a_const: float, tau: float) -> float:
    """f(x) from Glickman's Glicko-2 paper, step 5. `a_const` is the fixed
    ln(sigma^2) constant — kept as a distinct name from the Illinois algorithm's
    own iterating A/B state so the two are never confused."""
    ex = math.exp(x)
    numerator = ex * (delta ** 2 - phi ** 2 - v - ex)
    denominator = 2.0 * (phi ** 2 + v + ex) ** 2
    return numerator / denominator - (x - a_const) / (tau ** 2)


def _solve_new_volatility(delta: float, phi: float, v: float, sigma: float, tau: float,
                           epsilon: float = 1e-6) -> float:
    """Illinois algorithm (regula falsi variant) root-find for the new
    volatility, exactly as specified in Glickman's Glicko-2 paper, step 5."""
    a_const = math.log(sigma ** 2)

    A = a_const
    if delta ** 2 > phi ** 2 + v:
        B = math.log(delta ** 2 - phi ** 2 - v)
    else:
        k = 1
        while _volatility_objective(a_const - k * tau, delta, phi, v, a_const, tau) < 0:
            k += 1
        B = a_const - k * tau

    fA = _volatility_objective(A, delta, phi, v, a_const, tau)
    fB = _volatility_objective(B, delta, phi, v, a_const, tau)

    while abs(B - A) > epsilon:
        C = A + (A - B) * fA / (fB - fA)
        fC = _volatility_objective(C, delta, phi, v, a_const, tau)
        if fC * fB < 0:
            A, fA = B, fB
        else:
            fA = fA / 2.0
        B, fB = C, fC

    return math.exp(A / 2.0)


def glicko2_update(
    player: Glicko2Rating,
    results: list[tuple[Glicko2Rating, float]],
    tau: float = DEFAULT_TAU,
) -> Glicko2Rating:
    """
    Update `player`'s rating given a list of (opponent, score) results in a
    single rating period, where score is 1.0 = win, 0.0 = loss, 0.5 = draw,
    from `player`'s perspective. Follows Glickman's Glicko-2 specification
    exactly (http://www.glicko.net/glicko/glicko2.pdf).
    """
    mu, phi = player.to_internal()

    if not results:
        # No games this period: RD grows toward uncertainty; rating/volatility unchanged.
        phi_star = math.sqrt(phi ** 2 + player.volatility ** 2)
        return Glicko2Rating.from_internal(mu, phi_star, player.volatility)

    g_list: list[float] = []
    e_list: list[float] = []
    scores: list[float] = []
    for opponent, score in results:
        mu_j, phi_j = opponent.to_internal()
        g_j = _g(phi_j)
        e_j = _e(mu, mu_j, phi_j)
        g_list.append(g_j)
        e_list.append(e_j)
        scores.append(score)

    v_inv = sum(g_j ** 2 * e_j * (1.0 - e_j) for g_j, e_j in zip(g_list, e_list))
    v = 1.0 / v_inv

    weighted_sum = sum(g_j * (s_j - e_j) for g_j, s_j, e_j in zip(g_list, scores, e_list))
    delta = v * weighted_sum

    new_sigma = _solve_new_volatility(delta, phi, v, player.volatility, tau)

    phi_star = math.sqrt(phi ** 2 + new_sigma ** 2)
    phi_prime = 1.0 / math.sqrt(1.0 / phi_star ** 2 + 1.0 / v)
    mu_prime = mu + phi_prime ** 2 * weighted_sum

    return Glicko2Rating.from_internal(mu_prime, phi_prime, new_sigma)


def update_single_game(
    player: Glicko2Rating,
    opponent: Glicko2Rating,
    score: float,
    tau: float = DEFAULT_TAU,
) -> Glicko2Rating:
    """Convenience wrapper: one player, one opponent, one game."""
    return glicko2_update(player, [(opponent, score)], tau=tau)


# ─────────────────────────────────────────────────────────────────────────────
# Session-log ingestion and chronological replay
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SolveEvent:
    ts: datetime
    username: str
    puzzle_id: str
    puzzle_rating: float
    solved: bool


def _parse_timestamp(ts: str) -> Optional[datetime]:
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError):
        return None


def load_solve_events(sessions_dir: str | Path) -> list[SolveEvent]:
    """Read every data/sessions/*.json file and return all valid solve events,
    sorted chronologically. Malformed entries are skipped, not raised."""
    sessions_dir = Path(sessions_dir)
    events: list[SolveEvent] = []

    for path in sorted(sessions_dir.glob("*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable session file %s: %s", path.name, exc)
            continue

        username = state.get("username") or path.stem
        for entry in state.get("history", []):
            ts = _parse_timestamp(entry.get("ts", ""))
            puzzle_id = entry.get("puzzleId")
            rating = entry.get("rating")
            solved = entry.get("solved")
            if ts is None or not puzzle_id or rating in (None, 0) or solved is None:
                continue
            events.append(SolveEvent(
                ts=ts, username=username, puzzle_id=puzzle_id,
                puzzle_rating=float(rating), solved=bool(solved),
            ))

    events.sort(key=lambda e: e.ts)
    return events


def _seed_player_ratings(sessions_dir: Path) -> dict[str, Glicko2Rating]:
    """Seed each player's initial rating from their persisted `estimatedElo`,
    where available, with a moderate (not maximal) starting RD."""
    seeded: dict[str, Glicko2Rating] = {}
    for path in sorted(sessions_dir.glob("*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        username = state.get("username") or path.stem
        elo = state.get("estimatedElo")
        if elo:
            seeded[username] = Glicko2Rating(rating=float(elo), rd=SEEDED_PLAYER_RD)
    return seeded


def expected_score(player: Glicko2Rating, opponent: Glicko2Rating) -> float:
    """Glicko-2's predicted probability that `player` beats (solves) `opponent`."""
    mu, _ = player.to_internal()
    mu_j, phi_j = opponent.to_internal()
    return _e(mu, mu_j, phi_j)


def fit_from_sessions(
    sessions_dir: str | Path,
    tau: float = DEFAULT_TAU,
    *,
    freeze_pool: bool = False,
    on_event=None,
) -> dict:
    """
    Replay every solve event across all session files in chronological order.

    freeze_pool=False (the original design): symmetric Glicko-2 update of both
        the player and the puzzle on every event (the puzzle "wins" exactly
        when the player fails), puzzles seeded at RD 200.
    freeze_pool=True: Lichess pool puzzles keep their published rating
        (RD 75 — it was fitted from thousands of attempts, far more evidence
        than this app will ever have for them); only players and puzzles mined
        from the player's own games (RD 300) are updated. This is the better
        design at small data volumes: 105 events cannot improve on a rating
        built from thousands, but they can calibrate the players and the
        heuristic ratings of mined puzzles.

    on_event(event, predicted_probability) is called BEFORE each update, for
    prequential (predict-then-learn) evaluation.

    Returns
    -------
    {
        "events_processed": int,
        "players": {username: Glicko2Rating.to_dict(), ...},
        "puzzles": {puzzle_id: Glicko2Rating.to_dict(), ...},
    }
    """
    sessions_dir = Path(sessions_dir)
    events = load_solve_events(sessions_dir)

    players: dict[str, Glicko2Rating] = _seed_player_ratings(sessions_dir)
    puzzles: dict[str, Glicko2Rating] = {}

    for event in events:
        player = players.get(event.username) or Glicko2Rating()
        mined = is_mined_puzzle(event.puzzle_id)
        seed_rd = (MINED_PUZZLE_RD if mined else POOL_PUZZLE_RD) if freeze_pool else PUZZLE_SEED_RD
        puzzle = puzzles.get(event.puzzle_id) or Glicko2Rating(
            rating=event.puzzle_rating, rd=seed_rd,
        )
        if on_event is not None:
            on_event(event, expected_score(player, puzzle))

        player_score = 1.0 if event.solved else 0.0

        new_player = update_single_game(player, puzzle, player_score, tau=tau)
        players[event.username] = new_player
        if freeze_pool and not mined:
            puzzles[event.puzzle_id] = puzzle
        else:
            puzzles[event.puzzle_id] = update_single_game(puzzle, player, 1.0 - player_score, tau=tau)

    return {
        "events_processed": len(events),
        "players": {u: r.to_dict() for u, r in players.items()},
        "puzzles": {p: r.to_dict() for p, r in puzzles.items()},
    }


def save_fitted_ratings(result: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info(
        "Fitted %d players and %d puzzles from %d events -> %s",
        len(result["players"]), len(result["puzzles"]), result["events_processed"], out_path,
    )


def load_fitted_puzzle_rating(puzzle_id: str, fitted_path: str | Path) -> Optional[float]:
    """Convenience lookup for later wiring into app.py's puzzle selection —
    NOT called anywhere yet. Returns None if the file or puzzle isn't found,
    so callers can fall back to the static Lichess Rating field."""
    fitted_path = Path(fitted_path)
    if not fitted_path.exists():
        return None
    try:
        data = json.loads(fitted_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    puzzle = data.get("puzzles", {}).get(puzzle_id)
    return puzzle["rating"] if puzzle else None


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Fit player and puzzle Glicko-2 ratings from data/sessions/*.json")
    ap.add_argument("--sessions-dir", default="data/sessions")
    ap.add_argument("--out", default="data/processed/fitted_ratings.json")
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    result = fit_from_sessions(args.sessions_dir, tau=args.tau)
    save_fitted_ratings(result, args.out)

    print(f"\nEvents processed: {result['events_processed']}")
    print(f"Players fitted:   {len(result['players'])}")
    print(f"Puzzles fitted:   {len(result['puzzles'])}")


if __name__ == "__main__":
    main()
