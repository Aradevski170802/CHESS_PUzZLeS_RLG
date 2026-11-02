"""
Aggregate statistics of the mined puzzles and of the real attempts, with no user data.

The raw files in data/user_puzzles and data/sessions name their users and hold account
fields, so they stay private. This script keeps only totals, which the notebooks read.
From the session files it reads the attempt history and nothing else.

Output: eval/research/mined_puzzles_summary.json
Usage:  python -m scripts.research.summarise_mined_puzzles
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

PUZZLES = Path("data/user_puzzles")
SESSIONS = Path("data/sessions")
OUT = Path("eval/research/mined_puzzles_summary.json")


def main() -> None:
    files = sorted(PUZZLES.glob("*.json"))
    kept = [p for f in files for p in json.loads(f.read_text("utf-8"))]
    retired = [p for f in sorted((PUZZLES / "retired").glob("*.json"))
               for p in json.loads(f.read_text("utf-8"))]
    maia = sorted(round(p["maiaProbability"], 4) for p in kept
                  if p.get("maiaProbability") is not None)

    attempts = solved = mined_attempts = mined_solved = players = 0
    for f in sorted(SESSIONS.glob("*.json")):
        history = [e for e in json.loads(f.read_text("utf-8")).get("history") or []
                   if e.get("solved") is not None and e.get("rating")]
        players += bool(history)
        for e in history:
            attempts += 1
            solved += bool(e["solved"])
            if str(e.get("puzzleId", "")).startswith("gen_"):
                mined_attempts += 1
                mined_solved += bool(e["solved"])

    summary = {
        "users_with_puzzles": len(files),
        "puzzles_kept": len(kept),
        "puzzles_retired": len(retired),
        "solver_moves": dict(sorted(Counter(len(p["Moves"].split()) // 2 for p in kept).items())),
        "categories": dict(Counter(p.get("PrimaryCategory", "General") for p in kept).most_common()),
        "median_mistake_cp": statistics.median(p["evalDrop"] for p in kept if p.get("evalDrop") is not None),
        "median_margin_cp": statistics.median(p["clarityCp"] for p in kept if p.get("clarityCp") is not None),
        "maia_probabilities": maia,
        "attempts": {"players": players, "total": attempts, "solved": solved,
                     "mined_total": mined_attempts, "mined_solved": mined_solved},
    }
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {OUT}: {len(kept)} puzzles kept, {attempts} attempts by {players} players")


if __name__ == "__main__":
    main()
