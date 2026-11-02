"""
Add Maia's "how likely is a player of your rating to find this" to stored mined
puzzles (src/puzzles/generator.py: annotate_maia).

New puzzles are annotated when they are mined, from the owner's rating in the game they
came from. Puzzles mined before that carry no rating, so the owner's current estimated
rating (data/sessions/<user>.json) stands in for it; a user without a session file is
left as it is.

Usage:  python scripts/annotate_maia.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.neural import maia                                         # noqa: E402
from src.puzzles.generator import USER_PUZZLES_DIR, annotate_maia   # noqa: E402

SESSIONS = Path("data/sessions")


def main() -> None:
    if not maia.available():
        sys.exit("lc0 or the Maia weights are missing (see README.md, Data)")
    for file in sorted(USER_PUZZLES_DIR.glob("*.json")):
        session = SESSIONS / file.name
        rating = (json.loads(session.read_text("utf-8")).get("estimatedElo")
                  if session.exists() else None)
        puzzles = json.loads(file.read_text("utf-8"))
        changed = annotate_maia(puzzles, rating=rating)
        if changed:
            tmp = file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(puzzles, indent=2), encoding="utf-8")
            tmp.replace(file)
        print(f"{file.stem}: {changed} of {len(puzzles)} puzzles annotated "
              f"(fallback rating {rating})")


if __name__ == "__main__":
    main()
