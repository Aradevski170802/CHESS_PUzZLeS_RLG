"""
Chess.com Public API client.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 AUTHENTICATION NOTE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chess.com's Published Data API is fully public — no OAuth,
no API key, no login token required. All game histories are
public by default (per Chess.com's Terms of Service §3).

Our "login" flow is therefore:
  1. Player types their Chess.com username.
  2. We fetch their public profile (avatar, name, ratings).
  3. Player sees their profile card and clicks "This is me".
  4. We fetch their game archives and begin analysis.

This is the same approach used by Aimchess, Chessable, and
other third-party Chess.com tools.

Rate limit: Chess.com recommends ≤ 1 request/second to
archive endpoints. We add a configurable delay (default 0.5s)
and cache all responses to disk to avoid re-fetching.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.chess.com/pub"
CACHE_DIR = Path("data/cache/chess_com")

# Identifies our app to Chess.com (polite crawling practice)
HEADERS = {
    "User-Agent": "AdaptiveChessPuzzleAdvisor/1.0 (MSc dissertation, City College York)"
}

# Delay between archive fetches to respect Chess.com's rate limit
REQUEST_DELAY_SEC = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_player_profile(username: str) -> dict:
    """
    Fetch a player's public profile and current ratings.

    Returns
    -------
    {
        "username":    str,
        "name":        str,          # display name (may be empty)
        "title":       str,          # FIDE title e.g. "GM" (may be empty)
        "avatar":      str,          # URL to profile picture
        "country":     str,          # ISO country code e.g. "MK"
        "followers":   int,
        "joined":      int,          # Unix timestamp
        "profile_url": str,
        "ratings": {
            "rapid":   int | None,
            "blitz":   int | None,
            "bullet":  int | None,
            "daily":   int | None,
        }
    }

    Raises requests.HTTPError if the username does not exist (404).
    """
    # Profile endpoint
    profile_data = _fetch(
        f"{BASE_URL}/player/{username.lower()}",
        cache_key=f"profile_{username.lower()}",
    )

    # Stats endpoint (ratings per time control)
    stats_data = _fetch(
        f"{BASE_URL}/player/{username.lower()}/stats",
        cache_key=f"stats_{username.lower()}",
    )

    # Extract ratings
    ratings: dict[str, Optional[int]] = {}
    for tc_key, label in [
        ("chess_rapid",  "rapid"),
        ("chess_blitz",  "blitz"),
        ("chess_bullet", "bullet"),
        ("chess_daily",  "daily"),
    ]:
        last = stats_data.get(tc_key, {}).get("last", {})
        ratings[label] = last.get("rating") or None

    # Country comes as a URL like https://api.chess.com/pub/country/MK
    country_url = profile_data.get("country", "")
    country_code = country_url.split("/")[-1] if country_url else ""

    return {
        "username":    profile_data.get("username", username),
        "name":        profile_data.get("name", ""),
        "title":       profile_data.get("title", ""),
        "avatar":      profile_data.get("avatar", ""),
        "country":     country_code,
        "followers":   profile_data.get("followers", 0),
        "joined":      profile_data.get("joined", 0),
        "profile_url": profile_data.get("url", f"https://www.chess.com/member/{username}"),
        "ratings":     ratings,
    }


def get_best_rating(profile: dict) -> int:
    """Return the player's highest available rating across all time controls."""
    ratings = profile.get("ratings", {})
    valid = [r for r in ratings.values() if r is not None]
    return max(valid) if valid else 1200  # default if no rating found


def get_recent_games(
    username: str,
    n: int = 100,
    *,
    rated_only: bool = True,
    time_controls: Optional[list[str]] = None,
) -> list[str]:
    """
    Fetch the last N games for a player as PGN strings.

    Parameters
    ----------
    username      : Chess.com username
    n             : number of games to return (default 100)
    rated_only    : skip unrated games (default True)
    time_controls : if set, only include these time controls
                    e.g. ["600", "180", "60"] for rapid/blitz/bullet

    Returns
    -------
    List of PGN strings, newest game first.
    """
    archives_data = _fetch(
        f"{BASE_URL}/player/{username.lower()}/games/archives",
        cache_key=f"archives_{username.lower()}",
    )
    archive_urls: list[str] = archives_data.get("archives", [])

    if not archive_urls:
        logger.warning("No game archives found for %s", username)
        return []

    pgn_strings: list[str] = []

    # Iterate from most recent month backwards
    for archive_url in reversed(archive_urls):
        if len(pgn_strings) >= n:
            break

        # e.g. .../2026/06  →  cache key: games_username_2026_06
        parts = archive_url.rstrip("/").split("/")
        year, month = parts[-2], parts[-1]
        cache_key = f"games_{username.lower()}_{year}_{month}"

        monthly = _fetch(archive_url, cache_key=cache_key)
        games: list[dict] = monthly.get("games", [])

        # Walk newest-first within the month
        for game in reversed(games):
            if len(pgn_strings) >= n:
                break

            if rated_only and not game.get("rated", False):
                continue

            pgn = game.get("pgn", "")
            if not pgn:
                continue

            # Filter by time control if requested
            if time_controls:
                tc = game.get("time_control", "")
                if not any(tc.startswith(t) for t in time_controls):
                    continue

            pgn_strings.append(pgn)

        time.sleep(REQUEST_DELAY_SEC)

    logger.info("Fetched %d games for %s", len(pgn_strings), username)
    return pgn_strings


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch(url: str, cache_key: str) -> dict:
    """
    Return cached JSON if available; otherwise fetch from Chess.com and cache.
    Cache is stored in data/cache/chess_com/<cache_key>.json
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        logger.debug("Cache hit: %s", cache_key)
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)

    logger.debug("Fetching: %s", url)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    return data


def clear_cache(username: str) -> None:
    """Delete all cached data for a username (forces re-fetch on next call)."""
    username = username.lower()
    for f in CACHE_DIR.glob(f"*{username}*.json"):
        f.unlink()
        logger.info("Deleted cache: %s", f.name)
