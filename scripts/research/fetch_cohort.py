"""
Build a real-player research cohort from the public Chess.com API.

Why
────
The app has only a handful of real users, which is far too few to validate
the weakness models on real players. But every Chess.com player's rated games
are public, so the *analysis* side of the system can be evaluated on hundreds
of real players who never touch the app. This script collects that cohort.

What it does
─────────────
1. Draws candidate usernames from three sources, in priority order:
   a. Snowball: opponents found in already-collected games, whose game
      rating (from the PGN headers) falls in a band that still needs players.
   b. Titled-player lists (NM … GM) while the top bands are unfilled.
   c. Random members of Chess.com country player lists.
   Country lists alone are dominated by beginners, so the upper bands would
   never fill; snowball + titled sampling fixes that. The trade-off — the
   cohort is not a uniform random sample of Chess.com — is stated in the
   evaluation write-up.
2. Keeps players who are active (a rated rapid or blitz game in the last
   year) and experienced (≥ MIN_RECORD games in that time class).
3. Stratifies them into rating bands so no band dominates the evaluation.
4. Downloads their most recent standard-chess (no variants) rated rapid and
   blitz games, newest first, until GAMES_PER_PLAYER are collected.

Privacy
────────
Outputs are pseudonymised: each player is stored under a salted SHA-256
prefix. The username ↔ id mapping lives only in manifest_private.json. Raw
API responses stay in the existing gitignored cache, and data/research/ is
gitignored too.

Politeness
───────────
All requests go through the project's fetcher (disk-cached, identifying
User-Agent) with a 1 s delay, per Chess.com's published-data API guidance.
The script is resumable: re-running skips everything already collected.

Usage
─────
    python -m scripts.research.fetch_cohort --per-band 60
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.api import chess_com_fetcher as ccf

OUT_DIR = Path("data/research/cohort")
GAMES_DIR = OUT_DIR / "games"
MANIFEST = OUT_DIR / "manifest_private.json"
SALT = "adaptive-chess-cohort-v1"

COUNTRIES = ["US", "GB", "IN", "DE", "FR", "ES", "BR", "CA", "AU", "NL",
             "PL", "IT", "SE", "RU", "PH", "MK", "RS", "TR", "AR", "MX"]
BANDS = [(0, 1000), (1000, 1400), (1400, 1800), (1800, 2200), (2200, 3500)]
TIME_CLASSES = ("rapid", "blitz")
MIN_RECORD = 100
ACTIVE_WITHIN_DAYS = 365
GAMES_PER_PLAYER = 80
DELAY = 1.0

logger = logging.getLogger("cohort")


def pid(username: str) -> str:
    return hashlib.sha256(f"{SALT}:{username.lower()}".encode()).hexdigest()[:12]


def band_of(rating: int) -> int | None:
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= rating < hi:
            return i
    return None


def fetch(url: str, key: str) -> dict | None:
    cached = (ccf.CACHE_DIR / f"{key}.json").exists()
    try:
        data = ccf._fetch(url, cache_key=key)
    except requests.HTTPError as exc:
        logger.debug("HTTP error %s for %s", exc, url)
        data = None
    except requests.RequestException as exc:
        logger.warning("Network error for %s: %s", url, exc)
        time.sleep(5)
        data = None
    if not cached:
        time.sleep(DELAY)
    return data


def primary_rating(stats: dict) -> tuple[str, int, int] | None:
    """(time_class, rating, games) for the player's main active time class."""
    now = datetime.now(timezone.utc).timestamp()
    best = None
    for tc in TIME_CLASSES:
        block = stats.get(f"chess_{tc}") or {}
        last = block.get("last") or {}
        rec = block.get("record") or {}
        games = int(rec.get("win", 0)) + int(rec.get("loss", 0)) + int(rec.get("draw", 0))
        if not last.get("rating") or games < MIN_RECORD:
            continue
        if now - float(last.get("date", 0)) > ACTIVE_WITHIN_DAYS * 86400:
            continue
        if best is None or games > best[2]:
            best = (tc, int(last["rating"]), games)
    return best


def collect_games(username: str) -> list[dict]:
    """Newest-first standard rated rapid/blitz games with their metadata."""
    arch = fetch(f"{ccf.BASE_URL}/player/{username}/games/archives", f"archives_{username}")
    games: list[dict] = []
    for url in reversed((arch or {}).get("archives", [])):
        if len(games) >= GAMES_PER_PLAYER:
            break
        y, m = url.rstrip("/").split("/")[-2:]
        monthly = fetch(url, f"games_{username}_{y}_{m}") or {}
        for g in reversed(monthly.get("games", [])):
            if len(games) >= GAMES_PER_PLAYER:
                break
            if (g.get("rules") != "chess" or not g.get("rated")
                    or g.get("time_class") not in TIME_CLASSES or not g.get("pgn")):
                continue
            games.append({
                "pgn": g["pgn"],
                "end_time": g.get("end_time"),
                "time_class": g.get("time_class"),
                "time_control": g.get("time_control"),
            })
    return games


TITLES = ["NM", "CM", "WCM", "WNM", "FM", "WFM", "IM", "WIM", "GM", "WGM"]
_TAG = re.compile(r'\[(White|Black|WhiteElo|BlackElo) "([^"]*)"\]')
SNOWBALL_PER_SOURCE = 4   # opponents taken per kept player per band, for diversity


def snowball(manifest: dict, needed: set[int], rng: random.Random) -> list[str]:
    """Opponents of kept players whose game rating lies in a needed band."""
    out: list[str] = []
    for username, info in manifest["players"].items():
        path = GAMES_DIR / f"{info['id']}.json"
        if not path.exists():
            continue
        by_band: dict[int, set[str]] = {}
        for g in json.loads(path.read_text("utf-8"))["games"]:
            tags = dict(_TAG.findall(g["pgn"][:1500]))
            for side, elo_key in (("White", "WhiteElo"), ("Black", "BlackElo")):
                name = tags.get(side, "").lower()
                if not name or name == username:
                    continue
                try:
                    band = band_of(int(tags.get(elo_key, "")))
                except ValueError:
                    continue
                if band in needed:
                    by_band.setdefault(band, set()).add(name)
        for names in by_band.values():
            names = sorted(names)
            rng.shuffle(names)
            out.extend(names[:SNOWBALL_PER_SOURCE])
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-band", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-candidates", type=int, default=6000)
    args = ap.parse_args()

    GAMES_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(OUT_DIR / "fetch.log", encoding="utf-8"),
                                  logging.StreamHandler()])
    manifest = json.loads(MANIFEST.read_text("utf-8")) if MANIFEST.exists() else {"players": {}}

    rng = random.Random(args.seed)
    country: list[str] = []
    for cc in COUNTRIES:
        data = fetch(f"{ccf.BASE_URL}/country/{cc}/players", f"country_{cc}")
        names = list((data or {}).get("players", []))
        rng.shuffle(names)
        country.extend(names[: args.max_candidates // len(COUNTRIES)])
    rng.shuffle(country)
    titled: list[str] = []
    for t in TITLES:
        data = fetch(f"{ccf.BASE_URL}/titled/{t}", f"titled_{t}")
        titled.extend((data or {}).get("players", []))
    rng.shuffle(titled)
    logger.info("candidates: %d country, %d titled", len(country), len(titled))

    per_band = [0] * len(BANDS)
    for info in manifest["players"].values():
        per_band[info["band"]] += 1
    rejected: set[str] = set()
    checked = 0

    def done() -> bool:
        return all(n >= args.per_band for n in per_band)

    while not done():
        needed = {i for i, n in enumerate(per_band) if n < args.per_band}
        pool = snowball(manifest, needed, rng)
        if needed & {3, 4}:
            pool += titled
        pool += country
        kept_this_pass = 0
        for username in pool:
            if done():
                break
            username = username.lower()
            if username in manifest["players"] or username in rejected:
                continue
            checked += 1
            rejected.add(username)       # re-admitted below only if kept
            stats = fetch(f"{ccf.BASE_URL}/player/{username}/stats", f"stats_{username}")
            main_tc = primary_rating(stats or {})
            if not main_tc:
                continue
            band = band_of(main_tc[1])
            if band is None or per_band[band] >= args.per_band:
                continue
            games = collect_games(username)
            if len(games) < GAMES_PER_PLAYER // 2:
                continue
            p = pid(username)
            (GAMES_DIR / f"{p}.json").write_text(json.dumps({
                "id": p, "band": band, "time_class": main_tc[0],
                "rating": main_tc[1], "games": games,
            }), encoding="utf-8")
            manifest["players"][username] = {"id": p, "band": band, "rating": main_tc[1],
                                             "time_class": main_tc[0], "n_games": len(games)}
            MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
            per_band[band] += 1
            kept_this_pass += 1
            logger.info("kept %s band=%d rating=%d games=%d | per-band %s | checked %d",
                        p, band, main_tc[1], len(games), per_band, checked)
            # New players bring new opponents: rebuild the snowball pool when
            # a still-needed band gains a source.
            if kept_this_pass % 10 == 0:
                break
        if kept_this_pass == 0:
            logger.info("no candidates left that fit the remaining bands")
            break

    logger.info("DONE. per-band counts: %s (checked %d candidates)", per_band, checked)


if __name__ == "__main__":
    main()
