"""Unit tests for src/analysis/difficulty_fitter.py.

The core Glicko-2 math is validated against Mark Glickman's own published worked
example (http://www.glicko.net/glicko/glicko2.pdf, "Example of the Glicko-2
system"): a player rated 1500/RD200/vol0.06 plays three opponents in one rating
period (beats a 1400/RD30, loses to a 1550/RD100, loses to a 1700/RD300) and
should end up at approximately rating=1464.06, RD=151.52, volatility=0.05999.
This is the standard reference test vector used across other Glicko-2
implementations, so a match here is strong evidence the algorithm transcription
is correct, independent of anything else in this codebase.
"""

import json

import pytest

from src.analysis.difficulty_fitter import (
    Glicko2Rating,
    SEEDED_PLAYER_RD,
    _seed_player_ratings,
    fit_from_sessions,
    glicko2_update,
    load_fitted_puzzle_rating,
    load_solve_events,
    save_fitted_ratings,
    update_single_game,
)


class TestGlicko2CanonicalExample:
    def test_matches_glickmans_published_worked_example(self):
        player = Glicko2Rating(rating=1500, rd=200, volatility=0.06)
        opponents = [
            (Glicko2Rating(rating=1400, rd=30), 1.0),
            (Glicko2Rating(rating=1550, rd=100), 0.0),
            (Glicko2Rating(rating=1700, rd=300), 0.0),
        ]

        result = glicko2_update(player, opponents, tau=0.5)

        assert result.rating == pytest.approx(1464.06, abs=0.5)
        assert result.rd == pytest.approx(151.52, abs=0.5)
        assert result.volatility == pytest.approx(0.05999, abs=0.0005)


class TestGlicko2Properties:
    def test_no_games_leaves_rating_and_volatility_unchanged_but_grows_rd(self):
        player = Glicko2Rating(rating=1500, rd=100, volatility=0.06)
        result = glicko2_update(player, [])
        assert result.rating == pytest.approx(1500)
        assert result.volatility == pytest.approx(0.06)
        assert result.rd > 100  # uncertainty grows when there is no new evidence

    def test_beating_a_similarly_rated_opponent_increases_rating(self):
        player = Glicko2Rating(rating=1500, rd=100, volatility=0.06)
        opponent = Glicko2Rating(rating=1500, rd=100, volatility=0.06)
        result = update_single_game(player, opponent, 1.0)
        assert result.rating > 1500

    def test_losing_to_a_similarly_rated_opponent_decreases_rating(self):
        player = Glicko2Rating(rating=1500, rd=100, volatility=0.06)
        opponent = Glicko2Rating(rating=1500, rd=100, volatility=0.06)
        result = update_single_game(player, opponent, 0.0)
        assert result.rating < 1500

    def test_a_win_and_a_loss_are_roughly_symmetric_for_equal_opponents(self):
        player = Glicko2Rating(rating=1500, rd=100, volatility=0.06)
        opponent = Glicko2Rating(rating=1500, rd=100, volatility=0.06)
        win = update_single_game(player, opponent, 1.0)
        loss = update_single_game(player, opponent, 0.0)
        assert (win.rating - 1500) == pytest.approx(-(loss.rating - 1500), rel=0.05)

    def test_scale_conversion_round_trips(self):
        original = Glicko2Rating(rating=1732.4, rd=88.1, volatility=0.061)
        mu, phi = original.to_internal()
        restored = Glicko2Rating.from_internal(mu, phi, original.volatility)
        assert restored.rating == pytest.approx(original.rating)
        assert restored.rd == pytest.approx(original.rd)


def _write_session(tmp_path, filename, username, history, estimated_elo=None):
    state = {"username": username, "history": history, "bestStreak": 0}
    if estimated_elo is not None:
        state["estimatedElo"] = estimated_elo
    (tmp_path / filename).write_text(json.dumps(state), encoding="utf-8")


class TestLoadSolveEvents:
    def test_events_are_sorted_chronologically_across_files(self, tmp_path):
        _write_session(tmp_path, "alice.json", "alice", [
            {"ts": "2026-01-02T00:00:00Z", "puzzleId": "p_later", "category": "Fork",
             "rating": 1500, "solved": True},
        ])
        _write_session(tmp_path, "bob.json", "bob", [
            {"ts": "2026-01-01T00:00:00Z", "puzzleId": "p_earlier", "category": "Pin",
             "rating": 1500, "solved": False},
        ])

        events = load_solve_events(tmp_path)

        assert [e.puzzle_id for e in events] == ["p_earlier", "p_later"]

    def test_malformed_entries_are_skipped(self, tmp_path):
        _write_session(tmp_path, "carol.json", "carol", [
            {"ts": "2026-01-01T00:00:00Z", "puzzleId": "good", "category": "Fork",
             "rating": 1500, "solved": True},
            {"ts": "2026-01-01T00:01:00Z", "puzzleId": "missing_rating", "category": "Fork",
             "solved": True},
            {"puzzleId": "missing_ts", "category": "Fork", "rating": 1500, "solved": True},
            {"ts": "2026-01-01T00:02:00Z", "category": "Fork", "rating": 1500, "solved": True},
        ])

        events = load_solve_events(tmp_path)

        assert len(events) == 1
        assert events[0].puzzle_id == "good"

    def test_missing_sessions_dir_files_are_tolerated(self, tmp_path):
        assert load_solve_events(tmp_path) == []  # empty dir, no files at all


class TestSeedPlayerRatings:
    def test_seeds_from_estimated_elo_when_present(self, tmp_path):
        _write_session(tmp_path, "dave.json", "dave", [], estimated_elo=1800)
        seeded = _seed_player_ratings(tmp_path)
        assert seeded["dave"].rating == 1800.0
        assert seeded["dave"].rd == SEEDED_PLAYER_RD

    def test_no_seed_when_estimated_elo_absent(self, tmp_path):
        _write_session(tmp_path, "erin.json", "erin", [])
        seeded = _seed_player_ratings(tmp_path)
        assert "erin" not in seeded


class TestFitFromSessions:
    def test_empty_directory_yields_zero_events(self, tmp_path):
        result = fit_from_sessions(tmp_path)
        assert result["events_processed"] == 0
        assert result["players"] == {}
        assert result["puzzles"] == {}

    def test_end_to_end_fit_produces_ratings_for_every_player_and_puzzle(self, tmp_path):
        _write_session(tmp_path, "frank.json", "frank", [
            {"ts": "2026-01-01T00:00:00Z", "puzzleId": "pA", "category": "Fork",
             "rating": 1400, "solved": True},
            {"ts": "2026-01-01T00:05:00Z", "puzzleId": "pB", "category": "Pin",
             "rating": 1600, "solved": False},
        ], estimated_elo=1500)

        result = fit_from_sessions(tmp_path)

        assert result["events_processed"] == 2
        assert set(result["players"].keys()) == {"frank"}
        assert set(result["puzzles"].keys()) == {"pA", "pB"}
        for rating_dict in list(result["players"].values()) + list(result["puzzles"].values()):
            assert set(rating_dict.keys()) == {"rating", "rd", "volatility"}

    def test_solving_a_puzzle_raises_player_rating_and_lowers_puzzle_rating(self, tmp_path):
        # A player solving a same-rated puzzle should end up rated higher than
        # they started, and the puzzle should end up rated lower (it "lost").
        _write_session(tmp_path, "gina.json", "gina", [
            {"ts": "2026-01-01T00:00:00Z", "puzzleId": "pC", "category": "Fork",
             "rating": 1500, "solved": True},
        ], estimated_elo=1500)

        result = fit_from_sessions(tmp_path)

        assert result["players"]["gina"]["rating"] > 1500
        assert result["puzzles"]["pC"]["rating"] < 1500


class TestPersistence:
    def test_save_and_load_fitted_puzzle_rating_round_trip(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        _write_session(sessions_dir, "henry.json", "henry", [
            {"ts": "2026-01-01T00:00:00Z", "puzzleId": "pD", "category": "Fork",
             "rating": 1500, "solved": True},
        ])

        # fit_from_sessions runs (and reads pD's input rating) before the output
        # file below is written, so writing the output into the same tmp_path
        # tree afterwards cannot leak back in as a spurious session file.
        result = fit_from_sessions(sessions_dir)
        out_path = tmp_path / "fitted_ratings.json"
        save_fitted_ratings(result, out_path)

        assert out_path.exists()
        rating = load_fitted_puzzle_rating("pD", out_path)
        assert rating == pytest.approx(result["puzzles"]["pD"]["rating"])

    def test_load_fitted_puzzle_rating_missing_file_returns_none(self, tmp_path):
        assert load_fitted_puzzle_rating("anything", tmp_path / "nope.json") is None

    def test_load_fitted_puzzle_rating_unknown_puzzle_returns_none(self, tmp_path):
        save_fitted_ratings({"events_processed": 0, "players": {}, "puzzles": {}}, tmp_path / "r.json")
        assert load_fitted_puzzle_rating("unknown", tmp_path / "r.json") is None
