"""Unit tests for puzzle_loader — no CSV required."""

import pandas as pd
import pytest

from src.data.puzzle_loader import (
    WEAKNESS_CATEGORIES,
    _enrich,
    _tags_to_categories,
    get_puzzles_for_player,
    moves_to_length,
    rating_to_tier,
)


def _make_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "PuzzleId": "TEST",
        "FEN": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "Moves": "e2e4 e7e5",
        "Rating": 1500,
        "RatingDeviation": 75,
        "Popularity": 80,
        "NbPlays": 1000,
        "Themes": "fork middlegame short",
        "GameUrl": "https://lichess.org/abc",
        "OpeningTags": float("nan"),
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


class TestRatingToTier:
    def test_beginner(self):
        assert rating_to_tier(500) == "Beginner"

    def test_boundary(self):
        assert rating_to_tier(1000) == "Easy"
        assert rating_to_tier(999) == "Beginner"

    def test_master(self):
        assert rating_to_tier(2500) == "Master"


class TestMovesToLength:
    def test_one_move(self):
        assert moves_to_length("e2e4") == "oneMove"

    def test_short(self):
        assert moves_to_length("e2e4 e7e5 d2d4") == "short"

    def test_long(self):
        assert moves_to_length("a1a2 b1b2 c1c2 d1d2 e1e2") == "long"

    def test_very_long(self):
        moves = " ".join(["a1a2"] * 9)
        assert moves_to_length(moves) == "veryLong"


class TestTagsToCategories:
    def test_fork(self):
        assert "Fork" in _tags_to_categories(["fork"])

    def test_mating_pattern(self):
        assert "Mating Pattern" in _tags_to_categories(["mateIn2"])

    def test_endgame_multi_category(self):
        cats = _tags_to_categories(["rookEndgame"])
        assert "Endgame" in cats
        assert "Rook Endgame" in cats

    def test_unknown_tag_ignored(self):
        assert _tags_to_categories(["unknownTag123"]) == []

    def test_no_duplicates(self):
        cats = _tags_to_categories(["fork", "fork", "pin"])
        assert cats.count("Fork") == 1


class TestEnrich:
    def test_adds_required_columns(self):
        df = _make_df([{}])
        enriched = _enrich(df)
        for col in ["ThemeList", "Categories", "PrimaryCategory", "DifficultyTier", "MoveLength"]:
            assert col in enriched.columns

    def test_primary_category_fork(self):
        df = _make_df([{"Themes": "fork middlegame short"}])
        enriched = _enrich(df)
        assert enriched.loc[0, "PrimaryCategory"] == "Fork"

    def test_primary_category_general_when_no_tactical_theme(self):
        df = _make_df([{"Themes": "middlegame short"}])
        enriched = _enrich(df)
        assert enriched.loc[0, "PrimaryCategory"] == "General"

    def test_difficulty_tier_assigned(self):
        df = _make_df([{"Rating": 1300}])
        enriched = _enrich(df)
        assert enriched.loc[0, "DifficultyTier"] == "Intermediate"


class TestGetPuzzlesForPlayer:
    def _make_pool(self):
        rows = [
            {"PuzzleId": f"P{i}", "Rating": 1500, "Popularity": 80, "Themes": theme}
            for i, theme in enumerate(["fork short", "pin short", "skewer short",
                                       "mateIn2 short", "rookEndgame long"])
        ]
        return _enrich(_make_df(rows))

    def test_returns_dataframe(self):
        pool = self._make_pool()
        result = get_puzzles_for_player(pool, 1500, ["Fork", "Pin"])
        assert isinstance(result, pd.DataFrame)

    def test_filters_by_category(self):
        pool = self._make_pool()
        result = get_puzzles_for_player(pool, 1500, ["Fork"])
        assert all("Fork" in cats for cats in result["Categories"])

    def test_empty_when_no_match(self):
        pool = self._make_pool()
        result = get_puzzles_for_player(pool, 1500, ["Zugzwang"])
        assert result.empty

    def test_no_duplicates(self):
        pool = self._make_pool()
        result = get_puzzles_for_player(pool, 1500, ["Endgame", "Rook Endgame"])
        assert result["PuzzleId"].nunique() == len(result)


class TestWEAKNESS_CATEGORIES:
    def test_no_duplicates(self):
        assert len(WEAKNESS_CATEGORIES) == len(set(WEAKNESS_CATEGORIES))

    def test_core_categories_present(self):
        for cat in ["Fork", "Pin", "Skewer", "Mating Pattern", "Endgame"]:
            assert cat in WEAKNESS_CATEGORIES
