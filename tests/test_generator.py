"""Unit tests for src/puzzles/generator.py.

Only the pure, engine-free surface is tested here (no live Stockfish process): score
conversion, sequence validation, tactic classification on hand-verified positions,
rating estimation, and the save/load JSON round trip. `_extract_stockfish` itself
requires a running engine and is intentionally out of scope for offline unit tests.
"""

import json

import chess
import chess.engine
import pytest

import src.puzzles.generator as generator

START_FEN = chess.Board().fen()

# Same hand-verified back-rank mate position used in test_puzzle_evaluator.py:
# black king g8 boxed by its own pawns f7/g7/h7, white rook a1 -> a8 is checkmate.
MATE_FEN = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"
MATE_MOVE = chess.Move.from_uci("a1a8")

# Promotion that does NOT deliver check: black king on h5, far from a8's rank/file/
# diagonal (file diff 7, rank diff 3 -> not a queen line), so a7a8=Q is unambiguously
# a quiet promotion rather than a mate/check, isolating the "Promotion" branch.
PROMOTION_FEN = "8/P7/8/7k/8/8/8/K7 w - - 0 1"
PROMOTION_MOVE = chess.Move.from_uci("a7a8q")


class TestPovCp:
    def test_positive_centipawn_score(self):
        score = chess.engine.PovScore(chess.engine.Cp(150), chess.WHITE)
        assert generator._pov_cp(score, chess.WHITE) == 150
        assert generator._pov_cp(score, chess.BLACK) == -150

    def test_mate_score_maps_to_mate_cp_constant(self):
        score = chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE)
        assert generator._pov_cp(score, chess.WHITE) == generator.MATE_CP
        assert generator._pov_cp(score, chess.BLACK) == -generator.MATE_CP


class TestValidateSequence:
    def test_legal_sequence_is_valid(self):
        assert generator._validate_sequence(START_FEN, ["e2e4", "e7e5"]) is True

    def test_illegal_move_is_invalid(self):
        assert generator._validate_sequence(START_FEN, ["e2e5"]) is False

    def test_malformed_fen_is_invalid(self):
        assert generator._validate_sequence("not a fen", ["e2e4"]) is False


class TestClassifyTactic:
    def test_checkmating_move_is_mating_pattern(self):
        board = chess.Board(MATE_FEN)
        assert generator._classify_tactic(board, MATE_MOVE) == "Mating Pattern"

    def test_promotion_without_check_is_promotion(self):
        board = chess.Board(PROMOTION_FEN)
        assert generator._classify_tactic(board, PROMOTION_MOVE) == "Promotion"

    def test_quiet_non_capturing_move_is_quiet_move(self):
        # No capture and no check falls through every tactical/endgame branch to the
        # dedicated "Quiet Move" label (added deliberately per the module's own
        # comment, so these no longer get lumped into the catch-all "General").
        board = chess.Board()  # start position
        move = chess.Move.from_uci("b1c3")  # Nc3, no capture, no check
        assert generator._classify_tactic(board, move) == "Quiet Move"

    def test_equal_value_capture_with_no_follow_up_is_general(self):
        # 1.e4 d5 2.exd5: a defended, roughly-equal-value pawn trade with no check,
        # no fork, no pin/skewer (pawn), no sacrifice (equal value), no x-ray
        # (pawn) -- the one case that still legitimately falls through to "General".
        fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
        board = chess.Board(fen)
        move = chess.Move.from_uci("e4d5")
        assert generator._classify_tactic(board, move) == "General"


class TestEstimateRating:
    def test_at_threshold_gives_base_rating(self):
        # eval_drop == PUZZLE_THRESHOLD, 2-move (minimum) solution -> pure base
        rating = generator._estimate_rating(
            eval_drop=generator.PUZZLE_THRESHOLD, solution_len=2
        )
        assert rating == 800

    def test_larger_eval_drop_increases_rating(self):
        low = generator._estimate_rating(eval_drop=generator.PUZZLE_THRESHOLD, solution_len=2)
        high = generator._estimate_rating(eval_drop=generator.PUZZLE_THRESHOLD + 400, solution_len=2)
        assert high > low

    def test_longer_solution_increases_rating(self):
        short = generator._estimate_rating(eval_drop=generator.PUZZLE_THRESHOLD, solution_len=2)
        longer = generator._estimate_rating(eval_drop=generator.PUZZLE_THRESHOLD, solution_len=4)
        assert longer == short + (4 - 2) * 150

    def test_rating_is_clamped_between_700_and_2400(self):
        low = generator._estimate_rating(eval_drop=-1000, solution_len=2)
        high = generator._estimate_rating(eval_drop=10_000, solution_len=20)
        assert low == 700
        assert high == 2400


class TestDifficultyTier:
    @pytest.mark.parametrize(
        "rating,expected",
        [
            (999, "Beginner"),
            (1000, "Intermediate"),
            (1300, "Advanced"),
            (1600, "Hard"),
            (1900, "Expert"),
        ],
    )
    def test_boundaries(self, rating, expected):
        assert generator._difficulty_tier(rating) == expected


class TestGenerateFromGamesWithoutStockfish:
    def test_returns_empty_list_when_no_stockfish_path(self):
        result = generator.generate_from_games(["dummy pgn"], "someuser", None)
        assert result == []


class TestUserPuzzlePersistence:
    def test_save_then_load_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(generator, "USER_PUZZLES_DIR", tmp_path)

        puzzles = [
            {"PuzzleId": "a1", "FEN": START_FEN, "Moves": "e2e4 e7e5"},
            {"PuzzleId": "a2", "FEN": START_FEN, "Moves": "d2d4 d7d5"},
        ]
        generator.save_user_puzzles("tester", puzzles)

        loaded = generator.load_user_puzzles("tester")
        assert {p["PuzzleId"] for p in loaded} == {"a1", "a2"}

    def test_save_deduplicates_by_puzzle_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(generator, "USER_PUZZLES_DIR", tmp_path)

        generator.save_user_puzzles(
            "tester", [{"PuzzleId": "a1", "FEN": START_FEN, "Moves": "e2e4 e7e5"}]
        )
        generator.save_user_puzzles(
            "tester",
            [
                {"PuzzleId": "a1", "FEN": START_FEN, "Moves": "e2e4 e7e5"},  # duplicate
                {"PuzzleId": "a2", "FEN": START_FEN, "Moves": "d2d4 d7d5"},  # new
            ],
        )

        loaded = generator.load_user_puzzles("tester")
        assert len(loaded) == 2
        assert {p["PuzzleId"] for p in loaded} == {"a1", "a2"}

    def test_load_missing_user_returns_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr(generator, "USER_PUZZLES_DIR", tmp_path)
        assert generator.load_user_puzzles("nobody") == []

    def test_load_corrupt_file_returns_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr(generator, "USER_PUZZLES_DIR", tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "corrupt.json").write_text("not valid json", encoding="utf-8")
        assert generator.load_user_puzzles("corrupt") == []
