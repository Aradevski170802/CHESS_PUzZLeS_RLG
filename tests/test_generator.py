"""Unit tests for src/puzzles/generator.py.

No live Stockfish process is used. The pure surface is tested directly: score
conversion, the uniqueness check, sequence validation, tactic classification on
hand-verified positions, rating estimation, and the save/load JSON round trip.
`_extract_stockfish` is tested against a scripted fake engine that plays real legal
moves, which is enough to pin down how the gates shape a puzzle's solution line.
"""


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


def _info(board, move, cp):
    """One multipv line as python-chess returns it, scored for the side to move."""
    return {"pv": [move], "score": chess.engine.PovScore(chess.engine.Cp(cp), board.turn)}


class TestIsUniqueBest:
    BOARD = chess.Board()
    E4, D4 = chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")

    def lines(self, best_cp, second_cp):
        return [_info(self.BOARD, self.E4, best_cp), _info(self.BOARD, self.D4, second_cp)]

    def test_clear_margin_is_unique(self):
        assert generator._is_unique_best(self.lines(300, 300 - generator.MIN_CLARITY_CP),
                                         chess.WHITE)

    def test_small_margin_is_not_unique(self):
        assert not generator._is_unique_best(self.lines(300, 250), chess.WHITE)

    def test_single_legal_move_is_unique(self):
        assert generator._is_unique_best([_info(self.BOARD, self.E4, 0)], chess.WHITE)

    def test_no_lines_is_not_unique(self):
        assert not generator._is_unique_best([], chess.WHITE)

    def test_two_mates_are_not_unique(self):
        mate = chess.engine.PovScore(chess.engine.Mate(2), chess.WHITE)
        infos = [{"pv": [self.E4], "score": mate}, {"pv": [self.D4], "score": mate}]
        assert not generator._is_unique_best(infos, chess.WHITE)


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


PGN = """[Event "Test"]
[White "tester"]
[Black "opponent"]
[WhiteElo "1480"]
[BlackElo "1510"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ *
"""


def _position_before_ply(ply):
    board = chess.Board()
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6", "d2d4"]
    for uci in moves[:ply]:
        board.push_uci(uci)
    return board


class FakeEngine:
    """Scripted stand-in for Stockfish that only ever plays legal moves.

    The position before White's 5.d4 is the one puzzle start: there the engine
    prefers another move by 300 cp. Every other position scores +900 and is skipped
    as already decided. Opponent replies are always forced (200 cp gap); the gap
    between the player's first and second choice after the first move is
    `later_gap`, and at verification depth it is `verify_gap`.
    """

    def __init__(self, later_gap, verify_gap=300):
        start = _position_before_ply(8)
        self.critical_fen, self.player = start.fen(), start.turn
        self.played = chess.Move.from_uci("d2d4")
        self.later_gap, self.verify_gap = later_gap, verify_gap

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def configure(self, options):
        pass

    @staticmethod
    def _two_moves(board, avoid=None):
        moves = [m for m in sorted(board.legal_moves, key=lambda m: m.uci()) if m != avoid]
        return moves[0], moves[1]

    def analyse(self, board, limit, multipv=None):
        if multipv is None:                    # the move the player actually made
            return {"score": chess.engine.PovScore(chess.engine.Cp(0), board.turn)}
        if limit.time is not None:             # gates 1-2, fast detection pass
            if board.fen() != self.critical_fen:
                return [_info(board, next(iter(board.legal_moves)), 900)]
            best, second = self._two_moves(board, self.played)
            return [_info(board, best, 300), _info(board, second, 0)]
        if limit.depth == generator.VERIFY_DEPTH:
            best, second = self._two_moves(board, self.played)
            return [_info(board, best, 300), _info(board, second, 300 - self.verify_gap)]
        first, second = self._two_moves(board)  # gate 3, the continuation
        if board.turn == self.player:
            return [_info(board, first, 300), _info(board, second, 300 - self.later_gap)]
        return [_info(board, first, 0), _info(board, second, -200)]


class TestExtractStockfishGates:
    @pytest.fixture
    def extract(self, monkeypatch):
        monkeypatch.setattr(generator, "get_predictor", lambda: None)

        def run(engine):
            monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci",
                                lambda *args, **kwargs: engine)
            return generator._extract_stockfish(PGN, "tester", "fake-stockfish")
        return run

    def test_unique_continuation_is_kept(self, extract):
        puzzles = extract(FakeEngine(later_gap=300))
        assert len(puzzles) == 1
        moves = puzzles[0]["Moves"].split()
        assert moves[0] == "g8f6"                           # the opponent's setup move
        assert len(moves) == 2 + generator.CONTINUATION_MOVES
        assert puzzles[0]["playerMoves"] == 1 + generator.CONTINUATION_MOVES // 2
        assert puzzles[0]["playerRating"] == 1480            # White's rating: the tester

    def test_line_ends_before_a_player_move_with_two_solutions(self, extract):
        puzzles = extract(FakeEngine(later_gap=10))
        assert len(puzzles) == 1
        moves = puzzles[0]["Moves"].split()
        # Opponent's setup move and the player's first move only: the reply that
        # led to the ambiguous position is dropped, so the line ends on a player move.
        assert len(moves) == 2
        assert puzzles[0]["playerMoves"] == 1
        assert generator._validate_sequence(puzzles[0]["FEN"], moves)

    def test_first_move_not_unique_at_full_depth_is_rejected(self, extract):
        assert extract(FakeEngine(later_gap=300, verify_gap=50)) == []


class TestAnnotateMaia:
    def test_skipped_when_maia_is_not_installed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MAIA_DIR", str(tmp_path))
        puzzles = [{"FEN": START_FEN, "Moves": "e2e4 e7e5", "playerRating": 1500}]
        assert generator.annotate_maia(puzzles) == 0
        assert "maiaProbability" not in puzzles[0]

    def test_records_probability_and_level_per_rating(self, monkeypatch):
        from src.neural import maia
        opened = []

        class FakeMaia:
            def __init__(self, level):
                opened.append(level)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                pass

            def probability(self, board, move):
                assert board.turn == chess.BLACK and move.uci() in ("e7e5", "d7d5")
                return 0.25

        monkeypatch.setattr(maia, "available", lambda: True)
        monkeypatch.setattr(maia, "Maia", FakeMaia)
        puzzles = [
            {"FEN": START_FEN, "Moves": "e2e4 e7e5", "playerRating": 1480},      # -> 1500
            {"FEN": START_FEN, "Moves": "d2d4 d7d5", "playerRating": None},      # fallback
            {"FEN": START_FEN, "Moves": "c2c4 e7e5", "playerRating": 1490,
             "maiaProbability": 0.5},                                          # already done
        ]
        assert generator.annotate_maia(puzzles, rating=1000) == 2
        assert (puzzles[0]["maiaProbability"], puzzles[0]["maiaLevel"]) == (0.25, 1500)
        assert puzzles[1]["maiaLevel"] == 1100            # 1000 is below Maia's range
        assert puzzles[2]["maiaProbability"] == 0.5
        assert sorted(opened) == [1100, 1500]             # one engine per level


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
