"""PuzzleNet inside the system: predictor loading, labeller choice, mined-puzzle
ratings and the IRT deviation. Uses a tiny random-weight model, so it checks the
plumbing, not the quality of the trained network."""
import chess
import numpy as np
import pytest

from src.neural import encoding as E
from src.neural import predictor as P
from src.neural.dataset import CATEGORIES, THEMES
from src.neural.network import NetConfig, PuzzleNet

FEN = "r1b1k1nr/pp3ppp/2n5/q1bQ4/4N3/6P1/PP2PP1P/R1B1KBNR w KQkq - 3 9"
MOVES = "c1d2 c5f2 e4f2 a5d5"


@pytest.fixture()
def tiny_model(tmp_path, monkeypatch):
    net = PuzzleNet(NetConfig(n_in=E.N_FEATURES, hidden=(8,), n_cat=len(CATEGORIES),
                              n_theme=len(THEMES)), seed=0)
    net.meta = {"encoding_version": E.ENCODING_VERSION, "features": "all",
                "categories": CATEGORIES, "themes": THEMES,
                "cont_mean": [0.0] * E.N_CONT, "cont_std": [1.0] * E.N_CONT,
                "rating_mean": 1500.0, "rating_sd": 500.0}
    path = tmp_path / "tiny.npz"
    net.save(path)
    monkeypatch.setenv("PUZZLENET", "on")
    monkeypatch.setenv("PUZZLENET_MODEL", str(path))
    monkeypatch.delenv("LABELLER", raising=False)
    P.reset_cache()
    yield path
    P.reset_cache()


def test_no_model_means_no_predictor(tmp_path, monkeypatch):
    monkeypatch.setenv("PUZZLENET", "on")
    monkeypatch.setenv("PUZZLENET_MODEL", str(tmp_path / "missing.npz"))
    P.reset_cache()
    assert P.get_predictor() is None


def test_predictor_loads_once_and_predicts(tiny_model):
    a = P.get_predictor()
    assert a is not None and P.get_predictor() is a
    pred = a.predict_puzzle(FEN, MOVES)
    assert pred.category in CATEGORIES
    assert 0.0 < pred.confidence <= 1.0
    assert sum(pred.probs.values()) == pytest.approx(1.0, abs=1e-4)
    assert pred.rating_sd > 0 and np.isfinite(pred.rating)


def test_predict_line_matches_predict_puzzle(tiny_model):
    net = P.get_predictor()
    board = chess.Board(FEN)
    board.push_uci("c1d2")
    line = [chess.Move.from_uci(u) for u in MOVES.split()[1:]]
    assert net.predict_line(board, line).probs == net.predict_puzzle(FEN, MOVES).probs


def test_encoding_version_mismatch_is_refused(tiny_model):
    net = PuzzleNet.load(tiny_model)
    net.meta["encoding_version"] = -1
    with pytest.raises(ValueError):
        P.PuzzleNetPredictor(net)


def test_labeller_choice(tiny_model, monkeypatch):
    from src.puzzles.labeller import active_labeller, label_line
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    line = [chess.Move.from_uci("a1a8")]
    assert active_labeller() == "neural"
    assert label_line(board, line) in CATEGORIES
    monkeypatch.setenv("LABELLER", "rules")
    assert active_labeller() == "rules"
    assert label_line(board, line) == "Mating Pattern"
    monkeypatch.setenv("LABELLER", "neural")
    monkeypatch.setenv("PUZZLENET", "off")
    P.reset_cache()
    assert active_labeller() == "rules"


def _mined(**kw):
    p = {"PuzzleId": "gen_x", "FEN": FEN, "Moves": MOVES, "Rating": 1234,
         "RatingDeviation": 150, "source": "generated", "PrimaryCategory": "Fork",
         "DifficultyTier": "Intermediate"}
    p.update(kw)
    return p


def test_rerate_records_the_network_estimate_without_changing_the_served_rating(tiny_model):
    """By default the network's difficulty is stored but not served: the app's own
    logs say the level does not transfer to a player's own positions."""
    from src.puzzles.generator import PUZZLENET_MIN_RD, rerate_puzzles
    mined, done, lichess = _mined(), _mined(puzzlenetRating=1500, Rating=999), \
        {"PuzzleId": "abc", "FEN": FEN, "Moves": MOVES, "Rating": 1500, "source": "lichess"}
    assert rerate_puzzles([mined, done, lichess]) == 1
    assert 400 <= mined["puzzlenetRating"] <= 3200
    assert mined["puzzlenetRd"] >= PUZZLENET_MIN_RD
    assert mined["Rating"] == 1234 and mined.get("ratingModel", "heuristic") == "heuristic"
    assert mined["PrimaryCategory"] == "Fork"           # category is never relabelled
    assert done["Rating"] == 999 and lichess["Rating"] == 1500


def test_mined_rating_env_switches_the_served_rating(tiny_model, monkeypatch):
    from src.puzzles.generator import rerate_puzzles, use_puzzlenet_rating
    monkeypatch.setenv("MINED_RATING", "puzzlenet")
    assert use_puzzlenet_rating()
    mined = _mined()
    assert rerate_puzzles([mined]) == 1
    assert mined["ratingModel"] == "puzzlenet"
    assert mined["heuristicRating"] == 1234
    assert mined["Rating"] == mined["puzzlenetRating"]
    assert mined["RatingDeviation"] == mined["puzzlenetRd"]


def test_rerate_is_a_no_op_without_a_model():
    from src.puzzles.generator import rerate_puzzles
    p = _mined()
    assert rerate_puzzles([p]) == 0 and "ratingModel" not in p


def test_load_user_puzzles_rerates_once_and_persists(tiny_model, tmp_path, monkeypatch):
    from src.puzzles import generator
    monkeypatch.setattr(generator, "USER_PUZZLES_DIR", tmp_path)
    generator.save_user_puzzles("u", [_mined()])
    assert generator.load_user_puzzles("u")[0].get("puzzlenetRating") is None  # default: no rerate
    first = generator.load_user_puzzles("u", rerate=True)[0]
    assert first["puzzlenetRating"] is not None
    again = generator.load_user_puzzles("u")[0]
    assert again["puzzlenetRating"] == first["puzzlenetRating"]


def test_puzzlenet_rd_floor():
    from src.puzzles.generator import PUZZLENET_MIN_RD, _puzzlenet_rd
    assert _puzzlenet_rd(10.0) == PUZZLENET_MIN_RD
    assert _puzzlenet_rd(240.4) == 240


def test_app_uses_the_puzzles_own_deviation():
    import importlib
    app = importlib.import_module("web.backend.app")
    assert app._puzzle_rd({"source": "lichess", "RatingDeviation": 300}) == app.DEFAULT_PUZZLE_RD
    assert app._puzzle_rd(_mined()) == app.MINED_PUZZLE_RD
    assert app._puzzle_rd(_mined(ratingModel="puzzlenet", RatingDeviation=210)) == 210.0
