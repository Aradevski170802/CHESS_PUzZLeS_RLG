"""Shared test configuration."""
import pytest


@pytest.fixture(autouse=True)
def _puzzlenet_off_by_default(monkeypatch):
    """Keep tests independent of whether a trained PuzzleNet model is installed:
    the analyzer, miner and app fall back to the rule-based tagger and heuristic
    rating. Tests of the neural path turn it back on explicitly."""
    from src.neural import predictor
    monkeypatch.setenv("PUZZLENET", "off")
    predictor.reset_cache()
    yield
    predictor.reset_cache()
