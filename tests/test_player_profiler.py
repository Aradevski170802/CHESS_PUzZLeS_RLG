"""Tests for opportunity aggregation, priors and the accuracy fix — engine-free."""

import math

import pytest

from src.classifier.player_profiler import (
    MIN_OPPORTUNITIES,
    PlayerProfile,
    aggregate_opportunities,
    build_profile,
    profile_to_bandit_priors,
    profile_to_irt_prior,
)
# Bound at import, before the autouse fixture below replaces it, so the
# loader itself can still be tested.
from src.classifier.player_profiler import load_category_norms as _real_load_norms
from src.classifier.stockfish_analyzer import GameAnalysis, Opportunity
from src.data.puzzle_loader import WEAKNESS_CATEGORIES


@pytest.fixture(autouse=True)
def _no_population_norms(monkeypatch):
    """Keep these tests hermetic: the shipped norms file must not change them.
    TestPopulationNorms installs its own norms explicitly."""
    import src.classifier.player_profiler as pp
    monkeypatch.setattr(pp, "load_category_norms", lambda *a, **k: None)


def _opp(cat, hit):
    return Opportunity(category=cat, hit=hit, wp_loss=0.0 if hit else 20.0, phase="middlegame", ply=20)


def _game(opps, end_time=None, time_class="rapid"):
    return GameAnalysis(opportunities=opps, end_time=end_time, time_class=time_class,
                        num_moves=40, avg_cp_loss=40.0, avg_wp_loss=4.0)


class TestAccuracyFix:
    def test_uses_win_percent_loss(self):
        p = PlayerProfile(username="u", estimated_elo=1500, avg_cp_loss=40.0, avg_wp_loss=4.0)
        assert p.accuracy_estimate == pytest.approx(103.1668 * math.exp(-0.04354 * 4.0) - 3.1669)
        assert p.accuracy_estimate > 80

    def test_centipawn_only_profiles_are_converted_not_misread(self):
        # The old bug read ACPL 40 as 40 win-% points -> ~15 % accuracy.
        p = PlayerProfile(username="u", estimated_elo=1500, avg_cp_loss=40.0)
        assert p.accuracy_estimate > 80


class TestAggregation:
    def test_hit_rate_uses_opportunities_as_denominator(self):
        stats, overall = aggregate_opportunities([
            _game([_opp("Fork", True), _opp("Fork", False), _opp("Pin", True), _opp("Pin", True)]),
        ])
        assert stats["Fork"]["raw_n"] == 2
        assert overall == pytest.approx(0.75)
        # shrunk toward 0.75, so between the raw 0.5 and the overall rate
        assert 0.5 < stats["Fork"]["rate"] < 0.75
        assert stats["Pin"]["rate"] < 1.0

    def test_recent_games_weigh_more(self):
        old_miss_new_hit = [_game([_opp("Fork", True)], end_time=2000),
                            _game([_opp("Fork", False)], end_time=1000)]
        old_hit_new_miss = [_game([_opp("Fork", False)], end_time=2000),
                            _game([_opp("Fork", True)], end_time=1000)]
        a, _ = aggregate_opportunities(old_miss_new_hit)
        b, _ = aggregate_opportunities(old_hit_new_miss)
        assert a["Fork"]["hits"] > b["Fork"]["hits"]

    def test_bullet_counts_less_than_rapid(self):
        stats, _ = aggregate_opportunities([_game([_opp("Fork", False)], time_class="bullet")])
        assert stats["Fork"]["n"] == pytest.approx(0.5)


class TestPriors:
    def _profile_with(self, opps_per_cat):
        games = [_game([_opp(c, h) for c, h in opps_per_cat])]
        return build_profile(games, "u", 1500)

    def test_evidence_scaled_beta_priors(self):
        opps = [("Fork", False)] * 6 + [("Pin", True)] * 6
        profile = self._profile_with(opps)
        priors = profile_to_bandit_priors(profile)
        fork_a, fork_b = priors["Fork"]
        pin_a, pin_b = priors["Pin"]
        assert fork_a / (fork_a + fork_b) < pin_a / (pin_a + pin_b)
        # an unseen category gets only a weak prior
        unseen = next(c for c in WEAKNESS_CATEGORIES if c not in ("Fork", "Pin"))
        assert sum(priors[unseen]) == pytest.approx(2.0 + 2.0)
        assert all(a >= 1.0 and b >= 1.0 for a, b in priors.values())

    def test_budget_caps_prior_strength(self):
        profile = self._profile_with([("Fork", False)] * 40)
        a, b = profile_to_bandit_priors(profile)["Fork"]
        assert a + b <= 2.0 + 10.0 + 1e-9

    def test_falls_back_to_weakness_scores_without_opportunities(self):
        profile = PlayerProfile(username="u", estimated_elo=1500,
                                weakness_scores={"Fork": 0.8, "Pin": 0.2})
        priors = profile_to_bandit_priors(profile)
        assert priors["Fork"] == pytest.approx((2.0, 8.0))
        assert priors["Pin"] == pytest.approx((8.0, 2.0))

    def test_irt_prior_is_negative_for_the_weak_category(self):
        opps = [("Fork", False)] * 8 + [("Pin", True)] * 8 + [("Skewer", True)] * 4
        prior = profile_to_irt_prior(self._profile_with(opps))
        assert prior["Fork"][0] < 0 < prior["Pin"][0]
        assert prior["Fork"][1] < 0.6          # evidence tightened the prior

    def test_min_opportunities_threshold(self):
        profile = self._profile_with([("Fork", False)] * (MIN_OPPORTUNITIES - 1))
        assert not profile.has_opportunity_data


class TestPopulationNorms:
    """Hierarchical shrinkage: compare a category with what is typical for it."""

    def _games(self):
        # Quiet Move is hard for everyone: this player finds 3 of 10 (typical),
        # Fork is easy: finds 9 of 10 (typical too).
        opps = [("Quiet Move", i < 3) for i in range(10)] + [("Fork", i < 9) for i in range(10)]
        return [_game([_opp(c, h) for c, h in opps])]

    def test_without_norms_a_hard_category_looks_like_a_personal_weakness(self):
        stats, overall = aggregate_opportunities(self._games())
        assert stats["Quiet Move"]["expected"] == pytest.approx(overall)
        prior = profile_to_irt_prior(build_profile(self._games(), "u", 1500))
        assert prior["Quiet Move"][0] < -0.5

    def test_with_norms_a_typical_player_has_no_personal_offset(self, monkeypatch, tmp_path):
        import json
        import src.classifier.player_profiler as pp
        overall = 12 / 20
        # offsets that make 0.3 and 0.9 exactly the expected rates at this level
        norms = {"offsets": {"Quiet Move": pp._logit(0.3) - pp._logit(overall),
                             "Fork": pp._logit(0.9) - pp._logit(overall)},
                 "shrink_strength": 4.0}
        path = tmp_path / "norms.json"
        path.write_text(json.dumps(norms), encoding="utf-8")
        monkeypatch.setattr(pp, "NORMS_PATH", path)
        pp._NORMS_CACHE.clear()
        monkeypatch.setattr(pp, "load_category_norms", lambda path=path: norms)
        prior = profile_to_irt_prior(build_profile(self._games(), "u", 1500))
        assert abs(prior["Quiet Move"][0]) < 0.1
        assert abs(prior["Fork"][0]) < 0.1

    def test_missing_norms_file_means_fallback(self, tmp_path):
        assert _real_load_norms(tmp_path / "nope.json") is None

    def test_loader_reads_a_norms_file(self, tmp_path):
        import json
        path = tmp_path / "norms.json"
        path.write_text(json.dumps({"offsets": {"Fork": 0.1}, "shrink_strength": 8}), encoding="utf-8")
        assert _real_load_norms(path)["offsets"]["Fork"] == 0.1

    def test_shipped_norms_file_is_valid(self):
        import json
        import src.classifier.player_profiler as pp
        norms = json.loads(pp.NORMS_PATH.read_text(encoding="utf-8"))
        assert norms["shrink_strength"] >= 1
        assert norms["players"] >= 100
        for cat in ("Fork", "Pin", "Hanging Piece", "Mating Pattern"):
            assert cat in norms["offsets"]
        assert "username" not in json.dumps(norms).lower()   # aggregate statistics only

    def test_heavy_pooling_keeps_the_irt_prior_wide(self, monkeypatch):
        import src.classifier.player_profiler as pp
        norms = {"offsets": {"Fork": 0.0, "Quiet Move": 0.0}, "shrink_strength": 256.0}
        monkeypatch.setattr(pp, "load_category_norms", lambda *a, **k: norms)
        prior = profile_to_irt_prior(build_profile(self._games(), "u", 1500))
        # 10 game opportunities barely count against 256 pseudo-opportunities,
        # so the prior sd stays close to the default and puzzles can move it.
        assert prior["Quiet Move"][1] > 0.58
        assert abs(prior["Quiet Move"][0]) < 0.25
