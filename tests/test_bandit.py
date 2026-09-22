"""Unit tests for the Thompson Sampling bandit — no chess engine, no I/O required."""

import numpy as np
import pytest

from src.recommender.bandit import ArmState, ThompsonBandit
from src.data.puzzle_loader import WEAKNESS_CATEGORIES


class TestArmState:
    def test_default_is_uniform_prior(self):
        arm = ArmState()
        assert arm.alpha == 1
        assert arm.beta == 1
        assert arm.mean == pytest.approx(0.5)

    def test_mean_reflects_alpha_beta(self):
        arm = ArmState(alpha=8, beta=2)
        assert arm.mean == pytest.approx(0.8)

    def test_to_dict(self):
        arm = ArmState(alpha=3, beta=7)
        assert arm.to_dict() == {"alpha": 3, "beta": 7}


class TestThompsonBanditInit:
    def test_default_has_one_arm_per_category(self):
        bandit = ThompsonBandit()
        assert set(bandit.arms.keys()) == set(WEAKNESS_CATEGORIES)
        assert all(arm.alpha == 1 and arm.beta == 1 for arm in bandit.arms.values())

    def test_priors_are_applied_to_known_categories(self):
        cat = WEAKNESS_CATEGORIES[0]
        bandit = ThompsonBandit(priors={cat: (2, 8)})
        assert bandit.arms[cat].alpha == 2
        assert bandit.arms[cat].beta == 8

    def test_categories_without_a_prior_stay_default(self):
        cat = WEAKNESS_CATEGORIES[0]
        other = WEAKNESS_CATEGORIES[1]
        bandit = ThompsonBandit(priors={cat: (2, 8)})
        assert bandit.arms[other].alpha == 1
        assert bandit.arms[other].beta == 1

    def test_unknown_prior_category_is_ignored_without_error(self):
        bandit = ThompsonBandit(priors={"Not A Real Category": (5, 5)})
        assert "Not A Real Category" not in bandit.arms
        assert len(bandit.arms) == len(WEAKNESS_CATEGORIES)

    def test_prior_values_are_clamped_to_at_least_one(self):
        cat = WEAKNESS_CATEGORIES[0]
        bandit = ThompsonBandit(priors={cat: (0, -5)})
        assert bandit.arms[cat].alpha == 1
        assert bandit.arms[cat].beta == 1


class TestUpdate:
    def test_solved_increments_alpha_and_streak(self):
        bandit = ThompsonBandit()
        cat = WEAKNESS_CATEGORIES[0]
        bandit.update(cat, True)
        assert bandit.arms[cat].alpha == 2
        assert bandit.arms[cat].beta == 1
        assert bandit.streak == 1
        assert bandit.best_streak == 1
        assert bandit.history == [{"cat": cat, "solved": True}]

    def test_failed_increments_beta_and_resets_streak(self):
        bandit = ThompsonBandit()
        cat = WEAKNESS_CATEGORIES[0]
        bandit.update(cat, True)
        bandit.update(cat, True)
        bandit.update(cat, False)
        assert bandit.arms[cat].beta == 2
        assert bandit.streak == 0
        assert bandit.best_streak == 2  # high-water mark is retained

    def test_unknown_category_is_a_silent_noop(self):
        bandit = ThompsonBandit()
        bandit.update("Not A Real Category", True)
        assert bandit.history == []
        assert bandit.streak == 0


class TestReadouts:
    def test_solve_rate_matches_arm_mean(self):
        bandit = ThompsonBandit()
        cat = WEAKNESS_CATEGORIES[0]
        bandit.arms[cat] = ArmState(alpha=3, beta=1)
        assert bandit.solve_rate(cat) == pytest.approx(0.75)

    def test_solve_rate_unknown_category_defaults_to_half(self):
        bandit = ThompsonBandit()
        assert bandit.solve_rate("Not A Real Category") == 0.5

    def test_weakness_map_covers_every_category(self):
        bandit = ThompsonBandit()
        wmap = bandit.weakness_map()
        assert set(wmap.keys()) == set(WEAKNESS_CATEGORIES)
        assert all(0.0 <= v <= 1.0 for v in wmap.values())

    def test_top_weaknesses_orders_lowest_solve_rate_first(self):
        bandit = ThompsonBandit()
        weak_cat = WEAKNESS_CATEGORIES[0]
        strong_cat = WEAKNESS_CATEGORIES[1]
        bandit.arms[weak_cat] = ArmState(alpha=1, beta=9)     # mean 0.1
        bandit.arms[strong_cat] = ArmState(alpha=9, beta=1)   # mean 0.9
        top = bandit.top_weaknesses(n=3)
        assert len(top) == 3
        assert top[0]["category"] == weak_cat
        assert top[0]["weakness"] == pytest.approx(0.1)

    def test_session_accuracy_with_no_history_is_zero(self):
        bandit = ThompsonBandit()
        assert bandit.session_accuracy() == 0.0

    def test_session_accuracy_reflects_solve_ratio(self):
        bandit = ThompsonBandit()
        cat = WEAKNESS_CATEGORIES[0]
        bandit.update(cat, True)
        bandit.update(cat, True)
        bandit.update(cat, True)
        bandit.update(cat, False)
        assert bandit.session_accuracy() == pytest.approx(0.75)

    def test_puzzles_played_matches_history_length(self):
        bandit = ThompsonBandit()
        cat = WEAKNESS_CATEGORIES[0]
        bandit.update(cat, True)
        bandit.update(cat, False)
        assert bandit.puzzles_played() == 2


class TestSelectOne:
    def test_returns_a_known_category(self):
        bandit = ThompsonBandit()
        for _ in range(20):
            assert bandit.select_one() in WEAKNESS_CATEGORIES

    def test_strongly_favours_the_weaker_arm(self):
        # A near-zero-mean arm vs. a near-one-mean arm should be picked
        # overwhelmingly more often than chance across many trials.
        np.random.seed(0)
        bandit = ThompsonBandit()
        weak_cat = WEAKNESS_CATEGORIES[0]
        strong_cat = WEAKNESS_CATEGORIES[1]
        bandit.arms[weak_cat] = ArmState(alpha=1, beta=200)
        bandit.arms[strong_cat] = ArmState(alpha=200, beta=1)
        for other_cat in WEAKNESS_CATEGORIES[2:]:
            bandit.arms[other_cat] = ArmState(alpha=5, beta=5)  # neutral, mean 0.5

        picks = [bandit.select_one() for _ in range(30)]
        weak_pick_rate = picks.count(weak_cat) / len(picks)
        assert weak_pick_rate >= 0.8


class TestPersistence:
    def test_to_dict_from_dict_roundtrip(self):
        bandit = ThompsonBandit()
        cat = WEAKNESS_CATEGORIES[0]
        bandit.update(cat, True)
        bandit.update(cat, False)

        restored = ThompsonBandit.from_dict(bandit.to_dict())

        assert restored.arms[cat].alpha == bandit.arms[cat].alpha
        assert restored.arms[cat].beta == bandit.arms[cat].beta
        assert restored.history == bandit.history
        assert restored.streak == bandit.streak
        assert restored.best_streak == bandit.best_streak

    def test_from_dict_ignores_unknown_categories(self):
        payload = {
            "arms": {"Not A Real Category": {"alpha": 5, "beta": 5}},
            "history": [],
            "streak": 0,
            "best_streak": 0,
        }
        restored = ThompsonBandit.from_dict(payload)
        assert "Not A Real Category" not in restored.arms
        assert len(restored.arms) == len(WEAKNESS_CATEGORIES)

    def test_roundtrip_preserves_discount_and_anchors(self):
        cat = WEAKNESS_CATEGORIES[0]
        bandit = ThompsonBandit(priors={cat: (2.5, 7.5)}, discount=0.95)
        restored = ThompsonBandit.from_dict(bandit.to_dict())
        assert restored.discount == 0.95
        assert restored.prior_anchor[cat] == (2.5, 7.5)

    def test_old_format_without_anchors_still_loads(self):
        cat = WEAKNESS_CATEGORIES[0]
        payload = {"arms": {cat: {"alpha": 4, "beta": 2}}, "history": [],
                   "streak": 0, "best_streak": 0}
        restored = ThompsonBandit.from_dict(payload)
        assert restored.arms[cat].alpha == 4
        assert restored.discount == 1.0


class TestFractionalPriors:
    def test_float_priors_are_kept_exactly(self):
        cat = WEAKNESS_CATEGORIES[0]
        bandit = ThompsonBandit(priors={cat: (3.25, 6.75)})
        assert bandit.arms[cat].alpha == pytest.approx(3.25)
        assert bandit.arms[cat].beta == pytest.approx(6.75)


class TestDiscounting:
    def test_invalid_discount_is_rejected(self):
        with pytest.raises(ValueError):
            ThompsonBandit(discount=0.0)
        with pytest.raises(ValueError):
            ThompsonBandit(discount=1.5)

    def test_no_discount_keeps_exact_counts(self):
        bandit = ThompsonBandit()
        cat = WEAKNESS_CATEGORIES[0]
        for _ in range(10):
            bandit.update(cat, False)
        assert bandit.arms[cat].beta == pytest.approx(11.0)

    def test_discount_bounds_the_evidence_an_arm_can_hold(self):
        # Decay-then-increment has fixed point x = γx + 1, i.e. evidence
        # converges to 1/(1-γ) pseudo-observations -- 10 for γ = 0.9.
        bandit = ThompsonBandit(discount=0.9)
        cat = WEAKNESS_CATEGORIES[0]
        for _ in range(500):
            bandit.update(cat, False)
        assert bandit.arms[cat].beta - 1.0 == pytest.approx(1 / (1 - 0.9), rel=1e-3)

    def test_unplayed_arms_decay_back_to_their_prior(self):
        weak, other = WEAKNESS_CATEGORIES[0], WEAKNESS_CATEGORIES[1]
        bandit = ThompsonBandit(priors={other: (2, 8)}, discount=0.9)
        bandit.arms[other].beta = 30.0            # lots of old evidence
        for _ in range(100):
            bandit.update(weak, True)
        assert bandit.arms[other].beta == pytest.approx(8.0, abs=0.01)

    def test_recent_evidence_outweighs_old_evidence(self):
        # A player who failed forks early and then improved should look
        # stronger under discounting than under the stationary bandit.
        cat = WEAKNESS_CATEGORIES[0]
        stationary, discounted = ThompsonBandit(), ThompsonBandit(discount=0.9)
        for b in (stationary, discounted):
            for _ in range(30):
                b.update(cat, False)
            for _ in range(15):
                b.update(cat, True)
        assert discounted.solve_rate(cat) > stationary.solve_rate(cat)


class TestSelectionProbabilities:
    def test_they_form_a_distribution(self):
        probs = ThompsonBandit().selection_probabilities(
            n_samples=500, rng=np.random.default_rng(0))
        assert set(probs) == set(WEAKNESS_CATEGORIES)
        assert sum(probs.values()) == pytest.approx(1.0)

    def test_they_concentrate_on_the_weak_arm(self):
        bandit = ThompsonBandit()
        weak = WEAKNESS_CATEGORIES[0]
        bandit.arms[weak] = ArmState(alpha=1, beta=60)
        for other in WEAKNESS_CATEGORIES[1:]:
            bandit.arms[other] = ArmState(alpha=30, beta=10)
        probs = bandit.selection_probabilities(n_samples=2000, rng=np.random.default_rng(0))
        assert probs[weak] > 0.95

    def test_seeded_rng_makes_selection_reproducible(self):
        bandit = ThompsonBandit()
        a = [bandit.select_one(np.random.default_rng(7)) for _ in range(5)]
        b = [bandit.select_one(np.random.default_rng(7)) for _ in range(5)]
        assert a == b
