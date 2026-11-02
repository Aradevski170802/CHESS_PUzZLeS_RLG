"""Unit tests for the difficulty-aware IRT learner — no engine, no I/O."""

import math

import numpy as np
import pytest

from src.analysis.difficulty_fitter import Glicko2Rating, glicko2_update
from src.data.puzzle_loader import WEAKNESS_CATEGORIES
from src.recommender.irt_model import (
    LOGIT_SCALE,
    IRTLearner,
    logit_to_rating,
    rating_to_logit,
)

CAT = WEAKNESS_CATEGORIES[0]
OTHER = WEAKNESS_CATEGORIES[1]


def _theta_only_learner(rating, rd, drift=0.0):
    n = len(WEAKNESS_CATEGORIES) + 1
    mean = np.zeros(n)
    mean[0] = rating_to_logit(rating)
    cov = np.diag([(rd / LOGIT_SCALE) ** 2] + [0.36] * (n - 1))
    return IRTLearner(mean=mean, cov=cov, drift_theta=drift, drift_delta=0.0)


class TestScale:
    def test_rating_logit_roundtrip(self):
        assert logit_to_rating(rating_to_logit(1834.5)) == pytest.approx(1834.5)

    def test_400_points_is_ten_to_one_odds(self):
        assert math.exp(rating_to_logit(1900) - rating_to_logit(1500)) == pytest.approx(10.0, rel=1e-4)


class TestGlickoEquivalence:
    """With only θ in play, the update must be term-for-term Glicko."""

    def test_matches_glicko_formula_exactly(self):
        sigma = 0.06
        learner = _theta_only_learner(1500, 200, drift=sigma ** 2)
        learner.update("Not A Category", puzzle_rating=1400, solved=True, puzzle_rd=30)

        mu, phi = 0.0, 200 / LOGIT_SCALE
        mu_j, phi_j = rating_to_logit(1400), 30 / LOGIT_SCALE
        g = 1 / math.sqrt(1 + 3 * phi_j ** 2 / math.pi ** 2)
        e = 1 / (1 + math.exp(-g * (mu - mu_j)))
        v = 1 / (g ** 2 * e * (1 - e))
        phi_star2 = phi ** 2 + sigma ** 2
        phi_new2 = 1 / (1 / phi_star2 + 1 / v)
        mu_new = mu + phi_new2 * g * (1.0 - e)

        assert learner.mean[0] == pytest.approx(mu_new, abs=1e-9)
        assert learner.cov[0, 0] == pytest.approx(phi_new2, abs=1e-9)

    def test_agrees_with_the_validated_glicko2_implementation(self):
        # With a tiny system constant τ, Glicko-2 keeps volatility ~fixed, so
        # its update should land on the same rating and RD as the IRT step.
        learner = _theta_only_learner(1500, 200, drift=0.06 ** 2)
        learner.update("Not A Category", puzzle_rating=1700, solved=False, puzzle_rd=100)
        glicko = glicko2_update(
            Glicko2Rating(1500, 200, 0.06), [(Glicko2Rating(1700, 100), 0.0)], tau=0.001,
        )
        assert logit_to_rating(learner.mean[0]) == pytest.approx(glicko.rating, abs=0.5)
        assert LOGIT_SCALE * math.sqrt(learner.cov[0, 0]) == pytest.approx(glicko.rd, abs=0.5)


class TestPrediction:
    def test_harder_puzzles_are_less_likely_to_be_solved(self):
        learner = IRTLearner.new(elo=1500)
        assert learner.predict(CAT, 1200) > learner.predict(CAT, 1500) > learner.predict(CAT, 1900)

    def test_average_player_on_matched_puzzle_is_near_even(self):
        learner = IRTLearner.new(elo=1500)
        assert learner.predict(CAT, 1500) == pytest.approx(0.5, abs=1e-9)

    def test_update_returns_the_pre_outcome_prediction(self):
        learner = IRTLearner.new(elo=1500, drift_theta=0.0, drift_delta=0.0)
        expected = learner.predict(CAT, 1600)
        assert learner.update(CAT, 1600, True) == pytest.approx(expected)


class TestLearning:
    def test_solve_raises_and_fail_lowers_category_ability(self):
        up = IRTLearner.new(elo=1500)
        down = IRTLearner.new(elo=1500)
        before = up.category_rating(CAT)[0]
        up.update(CAT, 1500, True)
        down.update(CAT, 1500, False)
        assert up.category_rating(CAT)[0] > before > down.category_rating(CAT)[0]

    def test_failing_a_hard_puzzle_is_weaker_evidence_than_failing_an_easy_one(self):
        # The property the Beta bandit lacks: difficulty-adjusted evidence.
        hard = IRTLearner.new(elo=1500)
        easy = IRTLearner.new(elo=1500)
        hard.update(CAT, 2200, False)
        easy.update(CAT, 1000, False)
        i = WEAKNESS_CATEGORIES.index(CAT) + 1
        assert abs(hard.mean[i]) < abs(easy.mean[i])

    def test_evidence_shrinks_uncertainty(self):
        learner = IRTLearner.new(elo=1500, drift_theta=0.0, drift_delta=0.0)
        rd_before = learner.category_rating(CAT)[1]
        for _ in range(20):
            learner.update(CAT, 1500, True)
        assert learner.category_rating(CAT)[1] < rd_before

    def test_covariance_stays_positive_definite(self):
        rng = np.random.default_rng(1)
        learner = IRTLearner.new(elo=1400)
        for _ in range(300):
            cat = WEAKNESS_CATEGORIES[rng.integers(len(WEAKNESS_CATEGORIES))]
            learner.update(cat, rng.uniform(900, 2100), bool(rng.random() < 0.5))
        assert np.all(np.linalg.eigvalsh(learner.cov) > 0)
        assert np.allclose(learner.cov, learner.cov.T)

    def test_delta_prior_is_applied(self):
        learner = IRTLearner.new(elo=1500, delta_prior={CAT: (-0.8, 0.3)})
        i = WEAKNESS_CATEGORIES.index(CAT) + 1
        assert learner.mean[i] == pytest.approx(-0.8)
        assert learner.cov[i, i] == pytest.approx(0.09)


class TestDecisions:
    def test_selects_the_category_the_player_keeps_failing(self):
        learner = IRTLearner.new(elo=1500)
        for _ in range(15):
            learner.update(CAT, 1400, False)
        for other in WEAKNESS_CATEGORIES[1:]:
            for _ in range(3):
                learner.update(other, 1400, True)
        rng = np.random.default_rng(0)
        picks = [learner.select_category(rng) for _ in range(200)]
        assert picks.count(CAT) / len(picks) >= 0.8

    def test_allowed_restricts_the_choice(self):
        learner = IRTLearner.new(elo=1500)
        rng = np.random.default_rng(0)
        for _ in range(50):
            assert learner.select_category(rng, allowed=[OTHER]) == OTHER

    def test_selection_probabilities_form_a_distribution(self):
        probs = IRTLearner.new(elo=1500).selection_probabilities(
            n_samples=500, rng=np.random.default_rng(0))
        assert set(probs) == set(WEAKNESS_CATEGORIES)
        assert sum(probs.values()) == pytest.approx(1.0)

    def test_target_rating_hits_the_requested_success_probability(self):
        learner = IRTLearner.new(elo=1500, delta_prior={CAT: (-0.5, 0.3)})
        i = WEAKNESS_CATEGORIES.index(CAT) + 1
        ability = learner.mean[0] + learner.mean[i]
        target = learner.target_rating(CAT, p_target=0.7)
        z = ability - rating_to_logit(target)
        assert 1 / (1 + math.exp(-z)) == pytest.approx(0.7)

    def test_weaker_category_gets_an_easier_target(self):
        learner = IRTLearner.new(elo=1500, delta_prior={CAT: (-1.0, 0.3), OTHER: (1.0, 0.3)})
        assert learner.target_rating(CAT) < learner.target_rating(OTHER)


class TestReadoutsAndPersistence:
    def test_weakness_map_ranks_the_weak_category_first(self):
        learner = IRTLearner.new(elo=1500, delta_prior={CAT: (-1.2, 0.2)})
        assert learner.top_weaknesses(1)[0]["category"] == CAT
        assert all(0 < v < 1 for v in learner.weakness_map().values())

    def test_roundtrip(self):
        learner = IRTLearner.new(elo=1600)
        learner.update(CAT, 1550, False)
        restored = IRTLearner.from_dict(learner.to_dict())
        assert np.allclose(restored.mean, learner.mean, atol=1e-6)
        assert np.allclose(restored.cov, learner.cov, atol=1e-7)
        assert restored.n_updates == 1

    def test_from_dict_with_mismatched_categories_falls_back_to_prior(self):
        payload = IRTLearner.new(elo=1500).to_dict()
        payload["categories"] = ["Something", "Else"]
        restored = IRTLearner.from_dict(payload)
        assert restored.mean.shape == (len(WEAKNESS_CATEGORIES) + 1,)

    def test_from_history_replays_only_valid_attempts(self):
        history = [
            {"category": CAT, "rating": 1500, "solved": False},
            {"category": CAT, "rating": 0, "solved": True},        # no rating: skipped
            {"category": OTHER, "rating": 1400, "solved": True},
            {"category": CAT, "rating": 1450},                      # no outcome: skipped
        ]
        learner = IRTLearner.from_history(history, elo=1500)
        assert learner.n_updates == 2


class TestFamilyPrior:
    def test_covariance_is_positive_definite(self):
        from src.recommender.irt_model import family_covariance
        assert np.all(np.linalg.eigvalsh(family_covariance()) > 0)
        learner = IRTLearner.new(elo=1500, family_prior=True)
        assert np.all(np.linalg.eigvalsh(learner.cov) > 0)

    def test_marginal_variance_is_unchanged(self):
        plain = IRTLearner.new(elo=1500)
        fam = IRTLearner.new(elo=1500, family_prior=True)
        assert np.allclose(np.diag(plain.cov), np.diag(fam.cov))

    def test_evidence_is_shared_within_a_family_only(self):
        learner = IRTLearner.new(elo=1500, family_prior=True, drift_theta=0.0, drift_delta=0.0)
        idx = {c: WEAKNESS_CATEGORIES.index(c) + 1 for c in ("Pin", "Skewer", "Rook Endgame")}
        for _ in range(8):
            learner.update("Pin", 1500, False)
        # Skewer (same "line" family) is pulled down relative to an unrelated family.
        assert learner.mean[idx["Skewer"]] < learner.mean[idx["Rook Endgame"]]

    def test_plain_prior_keeps_categories_independent(self):
        learner = IRTLearner.new(elo=1500, drift_theta=0.0, drift_delta=0.0)
        i_s = WEAKNESS_CATEGORIES.index("Skewer") + 1
        i_r = WEAKNESS_CATEGORIES.index("Rook Endgame") + 1
        for _ in range(8):
            learner.update("Pin", 1500, False)
        assert learner.mean[i_s] == pytest.approx(learner.mean[i_r])
