"""Unit tests for src/classifier/ml_weakness_model.py.

These tests never touch real player data (only 4 real players currently
have mined-puzzle history -- far too few for a meaningful held-out-player
split, see the module docstring). Instead they validate the pipeline
mechanics -- feature extraction, synthetic data generation, cross-
validation, persistence, and the build_profile_ml() drop-in contract --
against synthetic data and hand-constructed GameAnalysis objects, none of
which require a live Stockfish process or network access.
"""

import math

import numpy as np
import pytest

from src.classifier.ml_weakness_model import (
    FEATURE_NAMES,
    _COARSE_FEATURE_NAMES,
    _profile_to_features,
    _simulate_one_player,
    build_profile_ml,
    cross_validate,
    fit_model,
    generate_synthetic_dataset,
    load_model,
    save_model,
    train_and_evaluate,
)
from src.classifier.player_profiler import PlayerProfile, build_profile
from src.classifier.stockfish_analyzer import GameAnalysis, MoveError
from src.data.puzzle_loader import WEAKNESS_CATEGORIES

DUMMY_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _move_error(phase="middlegame", severity="blunder", cp_loss=300, category="General"):
    return MoveError(
        move_number=10, fen_before=DUMMY_FEN, move_uci="e2e4",
        cp_loss=cp_loss, severity=severity, phase=phase, category=category,
    )


class TestProfileToFeatures:
    def test_known_profile_produces_expected_feature_vector(self):
        profile = PlayerProfile(username="tester", estimated_elo=1500)
        profile.games_analysed = 10
        profile.blunders_total = 4
        profile.mistakes_total = 6
        profile.inaccuracies_total = 10
        profile.errors_by_phase = {"middlegame": 12, "endgame": 4, "opening": 4}
        profile.avg_cp_loss = 60.0  # -> accuracy_estimate derived from this

        features = _profile_to_features(profile)

        assert features.shape == (len(FEATURE_NAMES),)
        total_errors = 20
        assert features[0] == pytest.approx(12 / total_errors)  # middlegame_rate
        assert features[1] == pytest.approx(4 / total_errors)  # endgame_rate
        assert features[2] == pytest.approx(4 / 10)  # blunders_per_game
        assert features[3] == pytest.approx(6 / 10)  # mistakes_per_game
        assert features[4] == pytest.approx(10 / 10)  # inaccuracies_per_game
        assert features[5] == pytest.approx(profile.accuracy_estimate / 100.0)
        assert features[6] == pytest.approx(math.log1p(10))
        # errors_by_category was never set -> every per-category rate is 0.
        assert np.all(features[len(_COARSE_FEATURE_NAMES):] == 0.0)

    def test_zero_games_and_zero_errors_do_not_divide_by_zero(self):
        profile = PlayerProfile(username="empty", estimated_elo=1200)
        features = _profile_to_features(profile)
        assert np.all(np.isfinite(features))

    def test_category_rate_features_reflect_errors_by_category(self):
        profile = PlayerProfile(username="tester", estimated_elo=1500)
        profile.games_analysed = 5
        profile.errors_by_phase = {"middlegame": 16, "endgame": 4}  # total 20
        profile.errors_by_category = {"Fork": 5, "Pin": 3}

        features = _profile_to_features(profile)

        fork_idx = len(_COARSE_FEATURE_NAMES) + WEAKNESS_CATEGORIES.index("Fork")
        pin_idx = len(_COARSE_FEATURE_NAMES) + WEAKNESS_CATEGORIES.index("Pin")
        skewer_idx = len(_COARSE_FEATURE_NAMES) + WEAKNESS_CATEGORIES.index("Skewer")

        assert features[fork_idx] == pytest.approx(5 / 20)
        assert features[pin_idx] == pytest.approx(3 / 20)
        assert features[skewer_idx] == pytest.approx(0.0)  # never recorded -> 0 rate


class TestSyntheticData:
    def test_simulate_one_player_returns_correct_shapes_and_ranges(self):
        rng = np.random.default_rng(0)
        features, labels = _simulate_one_player(rng)

        assert features.shape == (len(FEATURE_NAMES),)
        assert labels.shape == (len(WEAKNESS_CATEGORIES),)
        assert np.all((labels >= 0.0) & (labels <= 1.0))
        assert np.all(np.isfinite(features))
        # rate/accuracy features are bounded in [0, 1]
        assert 0.0 <= features[0] <= 1.0  # middlegame_rate
        assert 0.0 <= features[1] <= 1.0  # endgame_rate
        assert 0.0 <= features[5] <= 1.0  # accuracy_norm
        assert features[6] > 0.0  # log_games_analysed

    def test_category_rate_block_is_a_valid_probability_distribution(self):
        # The per-category block comes from a multinomial draw normalised by
        # its own total, so it must sum to 1 (a real category-count
        # distribution) regardless of which categories happen to be weak.
        rng = np.random.default_rng(3)
        features, _ = _simulate_one_player(rng)
        category_block = features[len(_COARSE_FEATURE_NAMES):]
        assert category_block.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.all(category_block >= 0.0)

    def test_generate_synthetic_dataset_shapes(self):
        X, Y = generate_synthetic_dataset(n_players=25, seed=1)
        assert X.shape == (25, len(FEATURE_NAMES))
        assert Y.shape == (25, len(WEAKNESS_CATEGORIES))
        assert np.all((Y >= 0.0) & (Y <= 1.0))

    def test_same_seed_is_reproducible(self):
        X1, Y1 = generate_synthetic_dataset(n_players=10, seed=7)
        X2, Y2 = generate_synthetic_dataset(n_players=10, seed=7)
        assert np.array_equal(X1, X2)
        assert np.array_equal(Y1, Y2)

    def test_different_seeds_produce_different_data(self):
        X1, _ = generate_synthetic_dataset(n_players=10, seed=1)
        X2, _ = generate_synthetic_dataset(n_players=10, seed=2)
        assert not np.array_equal(X1, X2)


class TestCrossValidation:
    def test_cross_validate_returns_one_fold_per_split(self):
        X, Y = generate_synthetic_dataset(n_players=60, seed=3)
        report = cross_validate(X, Y, n_splits=3, seed=3)

        assert report.n_players == 60
        assert report.n_splits == 3
        assert len(report.fold_metrics) == 3
        assert report.mean_mae > 0.0
        assert np.isfinite(report.mean_mae)
        for fm in report.fold_metrics:
            assert 0.0 <= fm.mae <= 1.0
            assert 0.0 <= fm.baseline_mae <= 1.0
            assert 0.0 <= fm.top_weak_category_in_top3 <= 1.0

    def test_model_is_not_dramatically_worse_than_naive_baseline_on_average(self):
        # A soft, low-flakiness signal-detection check: across enough held-out
        # (synthetic) players, a model that is actually learning something
        # from the feature/label correlation baked into the simulator should
        # not systematically lose to "always predict the training-set mean
        # vector". This does not assert the model wins every fold (noisy
        # categories mean it won't), only that it is competitive on average.
        X, Y = generate_synthetic_dataset(n_players=150, seed=11)
        report = cross_validate(X, Y, n_splits=5, seed=11)
        assert report.mean_mae < report.mean_baseline_mae * 1.15

    def test_fit_model_fits_training_data_much_better_than_naive_baseline(self):
        # A deterministic, low-risk sanity check on the model/training code
        # path itself (independent of CV-fold noise): a random forest fit on
        # its own training data should track it far more closely than a
        # constant "predict the mean" baseline would.
        X, Y = generate_synthetic_dataset(n_players=200, seed=5)
        model = fit_model(X, Y, seed=5)
        preds = model.predict(X)
        in_sample_mae = float(np.mean(np.abs(preds - Y)))

        baseline_pred = np.tile(Y.mean(axis=0), (len(Y), 1))
        baseline_mae = float(np.mean(np.abs(baseline_pred - Y)))

        assert in_sample_mae < baseline_mae


class TestTrainAndEvaluate:
    def test_train_and_evaluate_produces_report_and_fitted_model(self):
        report, model = train_and_evaluate(n_players=40, seed=2, n_splits=2)

        assert report.n_players == 40
        assert len(report.feature_importances) == len(FEATURE_NAMES)
        assert sum(report.feature_importances.values()) == pytest.approx(1.0, abs=1e-6)

        # The fitted model should be usable immediately for a single prediction.
        x = np.zeros((1, len(FEATURE_NAMES)))
        pred = model.predict(x)
        assert pred.shape == (1, len(WEAKNESS_CATEGORIES))


class TestPersistence:
    def test_save_and_load_model_round_trip(self, tmp_path):
        _, model = train_and_evaluate(n_players=30, seed=4, n_splits=2)
        path = tmp_path / "model.joblib"

        save_model(model, path)
        loaded = load_model(path)

        x = np.zeros((1, len(FEATURE_NAMES)))
        original_pred = model.predict(x)
        loaded_pred = loaded.predict(x)
        assert np.allclose(original_pred, loaded_pred)

    def test_load_missing_model_raises_helpful_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No trained weakness model"):
            load_model(tmp_path / "does_not_exist.joblib")


class TestBuildProfileMl:
    def _sample_analyses(self) -> list[GameAnalysis]:
        return [
            GameAnalysis(
                errors=[_move_error("middlegame", "blunder"), _move_error("middlegame", "mistake")],
                player_color="white", player_won=True,
                opening_eco="C50", opening_family="Italian Game",
                num_moves=40, avg_cp_loss=55.0, failed=False,
            ),
            GameAnalysis(
                errors=[_move_error("endgame", "inaccuracy")],
                player_color="black", player_won=False,
                opening_eco="B10", opening_family="Caro-Kann",
                num_moves=60, avg_cp_loss=40.0, failed=False,
            ),
        ]

    def test_returns_full_weakness_map_with_values_in_unit_interval(self, tmp_path):
        _, model = train_and_evaluate(n_players=30, seed=6, n_splits=2)
        model_path = tmp_path / "model.joblib"
        save_model(model, model_path)

        profile = build_profile_ml(self._sample_analyses(), "tester", 1500, model_path=model_path)

        assert set(profile.weakness_scores.keys()) == set(WEAKNESS_CATEGORIES)
        assert all(0.0 <= v <= 1.0 for v in profile.weakness_scores.values())

    def test_non_weakness_fields_match_rule_based_build_profile_exactly(self, tmp_path):
        # build_profile_ml() should only change *how* weakness_scores is
        # computed -- every other field must come out identical to calling
        # build_profile() directly, since it is a drop-in for the same slot.
        _, model = train_and_evaluate(n_players=30, seed=6, n_splits=2)
        model_path = tmp_path / "model.joblib"
        save_model(model, model_path)

        analyses = self._sample_analyses()
        rule_based = build_profile(analyses, "tester", 1500)
        ml_based = build_profile_ml(analyses, "tester", 1500, model_path=model_path)

        assert ml_based.username == rule_based.username
        assert ml_based.estimated_elo == rule_based.estimated_elo
        assert ml_based.games_analysed == rule_based.games_analysed
        assert ml_based.blunders_total == rule_based.blunders_total
        assert ml_based.mistakes_total == rule_based.mistakes_total
        assert ml_based.inaccuracies_total == rule_based.inaccuracies_total
        assert ml_based.errors_by_phase == rule_based.errors_by_phase
        assert ml_based.errors_by_category == rule_based.errors_by_category
        assert ml_based.avg_cp_loss == rule_based.avg_cp_loss
        assert ml_based.top_openings_white == rule_based.top_openings_white
        assert ml_based.top_openings_black == rule_based.top_openings_black
        # weakness_scores is the one field allowed (expected) to differ --
        # that's the entire point of build_profile_ml() -- so it is
        # deliberately not compared here.

    def test_no_successful_analyses_returns_flat_profile_without_requiring_a_model(self, tmp_path):
        failed_only = [GameAnalysis(failed=True)]
        # Deliberately point at a model file that does not exist -- this
        # branch must never need to load one.
        profile = build_profile_ml(
            failed_only, "nobody", 1200, model_path=tmp_path / "missing.joblib"
        )
        assert profile.weakness_scores == {cat: 0.5 for cat in WEAKNESS_CATEGORIES}

    def test_missing_model_file_raises_when_there_is_data_to_predict_from(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_profile_ml(
                self._sample_analyses(), "tester", 1500, model_path=tmp_path / "missing.joblib"
            )
