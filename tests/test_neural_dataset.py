"""PuzzleNet dataset: targets, splits, harness reproduction, and the block loader."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from src.neural import encoding as E
from src.neural.dataset import (CATEGORIES, TEST, THEMES, TRAIN, VAL, BlockLoader,
                                PuzzleData, category_of, feature_columns, hash_split,
                                make_input, theme_vector)


def test_hash_split_is_deterministic_with_the_intended_proportions():
    ids = [f"p{i:06d}" for i in range(40000)]
    splits = np.array([hash_split(i) for i in ids])
    assert [hash_split(i) for i in ids[:50]] == list(splits[:50])
    assert (splits == VAL).mean() == pytest.approx(0.02, abs=0.004)
    assert (splits == TEST).mean() == pytest.approx(0.03, abs=0.004)
    assert (splits == TRAIN).mean() == pytest.approx(0.95, abs=0.006)


def test_category_uses_motif_priority_not_tag_order():
    # "endgame" sorts before "fork" but the motif must win (the bug MOTIF_PRIORITY fixed).
    assert CATEGORIES[category_of("crushing endgame fork short")] == "Fork"
    assert CATEGORIES[category_of("advantage middlegame short")] == "General"


def test_theme_targets_exclude_source_and_length_tags():
    for tag in ("master", "masterVsMaster", "superGM", "oneMove", "short", "long", "veryLong"):
        assert tag not in THEMES
    v = theme_vector("fork master short middlegame")
    assert v.sum() == 2 and v[THEMES.index("fork")] == 1 and v[THEMES.index("middlegame")] == 1


def test_harness_reproduction_matches_validate_labeller(monkeypatch):
    """build_dataset.harness_ids draws the same puzzles as validate_labeller.load."""
    import scripts.research.validate_labeller as VL
    from scripts.neural.build_dataset import harness_ids
    from src.data.puzzle_loader import resolve_primary_category

    rng = np.random.default_rng(0)
    pool = ["fork middlegame", "pin endgame", "mateIn1 mate", "advantage short",
            "enPassant", "skewer crushing", "rookEndgame endgame"]
    n = 5000
    table = pa.table({
        "PuzzleId": [f"id{i:05d}" for i in range(n)],
        "FEN": ["8/8/8/8/8/8/8/8 w - - 0 1"] * n,
        "Moves": ["e2e4"] * n,
        "Themes": list(rng.choice(pool, n, p=[.3, .15, .3, .15, .01, .04, .05])),
    })
    monkeypatch.setattr(VL.pq, "read_table", lambda path, columns=None: table.select(columns))
    ref = set()
    for seed in (11,):
        u, st = VL.load(seed, 300, 40)
        ref |= set(u["PuzzleId"]) | set(st["PuzzleId"])
    df = table.select(["PuzzleId", "Themes"]).to_pandas()
    mine = harness_ids(df["PuzzleId"], df["Themes"].map(resolve_primary_category),
                       uniform_n=300, per_class=40)
    assert mine == ref


def test_feature_subsets_partition_the_input():
    raw, eng, full = feature_columns("raw"), feature_columns("engineered"), feature_columns("all")
    assert len(full) == E.N_FEATURES
    assert len(raw) + len(eng) == E.N_FEATURES
    assert not set(raw) & set(eng)
    assert E.BIT_NAMES[raw[-1]].startswith("move_m3")
    with pytest.raises(ValueError):
        feature_columns("nope")


def test_make_input_standardises_only_the_continuous_part():
    bits = np.random.default_rng(1).integers(0, 2, (4, E.N_BITS)).astype(np.uint8)
    cont = np.full((4, E.N_CONT), 3.0, dtype=np.float16)
    x = make_input(bits, cont, np.full(E.N_CONT, 1.0), np.full(E.N_CONT, 2.0))
    assert x.dtype == np.float32 and x.shape == (4, E.N_FEATURES)
    assert np.array_equal(x[:, :E.N_BITS], bits)
    assert np.allclose(x[:, E.N_BITS:], 1.0)
    assert make_input(bits, cont, 0.0, 1.0, feature_columns("raw")).shape == (4, len(feature_columns("raw")))


def _tiny_data(n=500, seed=0) -> PuzzleData:
    rng = np.random.default_rng(seed)
    bits = np.packbits(rng.integers(0, 2, (n, E.N_BITS)).astype(np.uint8), axis=1)
    return PuzzleData(
        bits=bits, cont=rng.standard_normal((n, E.N_CONT)).astype(np.float16),
        cat=rng.integers(0, len(CATEGORIES), n).astype(np.uint8),
        themes=np.packbits(rng.integers(0, 2, (n, len(THEMES))).astype(np.uint8), axis=1),
        rating=rng.integers(400, 3000, n).astype(np.int16),
        rd=rng.integers(70, 120, n).astype(np.int16),
        nb_plays=np.full(n, 100, np.int32), n_moves=np.full(n, 4, np.uint8),
        split=rng.choice([TRAIN, VAL], n, p=[0.8, 0.2]).astype(np.uint8),
        puzzle_id=np.array([f"x{i}" for i in range(n)]), info={})


@pytest.mark.parametrize("split", [TRAIN, VAL])
def test_block_loader_visits_every_row_once_per_epoch(split):
    d = _tiny_data()
    idx = d.indices(split)
    loader = BlockLoader(d, idx, batch_size=32, cont_mean=np.zeros(E.N_CONT),
                         cont_std=np.ones(E.N_CONT), rating_mean=1500.0, rating_sd=500.0,
                         block_rows=64, seed=3)
    seen = []
    for b in loader.epoch():
        assert b.x.shape[1] == E.N_FEATURES and b.theme.shape[1] == len(THEMES)
        seen.append(b)
    ratings = np.concatenate([b.rating for b in seen]) * 500.0 + 1500.0
    assert len(ratings) == len(idx)
    assert np.allclose(np.sort(ratings), np.sort(d.rating[idx].astype(float)), atol=1e-2)
    assert len(seen) >= len(loader)


def test_block_loader_rows_line_up_with_targets():
    d = _tiny_data(n=300, seed=4)
    loader = BlockLoader(d, d.indices(TRAIN), batch_size=50, cont_mean=np.zeros(E.N_CONT),
                         cont_std=np.ones(E.N_CONT), rating_mean=0.0, rating_sd=1.0)
    for rows, b in loader.all_batches():
        expect = np.unpackbits(d.bits[rows], axis=1, count=E.N_BITS)
        assert np.array_equal(b.x[:, :E.N_BITS], expect)
        assert np.array_equal(b.cat, d.cat[rows])
        assert np.allclose(b.rating, d.rating[rows])


class TestTruncatedLineAugmentation:
    """Training mixes in a second encoding of the same puzzles with their lines cut
    short, because game analysis hands the labeller a cut engine line rather than a
    puzzle solution (scripts/neural/engine_pv_check.py)."""

    def _pair(self):
        main = _tiny_data(n=400, seed=11)
        aug = _tiny_data(n=400, seed=12)          # same rows, different features
        aug.cat, aug.rating, aug.themes = main.cat, main.rating, main.themes
        return main, aug

    def _loader(self, main, aug, prob):
        return BlockLoader(main, main.indices(TRAIN), batch_size=64,
                           cont_mean=np.zeros(E.N_CONT), cont_std=np.ones(E.N_CONT),
                           rating_mean=0.0, rating_sd=1.0, augment=aug, augment_prob=prob,
                           block_rows=128, seed=5)

    def test_augmented_rows_are_mixed_in_at_about_the_given_rate(self):
        main, aug = self._pair()
        rows = np.sort(main.indices(TRAIN))
        packed, _ = self._loader(main, aug, 0.5)._read(rows)
        from_aug = (packed == np.asarray(aug.bits[rows])).all(axis=1)
        from_main = (packed == np.asarray(main.bits[rows])).all(axis=1)
        assert (from_aug | from_main).all()          # every row came from one of the two
        assert from_aug.mean() == pytest.approx(0.5, abs=0.12)

    def test_probability_zero_means_the_main_encoding_only(self):
        main, aug = self._pair()
        rows = np.sort(main.indices(TRAIN))[:64]
        packed, _ = self._loader(main, aug, 0.0)._read(rows)
        assert np.array_equal(packed, np.asarray(main.bits[rows]))

    def test_probability_one_means_the_augmented_encoding_only(self):
        main, aug = self._pair()
        rows = np.sort(main.indices(TRAIN))[:64]
        packed, _ = self._loader(main, aug, 1.0)._read(rows)
        assert np.array_equal(packed, np.asarray(aug.bits[rows]))

    def test_targets_always_come_from_the_main_dataset(self):
        main, aug = self._pair()
        aug.cat = (main.cat + 1) % len(CATEGORIES)          # deliberately wrong labels
        for rows, b in self._loader(main, aug, 1.0).all_batches():
            assert np.array_equal(b.cat, main.cat[rows])

    def test_mismatched_row_counts_are_refused(self):
        main, aug = self._pair()
        with pytest.raises(ValueError):
            BlockLoader(main, main.indices(TRAIN), 32, cont_mean=np.zeros(E.N_CONT),
                        cont_std=np.ones(E.N_CONT), rating_mean=0.0, rating_sd=1.0,
                        augment=_tiny_data(n=200), augment_prob=0.5)
