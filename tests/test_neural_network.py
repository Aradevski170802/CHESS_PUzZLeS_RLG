"""PuzzleNet network: finite-difference gradient checks, optimiser, persistence."""
import math

import numpy as np
import pytest

from src.neural.network import (AdamW, Batch, NetConfig, PuzzleNet, log_softmax,
                                lr_schedule)


def _batch(rng, n=7, n_in=11, n_cat=5, n_theme=4):
    return Batch(
        x=rng.standard_normal((n, n_in)),
        cat=rng.integers(0, n_cat, n),
        theme=(rng.random((n, n_theme)) < 0.3).astype(float),
        rating=rng.standard_normal(n),
        rho2=rng.uniform(0.01, 0.2, n),
    )


def _reference_loss(net: PuzzleNet, b: Batch, w_rating: np.ndarray) -> float:
    """Independent re-implementation of the loss with the β-NLL weights frozen
    (they are a stop-gradient in the analytic backward pass)."""
    cfg = net.config
    h = b.x
    for i in range(len(cfg.hidden)):
        h = np.maximum(h @ net.params[f"W{i}"] + net.params[f"b{i}"], 0.0)
    zc = h @ net.params["Wc"] + net.params["bc"]
    zt = h @ net.params["Wt"] + net.params["bt"]
    zr = h @ net.params["Wr"] + net.params["br"]
    total = 0.0
    if "cat" in cfg.heads:
        total += -log_softmax(zc)[np.arange(len(b.cat)), b.cat].mean()
    if "theme" in cfg.heads:
        total += cfg.theme_weight * (np.logaddexp(0, zt) - zt * b.theme).sum(axis=1).mean()
    if "rating" in cfg.heads:
        v = np.exp(zr[:, 1]) + b.rho2
        nll = 0.5 * (np.log(v) + (b.rating - zr[:, 0]) ** 2 / v)
        total += cfg.rating_weight * (w_rating * nll).mean()
    return float(total)


def _frozen_weights(net: PuzzleNet, b: Batch) -> np.ndarray:
    h = b.x
    for i in range(len(net.config.hidden)):
        h = np.maximum(h @ net.params[f"W{i}"] + net.params[f"b{i}"], 0.0)
    zr = h @ net.params["Wr"] + net.params["br"]
    return (np.exp(zr[:, 1]) + b.rho2) ** net.config.beta


@pytest.mark.parametrize("hidden,heads,beta", [
    ((9, 6), ("cat", "theme", "rating"), 0.5),
    ((9, 6), ("cat", "theme", "rating"), 0.0),
    ((8,), ("cat", "theme", "rating"), 1.0),
    ((), ("cat", "theme", "rating"), 0.5),      # the linear baseline
    ((9, 6), ("cat",), 0.5),                     # single-task ablations
    ((9, 6), ("rating",), 0.5),
])
def test_gradients_match_finite_differences(hidden, heads, beta):
    rng = np.random.default_rng(3)
    cfg = NetConfig(n_in=11, hidden=hidden, n_cat=5, n_theme=4, theme_weight=0.3,
                    rating_weight=0.8, beta=beta, heads=heads)
    net = PuzzleNet(cfg, seed=1, dtype=np.float64)
    for k in net.params:          # move biases off zero so ReLU kinks are unlikely
        net.params[k] += 0.05 * rng.standard_normal(net.params[k].shape)
    b = _batch(rng)
    w = _frozen_weights(net, b)
    losses, grads = net.loss_and_grads(b)
    assert losses["total"] == pytest.approx(_reference_loss(net, b, w), rel=1e-10)

    eps = 1e-6
    for name, p in net.params.items():
        flat = p.reshape(-1)
        for j in rng.choice(flat.size, size=min(12, flat.size), replace=False):
            old = flat[j]
            flat[j] = old + eps
            up = _reference_loss(net, b, w)
            flat[j] = old - eps
            down = _reference_loss(net, b, w)
            flat[j] = old
            numeric = (up - down) / (2 * eps)
            analytic = grads[name].reshape(-1)[j]
            assert analytic == pytest.approx(numeric, rel=1e-5, abs=1e-8), (name, j)


def test_heads_left_out_of_the_loss_get_no_gradient():
    rng = np.random.default_rng(0)
    net = PuzzleNet(NetConfig(n_in=11, hidden=(6,), n_cat=5, n_theme=4, heads=("cat",)),
                    dtype=np.float64)
    _, grads = net.loss_and_grads(_batch(rng))
    assert not grads["Wt"].any() and not grads["Wr"].any()
    assert grads["Wc"].any()


def test_known_rating_noise_lowers_the_weight_of_noisy_labels():
    """A puzzle with a large Lichess RD pulls the mean less than a precise one."""
    net = PuzzleNet(NetConfig(n_in=3, hidden=(), n_cat=2, n_theme=1, heads=("rating",),
                              beta=0.0), dtype=np.float64)
    x = np.ones((1, 3))

    def mean_grad(rho2):
        b = Batch(x=x, cat=np.zeros(1, int), theme=np.zeros((1, 1)),
                  rating=np.array([2.0]), rho2=np.array([rho2]))
        return abs(net.loss_and_grads(b)[1]["br"][0])

    assert mean_grad(4.0) < mean_grad(0.01)


def test_adamw_minimises_a_quadratic():
    target = np.array([3.0, -2.0])
    params = {"W": np.zeros(2)}
    opt = AdamW(params, weight_decay=0.0, clip_norm=None)
    for _ in range(2000):
        opt.step({"W": 2 * (params["W"] - target)}, lr=0.05)
    assert np.allclose(params["W"], target, atol=1e-3)


def test_weight_decay_applies_to_weights_not_biases():
    params = {"W": np.ones(3), "b": np.ones(3)}
    opt = AdamW(params, weight_decay=0.1, clip_norm=None)
    opt.step({"W": np.zeros(3), "b": np.zeros(3)}, lr=0.1)
    assert np.all(params["W"] < 1.0) and np.all(params["b"] == 1.0)


def test_lr_schedule_warms_up_then_decays():
    lrs = [lr_schedule(s, total=1000, peak=1e-3, warmup=100) for s in range(1000)]
    assert lrs[0] < lrs[50] < lrs[99] == pytest.approx(1e-3)
    assert lrs[500] < lrs[100] and lrs[-1] == pytest.approx(2e-5, rel=0.05)


def test_save_and_load_round_trip(tmp_path):
    rng = np.random.default_rng(5)
    net = PuzzleNet(NetConfig(n_in=11, hidden=(8, 4), n_cat=5, n_theme=4), seed=2)
    net.temperature, net.var_scale = 1.3, 0.7
    net.meta = {"rating_mean": 1500.0, "note": "test"}
    x = rng.standard_normal((6, 11)).astype(np.float32)
    net.save(tmp_path / "m.npz")
    back = PuzzleNet.load(tmp_path / "m.npz")
    a, b = net.predict(x), back.predict(x)
    for k in a:
        assert np.allclose(a[k], b[k])
    assert back.meta == net.meta and back.temperature == 1.3 and back.var_scale == 0.7


def test_half_precision_export_is_close(tmp_path):
    rng = np.random.default_rng(6)
    net = PuzzleNet(NetConfig(n_in=11, hidden=(8,), n_cat=5, n_theme=4), seed=2)
    x = rng.standard_normal((20, 11)).astype(np.float32)
    net.save(tmp_path / "h.npz", half=True)
    back = PuzzleNet.load(tmp_path / "h.npz")
    assert np.abs(net.predict(x)["cat_probs"] - back.predict(x)["cat_probs"]).max() < 1e-2


def test_temperature_softens_probabilities():
    rng = np.random.default_rng(7)
    net = PuzzleNet(NetConfig(n_in=11, hidden=(8,), n_cat=5, n_theme=4), seed=3)
    x = rng.standard_normal((30, 11)).astype(np.float32) * 5
    sharp = net.predict(x)["cat_probs"].max(axis=1).mean()
    net.temperature = 3.0
    assert net.predict(x)["cat_probs"].max(axis=1).mean() < sharp


def test_probabilities_are_normalised():
    rng = np.random.default_rng(8)
    net = PuzzleNet(NetConfig(n_in=11, hidden=(8,), n_cat=5, n_theme=4), seed=4)
    out = net.predict(rng.standard_normal((10, 11)).astype(np.float32))
    assert np.allclose(out["cat_probs"].sum(axis=1), 1.0, atol=1e-5)
    assert np.all((out["theme_probs"] > 0) & (out["theme_probs"] < 1))
    assert np.all(out["rating_sd"] > 0)
    assert math.isfinite(float(out["rating_mean"].sum()))
