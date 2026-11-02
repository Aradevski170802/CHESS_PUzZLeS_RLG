"""
PuzzleNet: a multi-task multilayer perceptron written from first principles in NumPy.

Why NumPy and not a framework
─────────────────────────────
The deployed model is a few matrix products, so the Flask app needs no deep-learning
dependency, and inference takes about a millisecond on a CPU. Writing the backward
pass by hand also keeps every step of the learning algorithm visible and testable:
tests/test_neural_network.py checks each gradient against finite differences.

Architecture
────────────
    x (3329) ─► [Linear ─► ReLU] x L  (shared trunk, default 512 ─► 256)
                    ├─► category head   Linear ─► softmax over 24 categories
                    ├─► theme head      Linear ─► 66 independent sigmoids
                    └─► rating head     Linear ─► (μ, s):  b ~ N(μ, exp(s))

Loss (per example, averaged over the batch)
───────────────────────────────────────────
    L = CE(softmax(z_c), y_c)
      + λ_t · Σ_j BCE(σ(z_t,j), y_t,j)
      + λ_r · sg(v^β) · ½ [ log v + (y_r − μ)² / v ],    v = exp(s) + ρ²

  * ρ is the puzzle's Lichess rating deviation, in the same standardised units as
    the rating. The observed rating is a noisy measurement of the puzzle's true
    difficulty, so the likelihood adds ρ² to the model's variance. exp(s) is then
    left to capture only the uncertainty that the content itself leaves: how hard
    the puzzle is given what is on the board. That is the quantity the IRT
    recommender needs for a mined puzzle, which has no Lichess rating at all.
  * sg(v^β) is β-NLL (Seitzer et al., 2022): the Gaussian NLL scaled by the
    stop-gradient of v^β. Plain NLL (β = 0) lets the network explain away hard
    examples by inflating their variance, which starves the mean of gradient.
    β = 1 gives the mean exactly the MSE gradient. β = 0.5 is the paper's
    recommended compromise.

Optimisation: AdamW (Loshchilov & Hutter, 2019) with decoupled weight decay on the
weight matrices only, global-norm gradient clipping, and linear warm-up followed by
cosine decay. Calibration after training fits two scalars on the validation split: a
softmax temperature T (Guo et al., 2017) and a variance scale c (v = c·exp(s) + ρ²).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

HEADS = ("cat", "theme", "rating")
S_MIN, S_MAX = -9.0, 4.0            # clamp on the log-variance output


@dataclass
class NetConfig:
    n_in: int
    hidden: tuple[int, ...] = (512, 256)
    n_cat: int = 24
    n_theme: int = 66
    theme_weight: float = 0.1       # λ_t: weight of the summed theme BCE
    rating_weight: float = 1.0      # λ_r
    beta: float = 0.5               # β-NLL exponent
    heads: tuple[str, ...] = HEADS  # heads that contribute to the loss (ablations)

    def to_json(self) -> str:
        d = asdict(self)
        d["hidden"], d["heads"] = list(self.hidden), list(self.heads)
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> "NetConfig":
        d = json.loads(s)
        d["hidden"], d["heads"] = tuple(d["hidden"]), tuple(d["heads"])
        return cls(**d)


@dataclass
class Batch:
    x: np.ndarray                     # (B, n_in) float32, already standardised
    cat: Optional[np.ndarray] = None  # (B,) int
    theme: Optional[np.ndarray] = None  # (B, n_theme) float32 0/1
    rating: Optional[np.ndarray] = None  # (B,) float32, standardised
    rho2: Optional[np.ndarray] = None    # (B,) float32, (RD / rating_sd)^2


def _softplus(z: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, z)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


class PuzzleNet:
    def __init__(self, config: NetConfig, seed: int = 0, dtype=np.float32):
        self.config = config
        self.dtype = dtype
        rng = np.random.default_rng(seed)
        self.params: dict[str, np.ndarray] = {}
        sizes = [config.n_in, *config.hidden]
        for i in range(len(config.hidden)):
            # He initialisation for ReLU layers.
            self.params[f"W{i}"] = (rng.standard_normal((sizes[i], sizes[i + 1]))
                                    * math.sqrt(2.0 / sizes[i])).astype(dtype)
            self.params[f"b{i}"] = np.zeros(sizes[i + 1], dtype=dtype)
        h = sizes[-1]
        for name, width in (("c", config.n_cat), ("t", config.n_theme), ("r", 2)):
            self.params[f"W{name}"] = (rng.standard_normal((h, width)) * math.sqrt(1.0 / h)).astype(dtype)
            self.params[f"b{name}"] = np.zeros(width, dtype=dtype)
        # Start the theme head at the log-odds of a rare theme, and the variance
        # head at log(0.5): a sane scale for a standardised rating.
        self.params["bt"][:] = -4.0
        self.params["br"][1] = math.log(0.5)
        # Filled in after training (see scripts/neural/train_puzzlenet.py).
        self.temperature = 1.0
        self.var_scale = 1.0
        self.meta: dict = {}

    # ── Forward ────────────────────────────────────────────────────────────
    def _trunk(self, x: np.ndarray) -> list[np.ndarray]:
        acts = [x]
        h = x
        for i in range(len(self.config.hidden)):
            h = h @ self.params[f"W{i}"]
            h += self.params[f"b{i}"]
            np.maximum(h, 0.0, out=h)
            acts.append(h)
        return acts

    def forward(self, x: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        acts = self._trunk(x)
        h = acts[-1]
        zc = h @ self.params["Wc"] + self.params["bc"]
        zt = h @ self.params["Wt"] + self.params["bt"]
        zr = h @ self.params["Wr"] + self.params["br"]
        return acts, zc, zt, zr

    # ── Loss and gradients ─────────────────────────────────────────────────
    def loss_and_grads(self, batch: Batch) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        cfg = self.config
        acts, zc, zt, zr = self.forward(batch.x)
        n = batch.x.shape[0]
        losses: dict[str, float] = {}
        dzc = np.zeros_like(zc)
        dzt = np.zeros_like(zt)
        dzr = np.zeros_like(zr)

        if "cat" in cfg.heads:
            logp = log_softmax(zc)
            losses["cat"] = float(-logp[np.arange(n), batch.cat].mean())
            dzc = np.exp(logp)
            dzc[np.arange(n), batch.cat] -= 1.0
            dzc /= n

        if "theme" in cfg.heads:
            y = batch.theme
            bce = _softplus(zt) - zt * y
            losses["theme"] = float(cfg.theme_weight * bce.sum(axis=1).mean())
            dzt = (cfg.theme_weight / n) * (_sigmoid(zt) - y)

        if "rating" in cfg.heads:
            mu, s_raw = zr[:, 0], zr[:, 1]
            s = np.clip(s_raw, S_MIN, S_MAX)
            var_m = np.exp(s)
            v = var_m + batch.rho2
            r = batch.rating - mu
            nll = 0.5 * (np.log(v) + r * r / v)
            w = v ** cfg.beta                     # stop-gradient weight (β-NLL)
            losses["rating"] = float(cfg.rating_weight * (w * nll).mean())
            scale = cfg.rating_weight * w / n
            dzr[:, 0] = scale * (-r / v)
            ds = scale * 0.5 * (1.0 / v - r * r / (v * v)) * var_m
            ds[(s_raw < S_MIN) | (s_raw > S_MAX)] = 0.0
            dzr[:, 1] = ds

        losses["total"] = float(sum(losses.values()))
        grads: dict[str, np.ndarray] = {}
        h = acts[-1]
        dh = np.zeros_like(h)
        for name, dz in (("c", dzc), ("t", dzt), ("r", dzr)):
            grads[f"W{name}"] = h.T @ dz
            grads[f"b{name}"] = dz.sum(axis=0)
            dh += dz @ self.params[f"W{name}"].T
        for i in reversed(range(len(cfg.hidden))):
            dz = dh * (acts[i + 1] > 0)
            grads[f"W{i}"] = acts[i].T @ dz
            grads[f"b{i}"] = dz.sum(axis=0)
            if i > 0:
                dh = dz @ self.params[f"W{i}"].T
        if not cfg.hidden:          # zero-hidden-layer (linear) model: heads read x
            pass
        return losses, grads

    # ── Prediction ─────────────────────────────────────────────────────────
    def predict(self, x: np.ndarray, batch_size: int = 8192) -> dict[str, np.ndarray]:
        """Calibrated outputs. rating_* are in standardised units; rating_sd is the
        content uncertainty only (no Lichess RD, since new puzzles have none)."""
        probs, themes, mus, sds = [], [], [], []
        for i in range(0, len(x), batch_size):
            _, zc, zt, zr = self.forward(x[i:i + batch_size])
            probs.append(np.exp(log_softmax(zc / self.temperature)))
            themes.append(_sigmoid(zt))
            mus.append(zr[:, 0])
            sds.append(np.sqrt(self.var_scale * np.exp(np.clip(zr[:, 1], S_MIN, S_MAX))))
        return {"cat_probs": np.concatenate(probs), "theme_probs": np.concatenate(themes),
                "rating_mean": np.concatenate(mus), "rating_sd": np.concatenate(sds)}

    def logits(self, x: np.ndarray, batch_size: int = 8192) -> tuple[np.ndarray, np.ndarray]:
        zcs, zrs = [], []
        for i in range(0, len(x), batch_size):
            _, zc, _, zr = self.forward(x[i:i + batch_size])
            zcs.append(zc)
            zrs.append(zr)
        return np.concatenate(zcs), np.concatenate(zrs)

    # ── Persistence ────────────────────────────────────────────────────────
    def save(self, path: Path, *, half: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        store = {f"p_{k}": (v.astype(np.float16) if half else v) for k, v in self.params.items()}
        np.savez(path, **store, config=np.array(self.config.to_json()),
                 temperature=np.array(self.temperature), var_scale=np.array(self.var_scale),
                 meta=np.array(json.dumps(self.meta)))

    @classmethod
    def load(cls, path: Path) -> "PuzzleNet":
        with np.load(Path(path), allow_pickle=False) as z:
            net = cls(NetConfig.from_json(str(z["config"])))
            for k in net.params:
                net.params[k] = z[f"p_{k}"].astype(np.float32)
            net.temperature = float(z["temperature"])
            net.var_scale = float(z["var_scale"])
            net.meta = json.loads(str(z["meta"]))
        return net

    def n_parameters(self) -> int:
        return int(sum(v.size for v in self.params.values()))


@dataclass
class AdamW:
    """Adam with decoupled weight decay, applied to weight matrices only."""
    params: dict[str, np.ndarray]
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1e-5
    clip_norm: Optional[float] = 5.0
    t: int = 0
    m: dict[str, np.ndarray] = field(default_factory=dict)
    v: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        for k, p in self.params.items():
            self.m[k] = np.zeros_like(p)
            self.v[k] = np.zeros_like(p)

    def step(self, grads: dict[str, np.ndarray], lr: float) -> float:
        norm = math.sqrt(sum(float(np.vdot(g, g)) for g in grads.values()))
        coef = 1.0
        if self.clip_norm and norm > self.clip_norm:
            coef = self.clip_norm / (norm + 1e-12)
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        step = lr / bc1
        for k, p in self.params.items():
            g = grads[k] if coef == 1.0 else grads[k] * coef
            m, v = self.m[k], self.v[k]
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * (g * g)
            if k.startswith("W") and self.weight_decay:
                p *= 1.0 - lr * self.weight_decay
            denom = np.sqrt(v / bc2)
            denom += self.eps
            p -= step * (m / denom)
        return norm


def lr_schedule(step: int, total: int, peak: float, warmup: int, floor: float = 0.02) -> float:
    """Linear warm-up to `peak`, then cosine decay to `floor * peak`."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress)))
