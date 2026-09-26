"""
Metrics and post-hoc calibration shared by the PuzzleNet training and evaluation
scripts. All ratings passed in here are in rating points unless a name says
otherwise.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import optimize, stats
from sklearn.metrics import average_precision_score, cohen_kappa_score, f1_score

from src.neural.network import log_softmax

_LOG_2PI = math.log(2 * math.pi)


def expected_calibration_error(y: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE (Guo et al., 2017): the confidence-weighted gap between
    confidence and accuracy over equal-width confidence bins."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def category_metrics(y: np.ndarray, probs: np.ndarray, n_classes: int) -> dict:
    pred = probs.argmax(axis=1)
    eps = 1e-12
    return {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, labels=np.arange(n_classes), average="macro",
                                   zero_division=0)),
        "cohens_kappa": float(cohen_kappa_score(y, pred)),
        "nll": float(-np.log(probs[np.arange(len(y)), y] + eps).mean()),
        "ece": expected_calibration_error(y, probs),
    }


def theme_metrics(y: np.ndarray, probs: np.ndarray, names: list[str], min_pos: int = 20) -> dict:
    eps = 1e-7
    p = np.clip(probs, eps, 1 - eps)
    bce = -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum(axis=1).mean()
    pred = probs >= 0.5
    tp = float((pred & (y > 0)).sum())
    micro_f1 = 2 * tp / max(1.0, float(pred.sum() + (y > 0).sum()))
    aps = {}
    for j, name in enumerate(names):
        if y[:, j].sum() >= min_pos:
            aps[name] = float(average_precision_score(y[:, j], probs[:, j]))
    return {"bce_sum": float(bce), "micro_f1": float(micro_f1),
            "macro_ap": float(np.mean(list(aps.values()))) if aps else None,
            "ap_per_theme": aps}


def rating_metrics(y: np.ndarray, mu: np.ndarray, sd_content: np.ndarray | None,
                   rd: np.ndarray | None) -> dict:
    """Point and probabilistic accuracy of predicted ratings against the observed
    Lichess ratings. The predictive distribution for an observed rating is
    N(mu, sd_content² + RD²): the model's content uncertainty plus the rating's own
    measurement noise."""
    err = mu - y
    out = {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "r2": float(1.0 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
        "pearson_r": float(np.corrcoef(mu, y)[0, 1]),
        "spearman_rho": float(stats.spearmanr(mu, y).statistic),
    }
    if sd_content is not None:
        var = sd_content ** 2 + (rd ** 2 if rd is not None else 0.0)
        out["nll"] = float(np.mean(0.5 * (_LOG_2PI + np.log(var) + err ** 2 / var)))
        z = np.abs(err) / np.sqrt(var)
        for level in (0.5, 0.8, 0.95):
            out[f"coverage_{int(level * 100)}"] = float(np.mean(z <= stats.norm.ppf(0.5 + level / 2)))
        out["mean_sd_content"] = float(np.mean(sd_content))
    return out


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature T minimising validation NLL of softmax(z / T)."""
    def nll(log_t: float) -> float:
        lp = log_softmax(logits / math.exp(log_t))
        return float(-lp[np.arange(len(y)), y].mean())
    res = optimize.minimize_scalar(nll, bounds=(math.log(0.25), math.log(8.0)), method="bounded")
    return float(math.exp(res.x))


def fit_variance_scale(y_std: np.ndarray, mu: np.ndarray, var_model: np.ndarray,
                       rho2: np.ndarray) -> float:
    """Scale c minimising validation Gaussian NLL with v = c·var_model + ρ²
    (standardised units)."""
    def nll(log_c: float) -> float:
        v = math.exp(log_c) * var_model + rho2
        return float(np.mean(0.5 * (np.log(v) + (y_std - mu) ** 2 / v)))
    res = optimize.minimize_scalar(nll, bounds=(math.log(0.05), math.log(20.0)), method="bounded")
    return float(math.exp(res.x))
