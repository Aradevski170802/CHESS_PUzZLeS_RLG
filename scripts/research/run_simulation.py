"""
Simulation study: does adaptive recommendation beat non-adaptive baselines,
and which adaptive policy is best?

Experiments
────────────
E1  Pre-registered replication. The protocol fixed in EVALUATION_METHODOLOGY
    before any result existed: 3 categories solved 30 % of the time, 20 at
    80 %, 200 runs × 150 attempts, uniform priors. Criteria:
      C1 weak-category targeting ≥ 80 % by trial 150 (trials 131–150)
      C2 posterior MAE ≤ 0.10 within 100 trials
      C3 Welch's t-test vs random, p < 0.01, Cohen's d > 0.8
E2  Main comparison on semi-synthetic players (real Lichess difficulty
    distributions, noisy Elo seed, game evidence mislabelled with the NEW
    tagger's measured confusion), under three learning-model assumptions.
E3  Label-quality ablation: perfect vs new-tagger vs old-tagger labels vs no
    game evidence — how much does the labeller matter downstream?
E4  Non-stationarity: at t = 75 the player fixes their worst weakness
    elsewhere. Which policies stop drilling it?
E5  Sensitivity: Beta prior strength; IRT target solve probability.

Design: common random numbers — for a given seed every policy meets the same
player and the same game evidence — so all policy comparisons are PAIRED
(paired t-test, Wilcoxon signed-rank, Cohen's d_z, bootstrap CI of the mean
difference), with Holm correction across the policies compared per metric.

Usage: python -m scripts.research.run_simulation [--runs 200] [--workers 14]
"""
from __future__ import annotations

import os

# One BLAS thread per worker process: 14 processes × 16 BLAS threads each
# oversubscribed the CPU and crashed the pool on Windows. Must be set before
# numpy is imported (children inherit the environment).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

from src.data.puzzle_loader import WEAKNESS_CATEGORIES, resolve_primary_category
from src.evaluation import simulation as sim

OUT = Path("eval/research")
POOL_CACHE = OUT / "sim_pool_ratings.npz"
CONFUSION_JSON = OUT / "labeller_validation_heldout.json"

POLICIES = {
    "Random":                 ("RandomPolicy", {}),
    "Round-robin":            ("RoundRobinPolicy", {}),
    "Static profile (top-3)": ("StaticProfilePolicy", {}),
    "Greedy (Beta mean)":     ("GreedyBetaPolicy", {}),
    "Beta-TS":                ("BetaTSPolicy", {}),
    "Beta-TS (γ=0.97)":       ("BetaTSPolicy", {"discount": 0.97}),
    "IRT-TS (Elo band)":      ("IRTTSPolicy", {"adaptive": False}),
    "IRT-TS":                 ("IRTTSPolicy", {}),
    "Oracle":                 ("OraclePolicy", {}),
}

# ─────────────────────────────────────────────────────────────────────────────
# Worker setup (each process loads the pool and confusion matrices once)
# ─────────────────────────────────────────────────────────────────────────────

_POOL = None
_CONF = None


def _init_worker(pool_path: str, conf_path: str) -> None:
    global _POOL, _CONF
    data = np.load(pool_path)
    _POOL = sim.PuzzlePool({c: data[c] for c in WEAKNESS_CATEGORIES if c in data.files})
    conf = json.loads(Path(conf_path).read_text("utf-8"))["confusion"]
    _CONF = {"labels": conf["labels"], "new": np.array(conf["new"]), "old": np.array(conf["old"])}


def _make(spec):
    cls_name, kwargs = spec
    return lambda: getattr(sim, cls_name)(**kwargs)


def _task(args):
    kind, spec, seeds, cfg = args
    out = []
    for seed in seeds:
        if kind == "fixed":
            r = sim.fixed_rate_episode(_make(spec), seed, T=cfg["T"])
        else:
            pop = sim.PopulationConfig(**cfg.get("pop", {}))
            if cfg.get("weights") is not None:
                pop.category_weights = np.array(cfg["weights"])
            learn = sim.LearnerConfig(**cfg.get("learn", {}))
            labels = cfg["labels"]
            conf = None if labels == "perfect" else _CONF[labels]
            r = sim.run_episode(_make(spec), seed, _POOL, pop, learn, conf,
                                _CONF["labels"], T=cfg["T"])
        out.append(r)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def mean_ci(x) -> dict:
    x = np.asarray(x, dtype=float)
    m = float(x.mean())
    h = float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0
    return {"mean": round(m, 4), "ci95": [round(m - h, 4), round(m + h, 4)]}


def paired(a, b, rng) -> dict:
    """a − b, paired by seed."""
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1)
    boot = rng.choice(d, size=(4000, len(d)), replace=True).mean(axis=1)
    try:
        w = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        w = 1.0
    return {
        "mean_diff": round(float(d.mean()), 4),
        "ci95": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
        "p_paired_t": float(stats.ttest_rel(a, b).pvalue) if sd > 0 else 1.0,
        "p_wilcoxon": w,
        "d_z": round(float(d.mean() / sd), 3) if sd > 0 else 0.0,
    }


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def summarise_runs(results: list, T: int) -> dict:
    """Per-run scalar summaries (one value per seed, so they pair up)."""
    wh = np.array([r.weak_hit for r in results])
    reg = np.array([r.regret for r in results])
    ps = np.array([r.p_served for r in results])
    sp = np.array([r.spearman for r in results])
    t3 = np.array([r.top3_recall for r in results])
    sw = np.array([r.switched for r in results])
    last = slice(max(0, T - 50), T)
    return {
        "targeting_first30": wh[:, :30].mean(axis=1),
        "targeting_last50": wh[:, last].mean(axis=1),
        "cum_regret": reg.sum(axis=1),
        "mean_p_served": ps.mean(axis=1),
        "frac_frustrating": (ps < 0.30).mean(axis=1),
        "frac_in_zpd": ((ps >= 0.50) & (ps <= 0.85)).mean(axis=1),
        "spearman_final": sp[:, -1],
        "top3_recall_final": t3[:, -1],
        "switch_rate": sw.mean(axis=1),
        "gain_weak": np.array([r.gain_weak for r in results]),
        "gain_all": np.array([r.gain_all for r in results]),
        "_curve_targeting": wh.mean(axis=0),
        "_curve_targeting_sd": wh.std(axis=0, ddof=1),
        "_curve_regret": reg.mean(axis=0),
        "_curve_p": ps.mean(axis=0),
        "_curve_top3": t3.mean(axis=0),
        "_n": len(results),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def build_pool_cache(per_cat: int = 8000, seed: int = 3) -> dict:
    import pyarrow.parquet as pq
    df = pq.read_table("data/processed/puzzles_full.parquet", columns=["Rating", "Themes"]).to_pandas()
    df["cat"] = df["Themes"].map(resolve_primary_category)
    rng = np.random.default_rng(seed)
    arrays, counts = {}, {}
    for c in WEAKNESS_CATEGORIES:
        r = df.loc[df["cat"] == c, "Rating"].to_numpy()
        counts[c] = int(len(r))
        arrays[c] = rng.choice(r, size=min(per_cat, len(r)), replace=False) if len(r) else np.array([1500.0])
    np.savez_compressed(POOL_CACHE, **arrays)
    (OUT / "sim_category_counts.json").write_text(json.dumps(counts, indent=1), encoding="utf-8")
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Experiments
# ─────────────────────────────────────────────────────────────────────────────

def run(ex: ProcessPoolExecutor, kind: str, names: list[str], cfg: dict, runs: int,
        chunk: int, specs: dict | None = None, keep_raw: bool = False):
    specs = specs or POLICIES
    seeds = list(range(1000, 1000 + runs))
    chunks = [seeds[i:i + chunk] for i in range(0, runs, chunk)]
    jobs = {n: [ex.submit(_task, (kind, specs[n], c, cfg)) for c in chunks] for n in names}
    raw = {n: [r for f in fs for r in f.result()] for n, fs in jobs.items()}
    summ = {n: summarise_runs(eps, cfg["T"]) for n, eps in raw.items()}
    return (summ, raw) if keep_raw else summ


def compare(summ: dict, metric: str, baseline: str, others: list[str], rng) -> dict:
    tests = {o: paired(summ[o][metric], summ[baseline][metric], rng) for o in others}
    adj = holm({o: t["p_paired_t"] for o, t in tests.items()})
    for o in tests:
        tests[o]["p_holm"] = adj[o]
    return tests


def table(summ: dict, metrics: list[str]) -> dict:
    return {n: {m: mean_ci(s[m]) for m in metrics} for n, s in summ.items()}


def experiment_e6(ex, args, weights, rng) -> dict:
    """
    E6: does a family-structured prior help? Tested where weaknesses really do
    cluster within tactic families AND where they do not — a structured prior
    should help in the first case and cost little in the second.
    """
    specs = {
        "Beta-TS": ("BetaTSPolicy", {}),
        "IRT-TS": ("IRTTSPolicy", {}),
        "IRT-TS (family prior)": ("IRTTSPolicy", {"family_prior": True}),
    }
    out = {}
    for label, pop in (("no family structure", {}),
                       ("weaknesses cluster in one family", {"family_sd": 0.4, "weak_family": True})):
        cfg = {"T": 150, "labels": "new", "weights": weights, "learn": {"model": "none"}, "pop": pop}
        summ = run(ex, "episode", list(specs), cfg, args.runs, args.chunk, specs=specs)
        out[label] = {
            "table": table(summ, ["targeting_first30", "targeting_last50", "cum_regret",
                                  "top3_recall_final"]),
            "family_vs_plain": {m: paired(summ["IRT-TS (family prior)"][m], summ["IRT-TS"][m], rng)
                                for m in ("targeting_first30", "targeting_last50", "cum_regret")},
        }
    return out


def experiment_e7(ex, args, rng) -> dict:
    """
    E7: robustness to realistic effect sizes. The real-player cohort found
    per-category skill differences far smaller than E2 assumed (split-half
    reliability near zero). Re-run the key comparison with the cohort's own
    category frequencies and hit rate, under three worlds.
    """
    ce = json.loads((OUT / "cohort_evaluation.json").read_text("utf-8"))
    sc = ce["simulator_calibration"]
    weights = [max(1e-4, sc["category_frequency"].get(c, 0.0)) for c in WEAKNESS_CATEGORIES]
    base = {"game_base_hit": sc["overall_hit_rate"],
            "opportunities_mean": sc["opportunities_per_player_train"]}
    worlds = {
        "original assumption (σ 0.35, 3 weak at −1.2)": {},
        "subtle weaknesses (σ 0.15, 3 weak at −0.5)": {"delta_sd": 0.15, "weak_delta_mean": -0.5,
                                                       "weak_delta_sd": 0.1},
        "no category weaknesses (σ 0)": {"delta_sd": 0.0, "n_weak": 0},
    }
    specs = {n: POLICIES[n] for n in ("Random", "Beta-TS", "IRT-TS", "Oracle")}
    out = {"calibration_source": "eval/research/cohort_evaluation.json", "worlds": {}}
    for label, pop in worlds.items():
        cfg = {"T": 150, "labels": "new", "weights": weights, "learn": {"model": "none"},
               "pop": {**base, **pop}}
        summ = run(ex, "episode", list(specs), cfg, args.runs, args.chunk, specs=specs)
        out["worlds"][label] = {
            "table": table(summ, ["targeting_last50", "cum_regret", "frac_frustrating",
                                  "frac_in_zpd", "top3_recall_final"]),
            "IRT_vs_Beta": {m: paired(summ["IRT-TS"][m], summ["Beta-TS"][m], rng)
                            for m in ("targeting_last50", "cum_regret", "frac_frustrating")},
            "Beta_vs_Random": {m: paired(summ["Beta-TS"][m], summ["Random"][m], rng)
                               for m in ("targeting_last50", "cum_regret")},
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--only", default="", help="run only these experiments, e.g. E6 "
                    "(results are merged into the existing JSON)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    t0 = time.time()

    counts = (json.loads((OUT / "sim_category_counts.json").read_text("utf-8"))
              if POOL_CACHE.exists() and (OUT / "sim_category_counts.json").exists()
              else build_pool_cache())
    weights = [max(1, counts.get(c, 1)) for c in WEAKNESS_CATEGORIES]
    results: dict = {"runs": args.runs, "design": "paired by seed (common random numbers)"}
    curves: dict = {}

    if args.only:
        only = set(args.only.split(","))
        results = json.loads((OUT / "simulation_results.json").read_text("utf-8"))
        curves = json.loads((OUT / "simulation_curves.json").read_text("utf-8"))
        with ProcessPoolExecutor(args.workers, initializer=_init_worker,
                                 initargs=(str(POOL_CACHE), str(CONFUSION_JSON))) as ex:
            if "E6" in only:
                results["E6_family"] = experiment_e6(ex, args, weights, rng)
            if "E7" in only:
                results["E7_realistic"] = experiment_e7(ex, args, rng)
        (OUT / "simulation_results.json").write_text(json.dumps(results, indent=1, default=float),
                                                     encoding="utf-8")
        print(f"merged {sorted(only)} -> simulation_results.json ({time.time() - t0:.0f}s)")
        return

    with ProcessPoolExecutor(args.workers, initializer=_init_worker,
                             initargs=(str(POOL_CACHE), str(CONFUSION_JSON))) as ex:
        # ── E1: pre-registered replication ─────────────────────────────────
        e1_names = ["Random", "Round-robin", "Greedy (Beta mean)", "Beta-TS"]
        e1, e1_raw = run(ex, "fixed", e1_names, {"T": 150}, args.runs, args.chunk, keep_raw=True)
        mae = {n: np.array([e.mae for e in e1_raw[n]]) for n in ("Beta-TS", "Random")}
        # The pre-registered targeting window is trials 131-150.
        last20 = {n: np.array([e.weak_hit[130:150].mean() for e in e1_raw[n]]) for n in e1_names}
        bt, rd = last20["Beta-TS"], last20["Random"]
        welch = stats.ttest_ind(bt, rd, equal_var=False)
        pooled = math.sqrt((bt.var(ddof=1) + rd.var(ddof=1)) / 2)
        d_welch = float((bt.mean() - rd.mean()) / pooled) if pooled > 0 else float("inf")
        last20 = last20["Beta-TS"]
        results["E1_preregistered"] = {
            "table": table(e1, ["targeting_first30", "targeting_last50", "cum_regret", "switch_rate"]),
            "C1_targeting_trials_131_150": {**mean_ci(last20), "threshold": 0.80,
                                            "pass": bool(last20.mean() >= 0.80)},
            "C2_mae_at_trial_100": {**mean_ci(mae["Beta-TS"][:, 99]), "threshold": 0.10,
                                    "pass": bool(mae["Beta-TS"][:, 99].mean() <= 0.10),
                                    "random_policy_mae_at_100": mean_ci(mae["Random"][:, 99])},
            "C3_vs_random": {"welch_t": float(welch.statistic), "p": float(welch.pvalue),
                             "cohens_d": round(d_welch, 3),
                             "pass": bool(welch.pvalue < 0.01 and d_welch > 0.8)},
        }
        curves["E1"] = {n: s["_curve_targeting"].tolist() for n, s in e1.items()}
        curves["E1_mae"] = {n: mae[n].mean(axis=0).tolist() for n in mae}
        print(f"E1 done  {time.time() - t0:.0f}s")

        # ── E2: main comparison under three learning models ────────────────
        e2_names = list(POLICIES)
        metrics = ["targeting_first30", "targeting_last50", "cum_regret", "mean_p_served",
                   "frac_frustrating", "frac_in_zpd", "spearman_final", "top3_recall_final",
                   "switch_rate", "gain_weak", "gain_all"]
        results["E2_main"] = {}
        for model in ("none", "zpd", "error"):
            cfg = {"T": 150, "labels": "new", "weights": weights, "learn": {"model": model}}
            e2 = run(ex, "episode", e2_names, cfg, args.runs, args.chunk)
            others = [n for n in e2_names if n not in ("Beta-TS", "Oracle")]
            block = {"table": table(e2, metrics),
                     "vs_BetaTS": {m: compare(e2, m, "Beta-TS", others, rng)
                                   for m in ("targeting_last50", "cum_regret", "top3_recall_final",
                                             "frac_frustrating", "gain_weak")},
                     "vs_Random": {m: compare(e2, m, "Random",
                                              [n for n in e2_names if n not in ("Random", "Oracle")], rng)
                                   for m in ("targeting_last50", "cum_regret", "gain_weak")}}
            results["E2_main"][model] = block
            curves[f"E2_{model}"] = {n: {"targeting": s["_curve_targeting"].tolist(),
                                         "regret": s["_curve_regret"].tolist(),
                                         "p": s["_curve_p"].tolist(),
                                         "top3": s["_curve_top3"].tolist()} for n, s in e2.items()}
            print(f"E2[{model}] done  {time.time() - t0:.0f}s")

        # ── E3: label-quality ablation ─────────────────────────────────────
        results["E3_labels"] = {}
        curves["E3"] = {}
        e3_all = {}
        for lab in ("perfect", "new", "old", "none"):
            cfg = {"T": 60, "labels": "new" if lab == "none" else lab, "weights": weights,
                   "learn": {"model": "none"},
                   "pop": {"opportunities_mean": 0.0} if lab == "none" else {}}
            e3 = run(ex, "episode", ["Beta-TS", "IRT-TS"], cfg, args.runs, args.chunk)
            e3_all[lab] = e3
            results["E3_labels"][lab] = table(e3, ["targeting_first30", "top3_recall_final",
                                                   "cum_regret"])
            curves["E3"][lab] = {n: s["_curve_top3"].tolist() for n, s in e3.items()}
        # Same seeds across label conditions, so label effects are paired too.
        results["E3_paired"] = {
            pol: {f"{a}_vs_{b}": paired(e3_all[a][pol]["targeting_first30"],
                                        e3_all[b][pol]["targeting_first30"], rng)
                  for a, b in (("new", "old"), ("perfect", "new"), ("new", "none"))}
            for pol in ("Beta-TS", "IRT-TS")
        }
        print(f"E3 done  {time.time() - t0:.0f}s")

        # ── E4: non-stationarity (change point) ────────────────────────────
        e4_specs = {
            "Beta-TS": ("BetaTSPolicy", {}),
            "Beta-TS (γ=0.97)": ("BetaTSPolicy", {"discount": 0.97}),
            "Beta-TS (γ=0.93)": ("BetaTSPolicy", {"discount": 0.93}),
            "IRT-TS": ("IRTTSPolicy", {}),
            "Oracle": ("OraclePolicy", {}),
        }
        cfg = {"T": 200, "labels": "new", "weights": weights,
               "learn": {"model": "changepoint", "changepoint_t": 75, "changepoint_gain": 1.5}}
        e4, e4_raw = run(ex, "episode", list(e4_specs), cfg, args.runs, args.chunk,
                         specs=e4_specs, keep_raw=True)
        post = {n: np.array([e.regret[75:].sum() for e in eps]) for n, eps in e4_raw.items()}
        results["E4_changepoint"] = {
            "post_change_regret": {n: mean_ci(v) for n, v in post.items()},
            "vs_BetaTS": {n: paired(post[n], post["Beta-TS"], rng) for n in post if n != "Beta-TS"},
        }
        curves["E4"] = {n: s["_curve_regret"].tolist() for n, s in e4.items()}
        print(f"E4 done  {time.time() - t0:.0f}s")

        # ── E5: sensitivity ────────────────────────────────────────────────
        e5_specs = {f"Beta-TS (prior×{s:g})": ("BetaTSPolicy", {"prior_scale": s})
                    for s in (0.2, 0.5, 1.0, 2.0)}
        e5_specs.update({f"IRT-TS (p*={p:g})": ("IRTTSPolicy", {"p_target": p})
                         for p in (0.5, 0.65, 0.8)})
        results["E5_sensitivity"] = {}
        for model in ("none", "zpd"):
            cfg = {"T": 150, "labels": "new", "weights": weights, "learn": {"model": model}}
            e5 = run(ex, "episode", list(e5_specs), cfg, args.runs, args.chunk, specs=e5_specs)
            results["E5_sensitivity"][model] = table(
                e5, ["targeting_last50", "cum_regret", "mean_p_served", "gain_weak"])
        print(f"E5 done  {time.time() - t0:.0f}s")

    results["elapsed_s"] = round(time.time() - t0, 1)
    (OUT / "simulation_results.json").write_text(json.dumps(results, indent=1, default=float),
                                                 encoding="utf-8")
    (OUT / "simulation_curves.json").write_text(json.dumps(curves), encoding="utf-8")
    print(f"saved -> {OUT / 'simulation_results.json'}  ({results['elapsed_s']}s)")


if __name__ == "__main__":
    main()
