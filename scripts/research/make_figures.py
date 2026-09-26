"""
Figures for the simulation study (eval/research/figures/*.png).

Reads simulation_results.json and simulation_curves.json written by
run_simulation.py. Usage: python -m scripts.research.make_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("eval/research")
FIG = R / "figures"

STYLE = {
    "Random": ("#9aa0ad", "--"), "Round-robin": ("#b8bdc7", ":"),
    "Static profile (top-3)": ("#c49a6c", "-."), "Greedy (Beta mean)": ("#8d6e63", "-"),
    "Beta-TS": ("#3b6fb6", "-"), "Beta-TS (γ=0.97)": ("#7aa0d6", "--"),
    "Beta-TS (γ=0.93)": ("#a8c0e6", ":"), "IRT-TS (Elo band)": ("#e0a030", "--"),
    "IRT-TS": ("#c0392b", "-"), "Oracle": ("#2e7d32", ":"),
}


def smooth(y, w=10):
    y = np.asarray(y, float)
    return np.convolve(y, np.ones(w) / w, mode="valid")


def line(ax, name, y, w=10, **kw):
    c, ls = STYLE.get(name, ("#555", "-"))
    ys = smooth(y, w)
    ax.plot(np.arange(w, w + len(ys)), ys, color=c, ls=ls, lw=2 if name in ("IRT-TS", "Beta-TS") else 1.4,
            label=name, **kw)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    res = json.loads((R / "simulation_results.json").read_text("utf-8"))
    cur = json.loads((R / "simulation_curves.json").read_text("utf-8"))
    n = res["runs"]

    # E1 — pre-registered protocol
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for name, y in cur["E1"].items():
        line(axes[0], name, y)
    axes[0].axhline(0.8, color="k", lw=1, ls="--")
    axes[0].text(2, 0.82, "pre-registered target: 80 %", fontsize=8)
    axes[0].set(title="E1 — weak-category targeting (fixed solve rates)",
                xlabel="attempt", ylabel="P(chosen category is truly weak), 10-trial mean", ylim=(0, 1.05))
    for name, y in cur["E1_mae"].items():
        line(axes[1], name, y, w=1)
    axes[1].axhline(0.10, color="k", lw=1, ls="--")
    axes[1].axvline(100, color="k", lw=0.8, ls=":")
    axes[1].set(title="E1 — posterior MAE vs true solve rates", xlabel="attempt", ylabel="MAE")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    e1 = res["E1_preregistered"]
    fig.suptitle(f"Pre-registered replication (n = {n} runs): C1 {'PASS' if e1['C1_targeting_trials_131_150']['pass'] else 'FAIL'} · "
                 f"C2 {'PASS' if e1['C2_mae_at_trial_100']['pass'] else 'FAIL'} · "
                 f"C3 {'PASS' if e1['C3_vs_random']['pass'] else 'FAIL'}", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "e1_preregistered.png", dpi=140)
    plt.close(fig)

    # E2 — targeting and served difficulty, no learning
    c = cur["E2_none"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for name, d in c.items():
        line(axes[0], name, d["targeting"])
        line(axes[1], name, d["p"])
    axes[0].set(title="E2 — weak-category targeting", xlabel="attempt",
                ylabel="P(chosen category among true bottom-3)", ylim=(0, 1.05))
    axes[1].axhspan(0.5, 0.85, color="#2e7d32", alpha=0.08, label="productive zone 0.50–0.85")
    axes[1].axhline(0.30, color="#c0392b", lw=0.8, ls=":")
    axes[1].set(title="E2 — true solve probability of the served puzzle", xlabel="attempt",
                ylabel="P(solve)", ylim=(0, 1))
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Semi-synthetic players, no learning (n = {n} paired runs)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "e2_targeting_and_difficulty.png", dpi=140)
    plt.close(fig)

    # E2 — summary bars across learning models
    pols = ["Random", "Static profile (top-3)", "Greedy (Beta mean)", "Beta-TS", "Beta-TS (γ=0.97)",
            "IRT-TS (Elo band)", "IRT-TS", "Oracle"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    specs = [("frac_frustrating", "none", "Share of puzzles with P(solve) < 0.30"),
             ("cum_regret", "none", "Cumulative regret (logits), no learning"),
             ("gain_weak", None, "Learning gain in initially-weak categories")]
    for ax, (metric, model, title) in zip(axes, specs):
        if model:
            tb = res["E2_main"][model]["table"]
            vals = [tb[p][metric]["mean"] for p in pols]
            err = [[tb[p][metric]["mean"] - tb[p][metric]["ci95"][0] for p in pols],
                   [tb[p][metric]["ci95"][1] - tb[p][metric]["mean"] for p in pols]]
            ax.barh(pols, vals, xerr=err, color=[STYLE[p][0] for p in pols])
        else:
            width = 0.4
            y = np.arange(len(pols))
            for j, (m, col) in enumerate((("zpd", "#6a9fd8"), ("error", "#e39b6b"))):
                tb = res["E2_main"][m]["table"]
                vals = [tb[p][metric]["mean"] for p in pols]
                err = [[tb[p][metric]["mean"] - tb[p][metric]["ci95"][0] for p in pols],
                       [tb[p][metric]["ci95"][1] - tb[p][metric]["mean"] for p in pols]]
                ax.barh(y + (j - 0.5) * width, vals, height=width, xerr=err, color=col,
                        label=f"learning model: {m}")
            ax.set_yticks(y, pols)
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "e2_summary.png", dpi=140)
    plt.close(fig)

    # E3 — label quality
    fig, ax = plt.subplots(figsize=(7, 4.3))
    labs = ["none", "old", "new", "perfect"]
    names = {"none": "no game\nevidence", "old": "old labeller", "new": "new tagger", "perfect": "perfect\nlabels"}
    x = np.arange(len(labs))
    for j, (pol, col) in enumerate((("Beta-TS", "#3b6fb6"), ("IRT-TS", "#c0392b"))):
        vals = [res["E3_labels"][l][pol]["targeting_first30"]["mean"] for l in labs]
        err = [[v - res["E3_labels"][l][pol]["targeting_first30"]["ci95"][0] for v, l in zip(vals, labs)],
               [res["E3_labels"][l][pol]["targeting_first30"]["ci95"][1] - v for v, l in zip(vals, labs)]]
        ax.bar(x + (j - 0.5) * 0.38, vals, 0.38, yerr=err, color=col, label=pol, capsize=3)
    ax.set_xticks(x, [names[l] for l in labs])
    ax.set_ylabel("targeting rate, first 30 attempts")
    ax.set_title("E3 — how much does tactic-label quality matter?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "e3_label_quality.png", dpi=140)
    plt.close(fig)

    # E4 — change point
    fig, ax = plt.subplots(figsize=(8, 4.3))
    for name, y in cur["E4"].items():
        line(ax, name, y)
    ax.axvline(75, color="k", lw=1, ls="--")
    ax.text(77, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] else 1, "player fixes worst\nweakness (t = 75)", fontsize=8)
    ax.set(title="E4 — regret when the player improves mid-session", xlabel="attempt",
           ylabel="per-step regret (logits), 10-trial mean")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "e4_changepoint.png", dpi=140)
    plt.close(fig)

    print("figures ->", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
