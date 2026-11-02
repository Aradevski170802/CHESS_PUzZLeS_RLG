"""
Figures for the Results chapter (Chapter 6), drawn from the saved evaluation results.

Every number in these figures is read from a results file; nothing is typed in by hand.

  fig6_1_strength       rating-band classifier: confusion matrix and most useful features
                        (eval/research/rating_classification.json)
  fig6_2_weakness       weakness models against the category base rate, and split-half reliability
                        (eval/research/cohort_evaluation_puzzlenet.json)
  fig6_3_mined          categories and Maia probabilities of the mined puzzles
                        (eval/research/mined_puzzles_summary.json, totals only)
  fig6_4_recall         per-category recall, rules vs PuzzleNet (eval/neural/puzzlenet_evaluation.json)
  fig6_5_difficulty     rating error of difficulty models, overall and by rating band
                        (eval/neural/puzzlenet_evaluation.json, eval/neural/maia_puzzle_difficulty.json)
  fig6_6_ablations      PuzzleNet design variants (eval/neural/puzzlenet_evaluation.json)
  fig5_1_learning_curve PuzzleNet learning curve (same file)
  fig6_7_human          difficulty for a player of a given rating (eval/neural/human_difficulty.json)
  fig6_8_sim_time       simulation, trial by trial (eval/research/simulation_curves.json)
  fig6_9_sim_summary    simulation trade-off and learning gains (eval/research/simulation_results.json)
  fig6_10_real_use      forecasting the real attempts (eval/research/difficulty_model_evaluation.json)

Colour: the system's own method is blue (slot 1 of the reference palette), comparisons are grey;
when series must be told apart, the first three palette slots are used (blue, orange, aqua),
which the palette documents as colour-blind safe all-pairs. Every multi-series chart has a legend.

Usage: python -m scripts.report_figures.make_results_figures
The notebooks in notebooks/ set INLINE = True to show the same figures instead of saving them.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "figures"
INLINE = False

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, GREY_LIGHT = "#a9a79f", "#d6d4cd"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8.5, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 9,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "legend.fontsize": 8, "lines.linewidth": 2.0, "figure.dpi": 100,
})


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def save(fig, name: str) -> None:
    if INLINE:
        from IPython.display import Image, display
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        display(Image(buffer.getvalue()))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("saved", name)


def hgrid(ax):
    ax.grid(axis="x")
    ax.set_axisbelow(True)


def vgrid(ax):
    ax.grid(axis="y")
    ax.set_axisbelow(True)


# ── O1: strength ─────────────────────────────────────────────────────────────

FEATURE_NAMES = {
    "mean_wp_loss": "Average loss of winning chance",
    "mean_cp_loss": "Average centipawn loss",
    "median_wp_loss": "Median loss of winning chance",
    "hit_rate_all": "Share of critical positions solved",
    "blunders_pg": "Blunders per game",
    "mean_moves": "Average game length",
    "hit_rate_quiet": "Quiet moves solved",
    "inaccuracies_pg": "Inaccuracies per game",
}


def strength() -> None:
    d = load("eval/research/rating_classification.json")
    cm = np.array(d["models"]["Logistic regression"]["confusion_matrix"])
    names = d["classes"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.75), gridspec_kw={"width_ratios": [1, 1.25]})
    share = cm / cm.sum(axis=1, keepdims=True)
    a.imshow(share, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "blues", ["#fcfcfb", "#cde2fb", "#5598e7", "#184f95"]), vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            a.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                   color="white" if share[i, j] > 0.5 else INK)
    a.set_xticks(range(5), names, rotation=35, ha="right")
    a.set_yticks(range(5), names)
    a.set_xlabel("predicted band")
    a.set_ylabel("true band")
    a.spines[["left", "bottom"]].set_visible(False)
    a.set_title("(a) True against predicted band")

    feats = d["top_features_rf"][::-1]
    y = np.arange(len(feats))
    b.barh(y, [f["importance"] for f in feats], color=BLUE, height=0.6)
    b.set_yticks(y, [FEATURE_NAMES.get(f["feature"], f["feature"]) for f in feats])
    for yi, f in zip(y, feats):
        b.text(f["importance"] + 0.003, yi, f"{f['importance']:.2f}", va="center", fontsize=7.5,
               color=INK2)
    b.set_xlabel("importance in the random forest")
    b.set_xlim(0, 0.19)
    hgrid(b)
    b.set_title("(b) Most useful features")
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig6_1_strength")


# ── O1: weaknesses ───────────────────────────────────────────────────────────

WEAK_NAMES = [
    ("Global base rate", "One miss rate for everyone"),
    ("Player base rate", "Player's own overall miss rate"),
    ("Rule-based scorer", "Hand-tuned weakness scores"),
    ("RF (synthetic-trained)", "Forest trained on simulated players"),
    ("EB, shrink to player rate", "Shrunk to the player's own rate"),
    ("EB, shrink to population norms", "Shrunk to population norms (production)"),
    ("RF (real-data)", "Forest trained on real players"),
]


def weakness() -> None:
    d = load("eval/research/cohort_evaluation_puzzlenet.json")
    m = d["models"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.6), gridspec_kw={"width_ratios": [1.45, 1]})
    y = np.arange(len(WEAK_NAMES))[::-1]
    for yi, (key, label) in zip(y, WEAK_NAMES):
        diff = m[key]["logloss_minus_category_base"]
        lo, hi = m[key]["ci95"]
        col = BLUE if "population" in key or "real-data" in key else GREY
        a.plot([lo, hi], [yi, yi], color=col, lw=1.6, solid_capstyle="round")
        a.plot(diff, yi, "o", color=col, ms=6, mec="white", mew=1.2)
    a.axvline(0, color=INK2, lw=0.9)
    a.text(0.0004, -0.55, "= category\n   base rate", fontsize=7, color=INK2, va="bottom")
    a.set_yticks(y, [lab for _, lab in WEAK_NAMES])
    a.set_ylim(-0.7, len(WEAK_NAMES) - 0.4)
    a.set_xlabel("log loss minus the category base rate\n(left of 0 = better; 95% CI)")
    hgrid(a)
    a.set_title("(a) Predicting future misses")

    rel = sorted(d["reliability"], key=lambda r: r["spearman_brown"])
    yy = np.arange(len(rel))
    vals = [r["spearman_brown"] for r in rel]
    b.barh(yy, vals, color=BLUE, height=0.6)
    b.set_yticks(yy, [f"{r['category']} ({r['players']})" for r in rel])
    b.axvline(0, color=INK2, lw=0.9)
    b.set_xlim(-1, 1)
    b.axvspan(0.7, 1.0, color="#f0efec", zorder=0)
    b.text(0.72, len(rel) - 0.75, "usable\n(> 0.7)", fontsize=7, color=INK2, va="top")
    b.set_xlabel("split-half reliability (Spearman-Brown)")
    hgrid(b)
    b.set_title("(b) Stability of category hit rates")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig6_2_weakness")


# ── O2: mined puzzles ────────────────────────────────────────────────────────

def mined() -> None:
    s = load("eval/research/mined_puzzles_summary.json")
    cats = s["categories"]
    top = sorted(cats, key=cats.get, reverse=True)[:7]
    rest = [c for c in cats if c not in top]
    labels = top + [f"Other ({len(rest)} categories)"]
    counts = [cats[c] for c in top] + [sum(cats[c] for c in rest)]
    probs = np.array(s["maia_probabilities"])

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.5), gridspec_kw={"width_ratios": [1, 1.3]})
    y = np.arange(len(labels))[::-1]
    a.barh(y, counts, color=[GREY if l == "General" else BLUE for l in labels], height=0.62)
    a.set_yticks(y, labels)
    for yi, c in zip(y, counts):
        a.text(c + 0.8, yi, str(c), va="center", fontsize=7.5, color=INK2)
    a.set_xlabel(f"mined puzzles (n = {s['puzzles_kept']})")
    hgrid(a)
    a.set_title("(a) Category of the mined puzzles")

    b.hist(probs, bins=np.linspace(0, 1, 11), color=BLUE, edgecolor="#fcfcfb", linewidth=1.5)
    med = float(np.median(probs))
    b.axvline(med, color=INK, lw=1.0, ls="--")
    b.text(med + 0.015, b.get_ylim()[1] * 0.93, f"median {med:.0%}", fontsize=7.5, color=INK)
    b.set_xticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    b.set_xlabel("Maia: chance that a player of the owner's rating\nfinds the first move")
    b.set_ylabel(f"puzzles (n = {len(probs)})")
    vgrid(b)
    b.set_title("(b) How hard for their owner")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig6_3_mined")


# ── O3: labels ───────────────────────────────────────────────────────────────

def recall() -> None:
    h = load("eval/neural/puzzlenet_evaluation.json")["harness"]
    rules, net = h["rules"]["per_class"], h["puzzlenet"]["per_class"]
    cats = sorted(rules, key=lambda c: net[c]["recall"])
    y = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for yi, c in zip(y, cats):
        r, n = rules[c]["recall"], net[c]["recall"]
        ax.plot([r, n], [yi, yi], color=GRID, lw=2.2, zorder=1, solid_capstyle="round")
    ax.scatter([rules[c]["recall"] for c in cats], y, color=GREY, s=26, zorder=2,
               label="Rule-based tagger", edgecolors="white", linewidths=0.8)
    ax.scatter([net[c]["recall"] for c in cats], y, color=BLUE, s=30, zorder=3,
               label="PuzzleNet", edgecolors="white", linewidths=0.8)
    ax.set_yticks(y, cats)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    ax.set_xlabel("recall: share of the category's puzzles labelled correctly (held-out sample)")
    hgrid(ax)
    ax.legend(loc="lower right")
    save(fig, "fig6_4_recall")


# ── O3: difficulty ───────────────────────────────────────────────────────────

def difficulty() -> None:
    ev = load("eval/neural/puzzlenet_evaluation.json")["test"]
    mp = load("eval/neural/maia_puzzle_difficulty.json")
    base = ev["rating_baselines"]
    rows = [
        ("Average rating", base["constant"]["rmse"], GREY),
        ("Miner's formula", base["miner_formula"]["rmse"], GREY),
        ("Length and mate (linear)", base["length_mate_linear"]["rmse"], GREY),
        ("Maia", mp["models"]["Maia"]["rmse"], GREY),
        ("Boosted trees, hand-built features", base["gbdt_engineered"]["rmse"], GREY),
        ("PuzzleNet", ev["puzzlenet"]["rating"]["rmse"], BLUE),
        ("PuzzleNet + Maia", mp["models"]["PuzzleNet + Maia"]["rmse"], BLUE),
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.6), gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(rows))[::-1]
    a.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.62)
    a.set_yticks(y, [r[0] for r in rows])
    for yi, r in zip(y, rows):
        a.text(r[1] + 6, yi, f"{r[1]:.0f}", va="center", fontsize=7.5, color=INK2)
    a.set_xlim(0, 620)
    a.set_xlabel("RMSE in rating points (lower is better)")
    hgrid(a)
    a.set_title("(a) Error in predicting the rating")

    bands = list(mp["rmse_by_rating_band"].keys())
    x = np.arange(len(bands))
    for name, col, mk in (("Maia", ORANGE, "s"), ("PuzzleNet", BLUE, "o"),
                          ("PuzzleNet + Maia", AQUA, "D")):
        vals = [mp["rmse_by_rating_band"][bd][name] for bd in bands]
        b.plot(x, vals, marker=mk, color=col, ms=5.5, label=name, mec="white", mew=0.8)
    b.set_xticks(x, ["< 1200", "1200–\n1600", "1600–\n2000", "2000 +"][:len(bands)])
    b.set_xlabel("Lichess rating of the puzzle")
    b.set_ylabel("RMSE (points)")
    b.set_ylim(0, 640)
    vgrid(b)
    b.legend(loc="upper left")
    b.set_title("(b) Error by rating band")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig6_5_difficulty")


ABL = [
    ("abl_full_1m", "Base model"),
    ("abl_linear", "No hidden layers"),
    ("abl_raw", "Raw board and moves only"),
    ("abl_engineered", "Hand-built features only"),
    ("abl_cat_only", "Category head only"),
    ("abl_rating_only", "Rating head only"),
    ("abl_beta0", "β = 0 (plain likelihood)"),
    ("abl_beta1", "β = 1"),
    ("abl_wide", "Wider layers (1024, 512)"),
]


def ablations() -> None:
    ms = load("eval/neural/puzzlenet_evaluation.json")["models"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.7), sharey=True)
    y = np.arange(len(ABL))[::-1]
    base_k = ms["abl_full_1m"]["cat"]["cohens_kappa"]
    base_r = ms["abl_full_1m"]["rating"]["rmse"]
    for ax, metric, lim, fmt, refv in ((a, "kappa", (0.70, 0.90), "{:.3f}", base_k),
                                       (b, "rmse", (250, 340), "{:.0f}", base_r)):
        for yi, (key, _) in zip(y, ABL):
            m = ms[key]
            has = ("cat" in m["setup"]["heads"]) if metric == "kappa" else ("rating" in m["setup"]["heads"])
            if not has:
                ax.text(lim[0] + 0.01 * (lim[1] - lim[0]), yi, "not trained", va="center",
                        fontsize=7, color=MUTED)
                continue
            v = m["cat"]["cohens_kappa"] if metric == "kappa" else m["rating"]["rmse"]
            col = BLUE if key == "abl_full_1m" else GREY
            ax.plot([lim[0], v], [yi, yi], color=GRID, lw=2, zorder=1)
            ax.plot(v, yi, "o", color=col, ms=6, mec="white", mew=1, zorder=2)
            ax.text(v + 0.012 * (lim[1] - lim[0]), yi + 0.28, fmt.format(v), fontsize=7, color=INK2)
        ax.axvline(refv, color=BLUE, lw=0.8, alpha=0.6)
        ax.set_xlim(*lim)
        hgrid(ax)
    a.set_yticks(y, [lab for _, lab in ABL])
    a.set_xlabel("Cohen's κ for the category (higher is better)")
    b.set_xlabel("RMSE for the rating (lower is better)")
    a.set_title("(a) Category")
    b.set_title("(b) Rating")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig6_6_ablations")


def learning_curve() -> None:
    ev = load("eval/neural/puzzlenet_evaluation.json")
    runs = [ev["models"][k] for k in ("lc_30k", "lc_100k", "lc_300k", "abl_full_1m")]
    runs.append(ev["test"]["puzzlenet"])
    n = [r["setup"]["train_size"] for r in runs]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
    for ax, vals, label, fmt in ((axes[0], [r["cat"]["cohens_kappa"] for r in runs],
                                  "Cohen's κ, category", "{:.2f}"),
                                 (axes[1], [r["rating"]["rmse"] for r in runs],
                                  "RMSE, rating (points)", "{:.0f}")):
        ax.semilogx(n, vals, "o-", color=BLUE, ms=5, mec="white", mew=0.8)
        for xi, yi in zip(n, vals):
            ax.annotate(fmt.format(yi), (xi, yi), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7.5, color=INK2)
        ax.set_xlabel("training puzzles (log scale)")
        ax.set_ylabel(label)
        lo, hi = min(vals), max(vals)
        ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.22 * (hi - lo))
        ax.grid(which="major")
        ax.set_axisbelow(True)
    axes[0].set_title("(a) Category")
    axes[1].set_title("(b) Rating")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig5_1_learning_curve")


# ── O3: difficulty for a given player ────────────────────────────────────────

HUMAN = [
    ("global base rate", "Global base rate"),
    ("player rating only", "Player's rating only"),
    ("category base rate", "Category base rate"),
    ("PuzzleNet puzzle difficulty", "PuzzleNet's population rating"),
    ("trunk + rating (human difficulty)", "PuzzleNet trunk + rating"),
    ("Maia at the player's level", "Maia at the player's level"),
    ("Maia, all levels + rating", "Maia, all levels + rating"),
]


def human() -> None:
    d = load("eval/neural/human_difficulty.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.6), gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(HUMAN))[::-1]
    aucs = [d["models"][k]["auc"] for k, _ in HUMAN]
    cols = [ORANGE if "Maia" in k else (BLUE if "trunk" in k else GREY) for k, _ in HUMAN]
    a.barh(y, [v - 0.5 for v in aucs], left=0.5, color=cols, height=0.62)
    for yi, v in zip(y, aucs):
        a.text(v + 0.005, yi, f"{v:.2f}", va="center", fontsize=7.5, color=INK2)
    a.set_yticks(y, [lab for _, lab in HUMAN])
    a.set_xlim(0.5, 0.86)
    a.set_xlabel("AUC (0.5 = chance)")
    hgrid(a)
    a.set_title("(a) Will this player find the move?")

    bands = list(d["by_rating_band"].keys())
    x = np.arange(len(bands))
    for key, lab, col, mk in (("category base rate", "Category base rate", GREY, "^"),
                              ("trunk + rating (human difficulty)", "PuzzleNet trunk + rating", BLUE, "o"),
                              ("Maia, all levels + rating", "Maia, all levels + rating", ORANGE, "s")):
        b.plot(x, [d["by_rating_band"][bd][key] for bd in bands], marker=mk, color=col, ms=5.5,
               label=lab, mec="white", mew=0.8)
    b.set_xticks(x, ["< 1200", "1200–\n1600", "1600–\n2000", "2000 +"][:len(bands)])
    b.set_xlabel("player's rating")
    b.set_ylabel("log loss (lower is better)")
    b.set_ylim(0.47, 0.72)
    vgrid(b)
    b.legend(loc="upper right", fontsize=7)
    b.set_title("(b) By the player's rating")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig6_7_human")


# ── O4/O5: simulation ────────────────────────────────────────────────────────

def _smooth(v, w=9):
    v = np.asarray(v, float)
    k = np.ones(w) / w
    pad = np.pad(v, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")


def sim_time() -> None:
    c = load("eval/research/simulation_curves.json")["E2_none"]
    pols = [("Random", "Random category", GREY, "-"), ("Beta-TS", "Beta bandit", ORANGE, "-"),
            ("IRT-TS", "IRT selector (production)", BLUE, "-")]
    t = np.arange(1, 151)
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for key, lab, col, ls in pols:
        a.plot(t, _smooth(c[key]["targeting"]), color=col, ls=ls, label=lab)
        b.plot(t, _smooth(c[key]["p"]), color=col, ls=ls, label=lab)
    a.axhline(3 / 23, color=MUTED, lw=0.8, ls="--")
    a.text(150, 3 / 23 + 0.02, "chance", ha="right", fontsize=7, color=MUTED)
    a.set_ylim(0, 0.6)
    a.set_yticks(np.linspace(0, 0.6, 4), [f"{v:.0%}" for v in np.linspace(0, 0.6, 4)])
    a.set_xlabel("attempt")
    a.set_ylabel("puzzles from the 3 weakest categories")
    vgrid(a)
    a.set_title("(a) Finding the weak categories")

    b.axhspan(0.50, 0.85, color="#eaf2fc", zorder=0)
    b.axhline(0.65, color=BLUE, lw=0.8, ls="--")
    b.text(4, 0.865, "well pitched (50–85%)", fontsize=7, color=INK2)
    b.text(4, 0.665, "target 65%", fontsize=7, color=BLUE)
    b.set_ylim(0.2, 1.0)
    b.set_yticks(np.linspace(0.2, 1.0, 5), [f"{v:.0%}" for v in np.linspace(0.2, 1.0, 5)])
    b.set_xlabel("attempt")
    b.set_ylabel("true solve chance of the puzzle")
    vgrid(b)
    b.set_title("(b) Difficulty of the served puzzles")
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig6_8_sim_time")


SIM_POLS = [
    ("Random", "Random"), ("Round-robin", "Round-robin"), ("Static profile (top-3)", "Static profile"),
    ("Greedy (Beta mean)", "Greedy"), ("Beta-TS", "Beta bandit"),
    ("IRT-TS (Elo band)", "IRT, rating ± 300"), ("IRT-TS", "IRT selector"), ("Oracle", "Oracle"),
]


def sim_summary() -> None:
    r = load("eval/research/simulation_results.json")["E2_main"]
    t = r["none"]["table"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.8), gridspec_kw={"width_ratios": [1.1, 1]})
    # Random and round-robin land on the same point, so they share one label.
    offsets = {"Random": (0, -13), "Round-robin": None, "Static profile (top-3)": (7, -3),
               "Greedy (Beta mean)": (-7, 4), "Beta-TS": (7, -8), "IRT-TS (Elo band)": (-7, 4),
               "IRT-TS": (7, -3), "Oracle": (7, -3)}
    names = {"Random": "Random, round-robin"}
    for key, lab in SIM_POLS:
        x, y = t[key]["frac_frustrating"]["mean"], t[key]["targeting_last50"]["mean"]
        col = BLUE if key == "IRT-TS" else (INK if key == "Oracle" else GREY)
        a.plot(x, y, "o", color=col, ms=7 if key == "IRT-TS" else 6, mec="white", mew=1)
        if offsets[key] is None:
            continue
        dx, dy = offsets[key]
        ha = "right" if dx < 0 else ("center" if dx == 0 else "left")
        a.annotate(names.get(key, lab), (x, y), textcoords="offset points", xytext=(dx, dy),
                   fontsize=7.5, color=BLUE if key == "IRT-TS" else INK2, ha=ha, va="center",
                   fontweight="bold" if key == "IRT-TS" else "normal")
    a.set_xlim(-0.03, 0.56)
    a.set_ylim(0, 1.08)
    a.set_xticks(np.linspace(0, 0.5, 6), [f"{v:.0%}" for v in np.linspace(0, 0.5, 6)])
    a.set_yticks(np.linspace(0, 1, 5), [f"{v:.0%}" for v in np.linspace(0, 1, 5)])
    a.set_xlabel("puzzles too hard (solve chance < 30%)")
    a.set_ylabel("targeting, last 50 attempts")
    a.grid(True)
    a.set_axisbelow(True)
    a.set_title("(a) Targeting against frustration")
    a.text(0.55, 1.0, "better: up and to the left", fontsize=7, color=MUTED, ha="right")

    pols = [("Random", "Random"), ("Static profile (top-3)", "Static\nprofile"),
            ("Beta-TS", "Beta\nbandit"), ("IRT-TS", "IRT\nselector")]
    x = np.arange(len(pols))
    w = 0.36
    for i, (world, lab, col) in enumerate((("zpd", "Learns most at the edge of ability", BLUE),
                                           ("error", "Learns most from failures", ORANGE))):
        vals = [r[world]["table"][k]["gain_weak"]["mean"] for k, _ in pols]
        b.bar(x + (i - 0.5) * w, vals, width=w - 0.04, color=col, label=lab)
        for bx, v in zip(x + (i - 0.5) * w, vals):
            b.text(bx, v + 0.015, f"{v:.2f}", ha="center", fontsize=7, color=INK2)
    b.set_xticks(x, [lab for _, lab in pols])
    b.set_ylabel("improvement of the weak categories\n(logits)")
    b.set_ylim(0, 0.85)
    vgrid(b)
    b.legend(loc="upper left", fontsize=7.2)
    b.set_title("(b) Learning under two assumptions")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig6_9_sim_summary")


# ── O5: real use ─────────────────────────────────────────────────────────────

REAL = [
    ("Static Elo", "Fixed Elo prediction"),
    ("Glicko-2 (pool frozen)", "Glicko-2, puzzle ratings fixed"),
    ("IRT, per category + PuzzleNet", "IRT + PuzzleNet ratings"),
    ("IRT, θ only", "IRT, overall ability only"),
    ("Glicko-2 (symmetric)", "Glicko-2"),
    ("IRT, per category", "IRT per category (production)"),
    ("Base rate", "Running average of solve rate"),
]


def real_use() -> None:
    d = load("eval/research/difficulty_model_evaluation.json")
    m = d["models"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    y = np.arange(len(REAL))[::-1]
    for ax_, metric in ((a, "log_loss"), (b, "mean_prediction")):
        for yi, (key, _) in zip(y, REAL):
            v = m[key][metric]
            col = BLUE if key == "IRT, per category" else GREY
            ax_.plot([0 if metric == "log_loss" else d["solve_rate"], v], [yi, yi], color=GRID, lw=2,
                     zorder=1)
            ax_.plot(v, yi, "o", color=col, ms=6, mec="white", mew=1, zorder=2)
            label = f"{v:.2f}" if metric == "log_loss" else f"{v:.0%}"
            ax_.text(v, yi + 0.32, label, fontsize=7, color=INK2, ha="center")
        hgrid(ax_)
        ax_.set_ylim(-1.0, len(REAL) - 0.35)
    a.set_yticks(y, [lab for _, lab in REAL])
    a.set_xlim(0, 1.4)
    a.set_xlabel("log loss (lower is better)")
    a.set_title("(a) Forecast quality")
    b.axvline(d["solve_rate"], color=INK, lw=1.0, ls="--")
    b.text(d["solve_rate"] + 0.008, -0.75, f"actual\n{d['solve_rate'] * 100:.1f}%", fontsize=7,
           ha="left", va="center", color=INK)
    b.axvline(0.65, color=BLUE, lw=0.8, ls=":")
    b.text(0.645, -0.75, "target\n65%", fontsize=7, color=BLUE, ha="right", va="center")
    b.set_xlim(0.35, 0.95)
    b.set_xticks(np.linspace(0.4, 0.9, 6), [f"{v:.0%}" for v in np.linspace(0.4, 0.9, 6)])
    b.set_xlabel("average predicted solve chance")
    b.set_title("(b) Expected against actual")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig6_10_real_use")


if __name__ == "__main__":
    strength()
    weakness()
    mined()
    recall()
    difficulty()
    ablations()
    learning_curve()
    human()
    sim_time()
    sim_summary()
    real_use()
