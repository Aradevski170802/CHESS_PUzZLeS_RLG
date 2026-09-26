"""
Figures for the PuzzleNet results. Reads only the JSON written by the training and
evaluation scripts, so it runs in seconds and never touches the dataset.

    eval/neural/training/<name>.json      learning curves
    eval/neural/puzzlenet_evaluation.json harness, test split, ablations
    eval/neural/engine_pv_check.json      deployment condition
    eval/neural/label_simulation.json     downstream simulation

Writes eval/neural/figures/*.png. Missing inputs are skipped with a note.
Usage: python -m scripts.neural.make_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

EV = Path("eval/neural")
FIG = EV / "figures"
RULES_C, NN_C, GREY = "#8c8c8c", "#1f5fa8", "#c9c9c9"


def load(name: str):
    p = EV / name
    return json.loads(p.read_text("utf-8")) if p.exists() else None


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name)


def training_curves() -> None:
    h = load("training/puzzlenet.json")
    if not h:
        return print("skip training curves (no eval/neural/training/puzzlenet.json)")
    tr, va = h["history"]["train"], h["history"]["val"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot([r["step"] for r in tr], [r["total"] for r in tr], color=NN_C, lw=1)
    ax[0].plot([r["step"] for r in va], [r["selection_score"] for r in va], "o-", color="#d9531e",
               ms=3, label="validation (selection score)")
    ax[0].set(title="Training loss (running mean) and validation score", xlabel="step")
    ax[0].legend()
    ax[1].plot([r["step"] for r in va], [r["cat_kappa"] for r in va], "o-", color=NN_C, ms=3,
               label="Cohen's κ")
    ax[1].plot([r["step"] for r in va], [r["cat_accuracy"] for r in va], "s-", color="#6aa84f",
               ms=3, label="accuracy")
    ax[1].set(title="Category head on validation", xlabel="step")
    ax[1].legend()
    ax[2].plot([r["step"] for r in va], [r["rating_rmse"] for r in va], "o-", color="#d9531e", ms=3)
    ax[2].set(title="Rating head: validation RMSE (rating points)", xlabel="step")
    for a in ax:
        a.grid(alpha=.3)
    save(fig, "nn_training_curves.png")


def harness() -> None:
    ev = load("puzzlenet_evaluation.json")
    if not ev:
        return print("skip harness figures (no puzzlenet_evaluation.json)")
    hz = ev["harness"]
    metrics = [("strict_agreement", "strict agreement\n(uniform)"), ("cohens_kappa", "Cohen's κ\n(uniform)"),
               ("macro_recall_stratified", "macro recall\n(stratified)"),
               ("macro_precision_uniform", "macro precision\n(uniform)")]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(metrics))
    for off, key, col, lab in ((-0.2, "rules", RULES_C, "rule-based tagger"),
                               (0.2, "puzzlenet", NN_C, "PuzzleNet")):
        vals = [hz[key][m] for m, _ in metrics]
        bars = ax.bar(x + off, vals, 0.4, color=col, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, [l for _, l in metrics])
    ax.set_ylim(0, 1.2)
    ax.set_title(f"Held-out labeller harness (seed {hz['seed']}): the same "
                 f"{hz['uniform_n']:,} + {hz['stratified_n']:,} puzzles for both labellers")
    ax.legend(loc="upper center", ncol=2, frameon=False)
    ax.grid(axis="y", alpha=.3)
    save(fig, "nn_harness_comparison.png")

    cats = list(hz["puzzlenet"]["per_class"])
    cats = [c for c in cats if hz["puzzlenet"]["per_class"][c]["support_stratified"]]
    rr = [hz["rules"]["per_class"][c]["recall"] or 0 for c in cats]
    nr = [hz["puzzlenet"]["per_class"][c]["recall"] or 0 for c in cats]
    order = np.argsort(nr)
    fig, ax = plt.subplots(figsize=(8, 8))
    y = np.arange(len(cats))
    ax.barh(y - 0.2, np.array(rr)[order], 0.4, color=RULES_C, label="rule-based tagger")
    ax.barh(y + 0.2, np.array(nr)[order], 0.4, color=NN_C, label="PuzzleNet")
    ax.set_yticks(y, [cats[i] for i in order], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("recall on the stratified harness sample")
    ax.set_title("Per-category recall")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=.3)
    save(fig, "nn_per_class_recall.png")

    conf = load("label_confusion.json")
    if conf:
        labels = conf["confusion"]["labels"]
        m = np.array(conf["confusion"]["puzzlenet"])
        keep = [i for i, c in enumerate(labels) if hz["puzzlenet"]["per_class"][c]["support_stratified"]]
        m = m[np.ix_(keep, range(len(labels)))]
        fig, ax = plt.subplots(figsize=(11, 8.5))
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(labels)), labels, rotation=75, ha="right", fontsize=8)
        ax.set_yticks(range(len(keep)), [labels[i] for i in keep], fontsize=8)
        for i in range(len(keep)):
            for j in range(len(labels)):
                if m[i, j] >= 0.05:
                    ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if m[i, j] > 0.55 else "black")
        ax.set_xlabel("PuzzleNet label")
        ax.set_ylabel("Lichess primary category")
        ax.set_title("PuzzleNet confusion on the held-out stratified sample (row-normalised)")
        fig.colorbar(im, fraction=0.03)
        save(fig, "nn_confusion.png")


def calibration_and_rating() -> None:
    ev = load("puzzlenet_evaluation.json")
    if not ev:
        return
    t = ev["test"]["puzzlenet"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    rel = t["calibration"]["reliability"]
    conf = [r["confidence"] for r in rel]
    acc = [r["accuracy"] for r in rel]
    ax[0].plot([0, 1], [0, 1], "--", color=GREY)
    ax[0].plot(conf, acc, "o-", color=NN_C)
    ax[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="confidence", ylabel="accuracy",
              title=f"Category reliability (test split)\nECE {t['calibration']['ece_uncalibrated']:.3f}"
                    f" -> {t['calibration']['ece_calibrated']:.3f} after temperature scaling")
    r = t["rating"]
    levels = [50, 80, 95]
    ax[1].bar([str(l) for l in levels], [r[f"coverage_{l}"] * 100 for l in levels], color=NN_C,
              label="observed")
    ax[1].plot([str(l) for l in levels], levels, "k_", ms=40, mew=2, label="nominal")
    ax[1].set(ylim=(0, 100), xlabel="predictive interval (%)", ylabel="coverage (%)",
              title="Rating intervals: nominal vs observed coverage")
    ax[1].legend()
    bands = t["by_band"]
    ax[2].bar([b["band"] for b in bands], [b["bias"] for b in bands], color="#d9531e")
    ax[2].axhline(0, color="k", lw=.8)
    ax[2].set(xlabel="true rating band", ylabel="mean error (predicted - true)",
              title="Regression to the mean by rating band")
    ax[2].tick_params(axis="x", labelrotation=20, labelsize=8)
    for a in ax:
        a.grid(alpha=.3)
    save(fig, "nn_calibration.png")

    base = ev["test"].get("rating_baselines", {})
    if base:
        names = {"constant": "constant", "miner_formula": "miner's formula",
                 "length_mate_linear": "length x mate\nlinear", "gbdt_engineered": "GBDT on\nhand-built"}
        rows = [(names[k], base[k]["rmse"], base[k].get("spearman_rho")) for k in names if k in base]
        rows.append(("PuzzleNet", r["rmse"], r["spearman_rho"]))
        fig, ax = plt.subplots(figsize=(9, 4))
        cols = [GREY] * (len(rows) - 1) + [NN_C]
        bars = ax.bar([n for n, _, _ in rows], [v for _, v, _ in rows], color=cols)
        for b, (_, v, rho) in zip(bars, rows):
            label = f"{v:.0f}" + (f"\nρ={rho:.2f}" if rho is not None else "")
            ax.text(b.get_x() + b.get_width() / 2, v + 5, label, ha="center", fontsize=9)
        ax.set_ylabel("RMSE vs Lichess rating (points)")
        ax.set_title(f"Difficulty prediction on the test split ({ev['test_n']:,} puzzles)")
        ax.grid(axis="y", alpha=.3)
        save(fig, "nn_rating_baselines.png")


def ablations() -> None:
    ev = load("puzzlenet_evaluation.json")
    if not ev or not ev.get("models"):
        return print("skip ablations")
    ms = ev["models"]
    abl = {k: v for k, v in ms.items() if k.startswith("abl_")}
    if abl:
        order = sorted(abl, key=lambda k: abl[k]["cat"]["cohens_kappa"])
        fig, ax = plt.subplots(1, 2, figsize=(14, 0.45 * len(order) + 1.5))
        y = np.arange(len(order))
        kap = [abl[k]["cat"]["cohens_kappa"] if "cat" in abl[k]["setup"]["heads"] else np.nan for k in order]
        rmse = [abl[k]["rating"]["rmse"] if "rating" in abl[k]["setup"]["heads"] else np.nan for k in order]
        ax[0].barh(y, kap, color=NN_C)
        ax[1].barh(y, rmse, color="#d9531e")
        for a, vals, fmt in ((ax[0], kap, "{:.3f}"), (ax[1], rmse, "{:.0f}")):
            a.set_yticks(y, [k.replace("abl_", "") for k in order])
            for yi, v in zip(y, vals):
                if np.isfinite(v):
                    a.text(v, yi, " " + fmt.format(v), va="center", fontsize=8)
            a.grid(axis="x", alpha=.3)
        ax[0].set_title("Category κ on the test split (1M-puzzle training subset)")
        ax[1].set_title("Rating RMSE on the test split")
        save(fig, "nn_ablations.png")
    lc = {k: v for k, v in ms.items() if k.startswith("lc_") or k == "abl_full_1m"}
    if lc and "full" not in lc:
        full = ev["test"]["puzzlenet"]
        lc["full"] = {"cat": full["cat"], "rating": full["rating"],
                      "setup": {"train_size": full["setup"]["train_size"]}}
    if len(lc) >= 3:
        pts = sorted(lc.values(), key=lambda v: v["setup"]["train_size"])
        n = [p["setup"]["train_size"] for p in pts]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].semilogx(n, [p["cat"]["cohens_kappa"] for p in pts], "o-", color=NN_C)
        ax[0].set(xlabel="training puzzles", ylabel="Cohen's κ (test)", title="Data scaling: category")
        ax[1].semilogx(n, [p["rating"]["rmse"] for p in pts], "o-", color="#d9531e")
        ax[1].set(xlabel="training puzzles", ylabel="RMSE (test)", title="Data scaling: rating")
        for a in ax:
            a.grid(alpha=.3, which="both")
        save(fig, "nn_data_scaling.png")


def engine_and_simulation() -> None:
    pv = load("engine_pv_check.json")
    if pv:
        cond = pv["conditions"]
        order = [k for k in ("rules_solution", "rules_engine_7", "puzzlenet_solution",
                             "puzzlenet_engine_1", "puzzlenet_engine_3", "puzzlenet_engine_5",
                             "puzzlenet_engine_7") if k in cond]
        fig, ax = plt.subplots(figsize=(10, 4))
        cols = [RULES_C if k.startswith("rules") else NN_C for k in order]
        bars = ax.bar(range(len(order)), [cond[k]["cohens_kappa"] for k in order], color=cols)
        for b, k in zip(bars, order):
            ax.text(b.get_x() + b.get_width() / 2, cond[k]["cohens_kappa"] + 0.01,
                    f"{cond[k]['cohens_kappa']:.2f}", ha="center", fontsize=9)
        ax.set_xticks(range(len(order)), [k.replace("_", "\n", 1) for k in order], fontsize=8)
        ax.set_ylabel("Cohen's κ vs Lichess category")
        ax.set_title(f"Solution line vs Stockfish line ({pv['n']:,} held-out puzzles, "
                     f"{pv['nodes']:,} nodes)")
        ax.grid(axis="y", alpha=.3)
        save(fig, "nn_engine_lines.png")
    sim = load("label_simulation.json")
    if sim:
        labs = ["none", "rules", "puzzlenet", "perfect"]
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(labs))
        for off, pol, col in ((-0.2, "Beta-TS", GREY), (0.2, "IRT-TS", NN_C)):
            m = [sim["labels"][l][pol]["targeting_first30"]["mean"] for l in labs]
            lo = [m[i] - sim["labels"][l][pol]["targeting_first30"]["ci95"][0] for i, l in enumerate(labs)]
            ax.bar(x + off, m, 0.4, yerr=lo, capsize=3, color=col, label=pol)
        ax.set_xticks(x, ["no game\nevidence", "rule-based\nlabels", "PuzzleNet\nlabels", "perfect\nlabels"])
        ax.set_ylabel("weak-category targeting, first 30 puzzles")
        ax.set_title(f"Downstream effect of label quality (simulation, {sim['runs']} paired runs)")
        ax.legend()
        ax.grid(axis="y", alpha=.3)
        save(fig, "nn_label_simulation.png")


if __name__ == "__main__":
    training_curves()
    harness()
    calibration_and_rating()
    ablations()
    engine_and_simulation()
