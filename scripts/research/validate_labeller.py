"""
Validate the tactic labeller (_classify_tactic) against Lichess puzzle themes.

Every weakness estimate in the system inherits the labeller's errors: game
analysis labels each missed best move with _classify_tactic, and those labels
decide which bandit arm the evidence goes to. This script measures how often
that label agrees with an independent source.

Reference labels: Lichess puzzle themes, resolved with the same
MOTIF_PRIORITY rule the app uses for the puzzle pool. Lichess themes are
themselves produced by an automatic tagger, so this is AGREEMENT with a
second system, not accuracy against human ground truth — stated wherever the
numbers are quoted.

Protocol
────────
Lichess puzzle FENs are the position BEFORE the opponent's move; the player's
first solution move is Moves[1] after pushing Moves[0]. That is exactly the
situation in game analysis (classify the best move in a position), so the
first-move label is the operative one.

    strict     label == resolved primary category
    lenient    label ∈ all categories the puzzle's themes map to
    any-move   some player move of the solution gets the primary label
               (separates "only looked at move 1 of a combination" from
               genuine mislabels)

Two samples: a UNIFORM random sample (prevalence-true: overall agreement,
Cohen's κ, per-class precision) and a STRATIFIED sample of up to N per class
(per-class recall with enough support for rare classes).

Usage: python -m scripts.research.validate_labeller
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import chess
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from src.data.puzzle_loader import THEME_CATEGORIES, WEAKNESS_CATEGORIES, resolve_primary_category
from src.puzzles.generator import _classify_tactic
from src.puzzles.tactic_tagger import tag_line

PARQUET = Path("data/processed/puzzles_full.parquet")
OUT = Path("eval/research")
LABELS = WEAKNESS_CATEGORIES + ["General"]


def load(seed: int, uniform_n: int, per_class: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pq.read_table(PARQUET, columns=["PuzzleId", "FEN", "Moves", "Themes"])
    df = table.to_pandas()
    df["primary"] = df["Themes"].map(resolve_primary_category)
    uniform = df.sample(n=uniform_n, random_state=seed)
    # Shuffle, then take the first `per_class` rows of each class: a uniform
    # sample of min(class size, per_class) per class.
    strat = df.sample(frac=1.0, random_state=seed).groupby("primary").head(per_class)
    return uniform.reset_index(drop=True), strat.reset_index(drop=True)


def label_puzzle(fen: str, moves: str) -> tuple[str, list[str]]:
    """(first-move label, labels of every player move in the solution)."""
    ms = moves.split()
    board = chess.Board(fen)
    board.push_uci(ms[0])
    labels = []
    for i, uci in enumerate(ms[1:]):
        mv = chess.Move.from_uci(uci)
        if i % 2 == 0:                      # player's moves: 1, 3, 5, ...
            labels.append(_classify_tactic(board, mv) if mv in board.legal_moves else "General")
        board.push(mv)
    return (labels[0] if labels else "General"), labels


def tag_new(fen: str, moves: str) -> str:
    """New line-level tagger on the solution line (mate decided by play-out)."""
    ms = moves.split()
    board = chess.Board(fen)
    board.push_uci(ms[0])
    return tag_line(board, [chess.Move.from_uci(u) for u in ms[1:]])


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    first, anym, lenient, new, new_len = [], [], [], [], []
    for fen, moves, themes, prim in zip(df["FEN"], df["Moves"], df["Themes"], df["primary"]):
        f, all_labels = label_puzzle(fen, moves)
        n = tag_new(fen, moves)
        cats = {c for t in themes.split() for c in THEME_CATEGORIES.get(t, [])}
        first.append(f)
        anym.append(prim in all_labels)
        lenient.append(f in cats or f == prim)
        new.append(n)
        new_len.append(n in cats or n == prim)
    out = df.copy()
    out["pred"], out["any_move"], out["lenient"] = first, anym, lenient
    out["pred_new"], out["lenient_new"] = new, new_len
    return out


def summarise(uniform: pd.DataFrame, strat: pd.DataFrame, col: str, lenient_col: str) -> dict:
    per_class = {}
    for c in LABELS:
        s = strat[strat["primary"] == c]
        u_pred = uniform[uniform[col] == c]
        per_class[c] = {
            "recall": round(float((s[col] == c).mean()), 4) if len(s) else None,
            "precision": round(float((u_pred["primary"] == c).mean()), 4) if len(u_pred) else None,
            "predicted_uniform": int(len(u_pred)),
        }
    recalls = [v["recall"] for v in per_class.values() if v["recall"] is not None]
    precisions = [v["precision"] for v in per_class.values() if v["precision"] is not None]
    emitted = set(uniform[col]) | set(strat[col])
    return {
        "strict_agreement": round(float((uniform[col] == uniform["primary"]).mean()), 4),
        "lenient_agreement": round(float(uniform[lenient_col].mean()), 4),
        "cohens_kappa": round(float(cohen_kappa_score(uniform["primary"], uniform[col])), 4),
        "macro_recall_stratified": round(float(np.mean(recalls)), 4),
        "macro_precision_uniform": round(float(np.mean(precisions)), 4),
        "labels_never_emitted": [c for c in WEAKNESS_CATEGORIES if c not in emitted],
        "per_class": per_class,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uniform", type=int, default=30000)
    ap.add_argument("--per-class", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _heldout")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    uniform, strat = load(args.seed, args.uniform, args.per_class)
    uniform, strat = annotate(uniform), annotate(strat)

    emitted = sorted(set(uniform["pred"]) | set(strat["pred"]))
    never = [c for c in WEAKNESS_CATEGORIES if c not in emitted]

    kappa = cohen_kappa_score(uniform["primary"], uniform["pred"])
    per_class = {}
    for c in LABELS:
        s = strat[strat["primary"] == c]
        u_pred = uniform[uniform["pred"] == c]
        per_class[c] = {
            "support_stratified": int(len(s)),
            "recall": round(float((s["pred"] == c).mean()), 4) if len(s) else None,
            "any_move_recall": round(float(s["any_move"].mean()), 4) if len(s) else None,
            "predicted_uniform": int(len(u_pred)),
            "precision": round(float((u_pred["primary"] == c).mean()), 4) if len(u_pred) else None,
            "prevalence_uniform": round(float((uniform["primary"] == c).mean()), 4),
        }
    recalls = [v["recall"] for v in per_class.values() if v["recall"] is not None]
    report = {
        "reference": "Lichess puzzle themes resolved by MOTIF_PRIORITY (automatic tagger: agreement, not accuracy)",
        "uniform_n": len(uniform), "stratified_n": len(strat),
        "strict_agreement": round(float((uniform["pred"] == uniform["primary"]).mean()), 4),
        "lenient_agreement": round(float(uniform["lenient"].mean()), 4),
        "any_move_agreement": round(float(uniform["any_move"].mean()), 4),
        "cohens_kappa": round(float(kappa), 4),
        "macro_recall_stratified": round(float(np.mean(recalls)), 4),
        "labels_never_emitted": never,
        "top_confusions": [
            {"true": t, "pred": p, "count": int(n)}
            for (t, p), n in Counter(zip(uniform["primary"], uniform["pred"])).most_common(40)
            if t != p
        ][:15],
        "per_class": per_class,
    }
    old = summarise(uniform, strat, "pred", "lenient")
    new = summarise(uniform, strat, "pred_new", "lenient_new")
    report["comparison"] = {
        "old_classify_tactic": {k: v for k, v in old.items() if k != "per_class"},
        "new_tactic_tagger":   {k: v for k, v in new.items() if k != "per_class"},
    }
    report["new_per_class"] = new["per_class"]
    # Row-normalised P(predicted | true) on the stratified sample, for the
    # label-noise model in src/evaluation/simulation.py.
    conf = {"labels": LABELS}
    for key, col in (("old", "pred"), ("new", "pred_new")):
        m = confusion_matrix(strat["primary"], strat[col], labels=LABELS).astype(float)
        conf[key] = np.round(m / np.maximum(m.sum(axis=1, keepdims=True), 1), 5).tolist()
    report["confusion"] = conf
    report["new_top_confusions"] = [
        {"true": t, "pred": p, "count": int(n)}
        for (t, p), n in Counter(zip(uniform["primary"], uniform["pred_new"])).most_common(40)
        if t != p
    ][:15]
    (OUT / f"labeller_validation{args.tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [c for c in LABELS if per_class[c]["support_stratified"]]
    idx = [LABELS.index(c) for c in rows]
    fig, axes = plt.subplots(1, 2, figsize=(22, 9.5))
    for ax, col, name, summ in ((axes[0], "pred", "Old: _classify_tactic (first move)", old),
                                (axes[1], "pred_new", "New: tactic_tagger (whole line)", new)):
        cm = confusion_matrix(strat["primary"], strat[col], labels=LABELS).astype(float)[idx]
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(LABELS)), LABELS, rotation=75, ha="right", fontsize=8)
        ax.set_yticks(range(len(rows)), rows, fontsize=8)
        ax.set_xlabel("predicted label")
        ax.set_ylabel("Lichess primary category")
        ax.set_title(f"{name}\nstrict {summ['strict_agreement']:.1%} · κ {summ['cohens_kappa']:.2f} · "
                     f"macro recall {summ['macro_recall_stratified']:.1%}", fontsize=11)
        for i in range(len(rows)):
            for j in range(len(LABELS)):
                if cm[i, j] >= 0.05:
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if cm[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=axes, fraction=0.015)
    fig.suptitle("Tactic labeller vs Lichess puzzle themes (row-normalised, stratified sample)", fontsize=13)
    fig.savefig(OUT / f"labeller_confusion{args.tag}.png", dpi=130, bbox_inches="tight")

    print(json.dumps(report["comparison"], indent=2))
    print("\nper-class recall (stratified) / precision (uniform):   old  ->  new")
    for c in LABELS:
        o, n = old["per_class"][c], new["per_class"][c]
        print(f"  {c:18s} R {o['recall']} -> {n['recall']}   P {o['precision']} -> {n['precision']}")


if __name__ == "__main__":
    main()
