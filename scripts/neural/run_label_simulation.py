"""
Downstream effect of the labeller: simulation E3 rerun with PuzzleNet's labels.

E3 (scripts/research/run_simulation.py) asks how much label quality matters for
early weakness targeting: game evidence reaches the recommender through the
labeller, and every mislabelled critical position is credited to the wrong category.
The label-noise model draws each observed label from the labeller's measured
P(predicted | true), estimated on the stratified harness sample. This script reruns
E3 on the same seeds (common random numbers, so every comparison is paired) with
four label sources:

    perfect     labels equal the true category
    puzzlenet   PuzzleNet's harness confusion matrix
    rules       the rule-based tagger's harness confusion matrix (E3's "new")
    none        no game evidence at all

Reads eval/neural/label_confusion.json (written by evaluate_puzzlenet.py).
Output: eval/neural/label_simulation.json

Usage: python -m scripts.neural.run_label_simulation [--runs 200] [--workers 10]
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse          # noqa: E402
import json              # noqa: E402
import sys               # noqa: E402
import time              # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import scripts.research.run_simulation as RS  # noqa: E402
from src.data.puzzle_loader import WEAKNESS_CATEGORIES  # noqa: E402
from src.evaluation import simulation as sim  # noqa: E402

CONFUSION = Path("eval/neural/label_confusion.json")
OUT = Path("eval/neural/label_simulation.json")


def _init(pool_path: str, conf_path: str) -> None:
    data = np.load(pool_path)
    RS._POOL = sim.PuzzlePool({c: data[c] for c in WEAKNESS_CATEGORIES if c in data.files})
    conf = json.loads(Path(conf_path).read_text("utf-8"))["confusion"]
    RS._CONF = {"labels": conf["labels"], "puzzlenet": np.array(conf["puzzlenet"]),
                "rules": np.array(conf["rules"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=20)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(0)
    counts = json.loads((RS.OUT / "sim_category_counts.json").read_text("utf-8"))
    weights = [max(1, counts.get(c, 1)) for c in WEAKNESS_CATEGORIES]
    labels_conf = json.loads(CONFUSION.read_text("utf-8"))["confusion"]["labels"]
    assert labels_conf == WEAKNESS_CATEGORIES + ["General"], "label order differs from the simulator"

    results, by_label = {"runs": args.runs, "design": "paired by seed (common random numbers)",
                         "T": 60, "labels": {}}, {}
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(str(RS.POOL_CACHE), str(CONFUSION))) as ex:
        for lab in ("perfect", "puzzlenet", "rules", "none"):
            cfg = {"T": 60, "labels": "rules" if lab == "none" else lab, "weights": weights,
                   "learn": {"model": "none"},
                   "pop": {"opportunities_mean": 0.0} if lab == "none" else {}}
            summ = RS.run(ex, "episode", ["Beta-TS", "IRT-TS"], cfg, args.runs, args.chunk)
            by_label[lab] = summ
            results["labels"][lab] = RS.table(summ, ["targeting_first30", "top3_recall_final",
                                                     "cum_regret"])
            print(f"  {lab:10s} done {time.time() - t0:.0f}s", flush=True)
    results["paired_targeting_first30"] = {
        pol: {f"{a}_vs_{b}": RS.paired(by_label[a][pol]["targeting_first30"],
                                       by_label[b][pol]["targeting_first30"], rng)
              for a, b in (("puzzlenet", "rules"), ("perfect", "puzzlenet"),
                           ("perfect", "rules"), ("puzzlenet", "none"))}
        for pol in ("Beta-TS", "IRT-TS")
    }
    gap = {}
    for pol in ("Beta-TS", "IRT-TS"):
        m = {lab: results["labels"][lab][pol]["targeting_first30"]["mean"] for lab in by_label}
        denom = m["perfect"] - m["rules"]
        gap[pol] = round((m["puzzlenet"] - m["rules"]) / denom, 3) if denom > 0 else None
    results["share_of_rules_to_perfect_gap_closed"] = gap
    results["seconds"] = round(time.time() - t0)
    OUT.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"labels": {k: {p: v[p]["targeting_first30"] for p in v}
                                 for k, v in results["labels"].items()},
                      "gap_closed": gap}, indent=1))


if __name__ == "__main__":
    main()
