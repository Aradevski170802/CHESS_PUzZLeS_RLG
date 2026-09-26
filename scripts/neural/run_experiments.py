"""
Every PuzzleNet training run reported in the dissertation, in order. Runs that
already have a model file are skipped, so the script can be resumed.

    puzzlenet        the production model: all ~5.5M training puzzles, exported to
                     src/data/models/puzzlenet.npz
    abl_*            ablations, all on the SAME 1M-puzzle training subset (seed 0)
                     full_1m      the reference: 512-256 trunk, all features, all heads
                     linear       no hidden layers (multinomial logistic regression plus
                                  a linear heteroscedastic regression)
                     wide         1024-512 trunk (does capacity matter at 1M?)
                     raw          boards and move blocks only
                     engineered   hand-built tactical features only
                     cat_only     category head only (is multi-task learning helping?)
                     rating_only  rating head only
                     beta0, beta1 plain Gaussian NLL and MSE-like beta-NLL
    lc_*             learning curve: 30k, 100k and 300k training puzzles, trained for
                     about the same number of steps

Usage: python -m scripts.neural.run_experiments [--only puzzlenet,abl_linear]
Logs:  data/neural/logs/<name>.log
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

MODELS = Path("data/neural/models")
LOGS = Path("data/neural/logs")
ONE_M = ["--train-size", "1000000"]

RUNS: list[tuple[str, list[str]]] = [
    ("puzzlenet", ["--epochs", "3", "--export"]),
    ("abl_full_1m", ONE_M),
    ("abl_linear", ONE_M + ["--hidden", ""]),
    ("abl_raw", ONE_M + ["--features", "raw"]),
    ("abl_engineered", ONE_M + ["--features", "engineered"]),
    ("abl_cat_only", ONE_M + ["--heads", "cat"]),
    ("abl_rating_only", ONE_M + ["--heads", "rating"]),
    ("abl_beta0", ONE_M + ["--beta", "0"]),
    ("abl_beta1", ONE_M + ["--beta", "1"]),
    ("abl_wide", ONE_M + ["--hidden", "1024,512"]),
    ("lc_30k", ["--train-size", "30000", "--epochs", "50"]),
    ("lc_100k", ["--train-size", "100000", "--epochs", "15"]),
    ("lc_300k", ["--train-size", "300000", "--epochs", "5"]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = {s for s in args.only.split(",") if s}
    LOGS.mkdir(parents=True, exist_ok=True)
    for name, extra in RUNS:
        if only and name not in only:
            continue
        if (MODELS / f"{name}.npz").exists():
            print(f"skip {name} (model exists)", flush=True)
            continue
        t0 = time.time()
        cmd = [sys.executable, "-m", "scripts.neural.train_puzzlenet", "--name", name, *extra]
        with open(LOGS / f"{name}.log", "w", encoding="utf-8") as log:
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        print(f"{name}: exit {code} after {time.time() - t0:.0f}s", flush=True)
        if code != 0:
            print(f"  see {LOGS / (name + '.log')}", flush=True)


if __name__ == "__main__":
    main()
