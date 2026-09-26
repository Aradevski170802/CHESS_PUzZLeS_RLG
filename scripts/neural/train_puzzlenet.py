"""
Train PuzzleNet on the encoded Lichess puzzles (scripts/neural/build_dataset.py).

    python -m scripts.neural.train_puzzlenet --name puzzlenet --epochs 4 --export
    python -m scripts.neural.train_puzzlenet --name abl_linear --hidden "" --train-size 1000000

Writes data/neural/models/<name>.npz (float32) and eval/neural/training/<name>.json
(the configuration, the learning curve and the calibrated validation metrics).
--export also writes the float16 copy the app loads: src/data/models/puzzlenet.npz.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.neural import encoding as E                                        # noqa: E402
from src.neural.dataset import (CATEGORIES, DATA_DIR, THEMES, TRAIN, VAL,    # noqa: E402
                                BlockLoader, feature_columns, load_dataset)
from src.neural.metrics import (category_metrics, fit_temperature,           # noqa: E402
                                fit_variance_scale, rating_metrics)
from src.neural.network import (HEADS, AdamW, NetConfig, PuzzleNet,          # noqa: E402
                                _sigmoid, log_softmax, lr_schedule)

MODELS_DIR = DATA_DIR / "models"
HISTORY_DIR = Path("eval/neural/training")
EXPORT_PATH = Path("src/data/models/puzzlenet.npz")


def collect(net: PuzzleNet, loader: BlockLoader) -> dict[str, np.ndarray]:
    """Raw outputs and targets over every row of a loader."""
    zc, zt, zr, cat, theme, rating, rho2 = [], [], [], [], [], [], []
    for _, b in loader.all_batches():
        _, c, t, r = net.forward(b.x)
        zc.append(c); zt.append(t); zr.append(r)
        cat.append(b.cat); theme.append(b.theme); rating.append(b.rating); rho2.append(b.rho2)
    cat_ = lambda xs: np.concatenate(xs)   # noqa: E731
    return {"zc": cat_(zc), "zt": cat_(zt), "zr": cat_(zr), "cat": cat_(cat),
            "theme": cat_(theme), "rating": cat_(rating), "rho2": cat_(rho2)}


def evaluate(net: PuzzleNet, loader: BlockLoader, rating_sd: float, *,
             temperature: float = 1.0, var_scale: float = 1.0) -> dict:
    o = collect(net, loader)
    cfg = net.config
    out: dict = {}
    probs = np.exp(log_softmax(o["zc"] / temperature))
    out["cat"] = category_metrics(o["cat"], probs, cfg.n_cat)
    p = np.clip(_sigmoid(o["zt"]), 1e-7, 1 - 1e-7)
    out["theme_bce_sum"] = float(-(o["theme"] * np.log(p) + (1 - o["theme"]) * np.log(1 - p))
                                 .sum(axis=1).mean())
    var_m = var_scale * np.exp(np.clip(o["zr"][:, 1], -9.0, 4.0))
    r = rating_metrics(o["rating"] * rating_sd, o["zr"][:, 0] * rating_sd,
                       np.sqrt(var_m) * rating_sd, np.sqrt(o["rho2"]) * rating_sd)
    out["rating"] = r
    # Model-selection score: the unweighted sum of the three negative log-likelihoods
    # (rating NLL in standardised units, so the three are on comparable scales).
    v = var_m + o["rho2"]
    rating_nll_std = float(np.mean(0.5 * (np.log(v) + (o["rating"] - o["zr"][:, 0]) ** 2 / v)))
    score = 0.0
    if "cat" in cfg.heads:
        score += out["cat"]["nll"]
    if "theme" in cfg.heads:
        score += cfg.theme_weight * out["theme_bce_sum"]
    if "rating" in cfg.heads:
        score += rating_nll_std
    out["selection_score"] = score
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--hidden", default="512,256", help='comma-separated widths; "" = linear model')
    ap.add_argument("--features", default="all", choices=["all", "raw", "engineered"])
    ap.add_argument("--heads", default=",".join(HEADS))
    ap.add_argument("--train-size", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--theme-weight", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--val-size", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--augment-dir", type=Path, default=None,
                    help="a truncated-line build of the same puzzles (build_dataset --truncate)")
    ap.add_argument("--augment-prob", type=float, default=0.0,
                    help="probability that a training row is drawn from --augment-dir")
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--history-dir", type=Path, default=HISTORY_DIR)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    data = load_dataset(args.data_dir, mmap=True, with_ids=False)
    info = data.info
    assert info["encoding_version"] == E.ENCODING_VERSION, "dataset was built with another encoding"
    rating_mean, rating_sd = info["rating_mean_train"], info["rating_sd_train"]

    train_idx = data.indices(TRAIN)
    if args.train_size and args.train_size < len(train_idx):
        train_idx = rng.choice(train_idx, args.train_size, replace=False)
    val_all = data.indices(VAL)
    val_sub = np.sort(rng.choice(val_all, min(args.val_size, len(val_all)), replace=False))

    sample = np.sort(rng.choice(train_idx, min(200_000, len(train_idx)), replace=False))
    cont_s = np.asarray(data.cont[sample], dtype=np.float32)
    cont_mean = cont_s.mean(axis=0)
    cont_std = np.maximum(cont_s.std(axis=0), 1e-3)
    columns = feature_columns(args.features)

    hidden = tuple(int(h) for h in args.hidden.split(",") if h.strip())
    heads = tuple(h for h in args.heads.split(",") if h)
    cfg = NetConfig(n_in=len(columns), hidden=hidden, n_cat=len(CATEGORIES), n_theme=len(THEMES),
                    theme_weight=args.theme_weight, beta=args.beta, heads=heads)
    net = PuzzleNet(cfg, seed=args.seed)
    opt = AdamW(net.params, weight_decay=args.wd)

    common = dict(cont_mean=cont_mean, cont_std=cont_std, rating_mean=rating_mean,
                  rating_sd=rating_sd, columns=columns)
    aug = load_dataset(args.augment_dir, mmap=True, with_ids=False) if args.augment_dir else None
    if aug is not None:
        print(f"[{args.name}] mixing in truncated lines from {args.augment_dir} "
              f"with probability {args.augment_prob}", flush=True)
    loader = BlockLoader(data, train_idx, args.batch, seed=args.seed,
                         augment=aug, augment_prob=args.augment_prob, **common)
    val_loader = BlockLoader(data, val_sub, 8192, **common)
    steps_per_epoch = len(loader)
    total = int(round(args.epochs * steps_per_epoch))
    warmup = min(args.warmup, max(1, total // 10))
    print(f"[{args.name}] {len(train_idx):,} training puzzles, {net.n_parameters():,} parameters, "
          f"{total:,} steps ({args.epochs} epochs of {steps_per_epoch:,})", flush=True)

    q: queue.Queue = queue.Queue(maxsize=6)
    stop = threading.Event()

    def produce() -> None:
        produced = 0
        while produced < total and not stop.is_set():
            for b in loader.epoch():
                q.put(b)
                produced += 1
                if produced >= total or stop.is_set():
                    break
        q.put(None)

    threading.Thread(target=produce, daemon=True).start()

    history = {"train": [], "val": []}
    running: dict[str, float] = {}
    best_score, best_params, best_step = float("inf"), None, 0
    step = 0
    t_train = time.time()
    while True:
        b = q.get()
        if b is None:
            break
        losses, grads = net.loss_and_grads(b)
        lr = lr_schedule(step, total, args.lr, warmup)
        gnorm = opt.step(grads, lr)
        for k, v in losses.items():
            running[k] = 0.98 * running.get(k, v) + 0.02 * v
        step += 1
        if step % 100 == 0:
            history["train"].append({"step": step, "lr": lr, "grad_norm": round(gnorm, 4),
                                     **{k: round(v, 5) for k, v in running.items()}})
        if step % args.eval_every == 0 or step == total:
            ev = evaluate(net, val_loader, rating_sd)
            rec = {"step": step, "epoch": round(step / steps_per_epoch, 3),
                   "selection_score": round(ev["selection_score"], 5),
                   "cat_accuracy": round(ev["cat"]["accuracy"], 4),
                   "cat_kappa": round(ev["cat"]["cohens_kappa"], 4),
                   "cat_nll": round(ev["cat"]["nll"], 4),
                   "theme_bce_sum": round(ev["theme_bce_sum"], 4),
                   "rating_rmse": round(ev["rating"]["rmse"], 1),
                   "rating_nll": round(ev["rating"]["nll"], 4),
                   "seconds": round(time.time() - t_train)}
            history["val"].append(rec)
            if ev["selection_score"] < best_score:
                best_score, best_step = ev["selection_score"], step
                best_params = {k: v.copy() for k, v in net.params.items()}
            sps = step / (time.time() - t_train)
            print(f"  step {step:6d}/{total}  {sps:.1f} steps/s  train {running['total']:.4f}  "
                  f"val acc {rec['cat_accuracy']:.4f} κ {rec['cat_kappa']:.4f} "
                  f"rmse {rec['rating_rmse']:.0f} score {rec['selection_score']:.4f}"
                  f"{'  *' if best_step == step else ''}", flush=True)
    stop.set()
    train_seconds = time.time() - t_train

    net.params.update(best_params)
    # Calibrate on the full validation split.
    full_val = BlockLoader(data, val_all, 8192, **common)
    o = collect(net, full_val)
    net.temperature = fit_temperature(o["zc"], o["cat"]) if "cat" in heads else 1.0
    net.var_scale = fit_variance_scale(o["rating"], o["zr"][:, 0],
                                       np.exp(np.clip(o["zr"][:, 1], -9.0, 4.0)),
                                       o["rho2"]) if "rating" in heads else 1.0
    uncal = evaluate(net, full_val, rating_sd)
    cal = evaluate(net, full_val, rating_sd, temperature=net.temperature, var_scale=net.var_scale)

    net.meta = {
        "name": args.name, "encoding_version": E.ENCODING_VERSION, "features": args.features,
        "categories": CATEGORIES, "themes": THEMES,
        "cont_mean": cont_mean.tolist(), "cont_std": cont_std.tolist(),
        "rating_mean": rating_mean, "rating_sd": rating_sd,
        "train_size": int(len(train_idx)), "epochs": args.epochs, "steps": step,
        "best_step": best_step, "batch": args.batch, "lr": args.lr, "weight_decay": args.wd,
        "seed": args.seed, "train_seconds": round(train_seconds),
        "augment_dir": str(args.augment_dir) if args.augment_dir else None,
        "augment_prob": args.augment_prob,
        "n_parameters": net.n_parameters(),
    }
    args.models_dir.mkdir(parents=True, exist_ok=True)
    net.save(args.models_dir / f"{args.name}.npz")
    if args.export:
        net.save(EXPORT_PATH, half=True)
    args.history_dir.mkdir(parents=True, exist_ok=True)
    report = {"name": args.name, "config": json.loads(cfg.to_json()), "args": vars(args),
              "meta": {k: v for k, v in net.meta.items() if k not in ("cont_mean", "cont_std")},
              "temperature": net.temperature, "var_scale": net.var_scale,
              "val_uncalibrated": uncal, "val_calibrated": cal, "history": history,
              "wall_seconds": round(time.time() - t0)}
    for part in (report["val_uncalibrated"], report["val_calibrated"]):
        part.get("rating", {}).pop("ap_per_theme", None)
    (args.history_dir / f"{args.name}.json").write_text(json.dumps(report, indent=2, default=str),
                                                        encoding="utf-8")
    print(f"[{args.name}] done in {time.time() - t0:.0f}s. best step {best_step}, "
          f"T={net.temperature:.3f}, var scale={net.var_scale:.3f}")
    print(json.dumps({"cat": cal["cat"], "rating": cal["rating"]}, indent=2))


if __name__ == "__main__":
    main()
