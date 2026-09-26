"""
Encode every Lichess puzzle for PuzzleNet.

Reads data/processed/puzzles_full.parquet and writes data/neural/ (gitignored):
    X_bits.npy          (N, 413) uint8   bit-packed binary features
    X_cont.npy          (N, 27)  float16 continuous features
    meta.npz            targets, split, ids
    dataset_info.json   encoding version, label lists, split and class counts

Usage: python -m scripts.neural.build_dataset [--workers 14] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Encoding is pure Python; one BLAS thread per worker keeps 14 spawned processes
# from each starting a full OpenBLAS thread pool (which exhausts memory on Windows).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.neural import encoding as E                                 # noqa: E402
from src.data.puzzle_loader import resolve_primary_category          # noqa: E402
from src.neural.dataset import (CAT_INDEX, CATEGORIES, DATA_DIR, HARNESS,  # noqa: E402
                                HARNESS_SEEDS, N_THEME_BYTES, SPLIT_NAMES, THEMES,
                                hash_split, theme_vector)

PARQUET = Path("data/processed/puzzles_full.parquet")
N_FULL = 5_877_641        # harness samples are only defined on the full table
CHUNK = 10_000


def _encode_truncated(fen: str, mv: str, row: int) -> tuple[np.ndarray, np.ndarray]:
    """Encode a puzzle as game analysis would see it: only the first few plies of the
    line, plus the engine's mate verdict for the whole line.

    In deployment the labeller is handed Stockfish's principal variation cut to a
    fixed depth, not a puzzle solution that stops exactly where the tactic ends
    (scripts/neural/engine_pv_check.py measures what that costs). Training on cut
    lines as well as whole ones teaches the network to name the tactic from its
    opening moves. The mate flag is kept from the FULL line because the analyzer
    always has the engine's mate score, even when its line stops short.
    """
    import chess
    ms = mv.split()
    board = chess.Board(fen)
    board.push_uci(ms[0])
    line = [chess.Move.from_uci(u) for u in ms[1:]]

    played = board.copy(stack=False)
    for m in line:
        if m not in played.legal_moves:
            break
        played.push(m)
    mate = played.is_checkmate()

    solver_moves = (len(line) + 1) // 2
    k = 1 + (row * 2654435761) % 3            # 1, 2 or 3 solver moves, stable per row
    k = min(k, solver_moves)
    return E.encode_line(board, line[:2 * k - 1], mate=mate)


def _encode_chunk(args: tuple[int, list[str], list[str], bool]) -> tuple[int, np.ndarray, np.ndarray, int]:
    start, fens, moves, truncate = args
    n = len(fens)
    bits = np.zeros((n, E.N_BITS), dtype=np.uint8)
    cont = np.zeros((n, E.N_CONT), dtype=np.float32)
    failed = 0
    for i, (fen, mv) in enumerate(zip(fens, moves)):
        try:
            bits[i], cont[i] = (_encode_truncated(fen, mv, start + i) if truncate
                                else E.encode_puzzle(fen, mv))
        except Exception:
            failed += 1
    return start, np.packbits(bits, axis=1), cont.astype(np.float16), failed


def _create_npy(path: Path, dtype, shape: tuple[int, int]) -> int:
    """Write a .npy header for an array of `shape` and pre-size the file; returns the
    header length (the byte offset of row 0)."""
    header = {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
              "fortran_order": False, "shape": shape}
    with open(path, "wb") as f:
        np.lib.format.write_array_header_1_0(f, header)
        offset = f.tell()
        f.truncate(offset + int(np.prod(shape)) * np.dtype(dtype).itemsize)
    return offset


def harness_samples(primary: pd.Series, seed: int, *, uniform_n: int = 30000,
                    per_class: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Row positions of one labeller-validation sample: (uniform, stratified).

    Reproduces validate_labeller.load() on a single-column frame. DataFrame.sample
    draws row positions from (row count, seed) alone, and groupby().head() depends
    only on group membership and row order, so the same calls on a frame with the
    same rows in the same order pick the same puzzles, without a second copy of
    every FEN and move list in memory. (tests/test_neural_dataset.py checks this
    against load() itself.)
    """
    df = pd.DataFrame({"primary": np.asarray(primary)})
    uniform = df.sample(n=uniform_n, random_state=seed)
    strat = df.sample(frac=1.0, random_state=seed).groupby("primary").head(per_class)
    return uniform.index.to_numpy(), strat.index.to_numpy()


def harness_ids(puzzle_ids: pd.Series, primary: pd.Series, *,
                uniform_n: int = 30000, per_class: int = 2000) -> set[str]:
    """PuzzleIds of the held-out labeller-validation sample(s) (HARNESS_SEEDS)."""
    ids = np.asarray(puzzle_ids)
    held: set[str] = set()
    for seed in HARNESS_SEEDS:
        u, st = harness_samples(primary, seed, uniform_n=uniform_n, per_class=per_class)
        held |= set(ids[u]) | set(ids[st])
    return held

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=None, help="encode only the first N puzzles")
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    ap.add_argument("--truncate", action="store_true",
                    help="encode each line cut to 1-3 solver moves, keeping the mate verdict "
                         "(the deployment condition; see _encode_truncated)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Targets and splits first, from the small columns only (this machine has
    # ~4 GB free; two full copies of the table do not fit).
    small = pq.read_table(PARQUET, columns=["PuzzleId", "Rating", "RatingDeviation", "NbPlays",
                                            "Themes"]).to_pandas()
    if args.limit:
        small = small.iloc[:args.limit]
    n = len(small)
    print(f"{n:,} puzzles loaded in {time.time() - t0:.0f}s", flush=True)
    primary = small["Themes"].map(resolve_primary_category)
    held = harness_ids(small["PuzzleId"], primary) if n == N_FULL else set()
    split = np.array([HARNESS if pid in held else hash_split(pid) for pid in small["PuzzleId"]],
                     dtype=np.uint8)
    cat = np.array([CAT_INDEX[c] for c in primary], dtype=np.uint8)
    themes = np.zeros((n, N_THEME_BYTES), dtype=np.uint8)
    for s in range(0, n, 500_000):
        block = np.stack([theme_vector(t) for t in small["Themes"].iloc[s:s + 500_000]])
        themes[s:s + len(block)] = np.packbits(block, axis=1)
    rating = small["Rating"].to_numpy(np.int16)
    rd = small["RatingDeviation"].to_numpy(np.int16)
    nb_plays = small["NbPlays"].to_numpy(np.int32)
    puzzle_id = small["PuzzleId"].to_numpy(dtype="U8")
    del small, primary
    print(f"targets and splits ready at {time.time() - t0:.0f}s "
          f"({int((split == HARNESS).sum()):,} harness puzzles held out)", flush=True)

    # Stream FEN/Moves in record batches with a bounded number of chunks in flight,
    # and write results with plain file writes. Holding every FEN in the parent and
    # writing through a memory map made the working set grow to several GB, and on
    # this machine that paged so hard the workers ran ~9x slower than benchmarked.
    bits_path, cont_path = args.out / "X_bits.npy", args.out / "X_cont.npy"
    bits_hdr = _create_npy(bits_path, np.uint8, (n, E.N_PACKED))
    cont_hdr = _create_npy(cont_path, np.float16, (n, E.N_CONT))
    n_moves = np.zeros(n, dtype=np.uint8)
    failed = done = 0
    t_enc = time.time()
    batches = pq.ParquetFile(PARQUET).iter_batches(batch_size=CHUNK, columns=["FEN", "Moves"])
    with open(bits_path, "r+b") as fb, open(cont_path, "r+b") as fc,             ProcessPoolExecutor(max_workers=args.workers) as pool:

        def write(fut) -> None:
            nonlocal failed, done
            start, b, c, f = fut.result()
            fb.seek(bits_hdr + start * E.N_PACKED)
            fb.write(b.tobytes())
            fc.seek(cont_hdr + start * E.N_CONT * 2)
            fc.write(c.tobytes())
            failed += f
            before, done = done, done + len(b)
            if before // 250_000 != done // 250_000 or done == n:
                rate = done / (time.time() - t_enc)
                print(f"  {done:,}/{n:,} encoded ({rate:,.0f}/s)", flush=True)

        pending: set = set()
        start = 0
        for batch in batches:
            if start >= n:
                break
            fens = batch.column(0).to_pylist()[:n - start]
            moves = batch.column(1).to_pylist()[:n - start]
            n_moves[start:start + len(moves)] = [m.count(" ") + 1 for m in moves]
            pending.add(pool.submit(_encode_chunk, (start, fens, moves, args.truncate)))
            start += len(fens)
            if len(pending) >= 2 * args.workers:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    write(fut)
        for fut in as_completed(pending):
            write(fut)
    assert done == n, f"encoded {done} of {n}"

    np.savez(args.out / "meta.npz", cat=cat, themes=themes, rating=rating, rd=rd,
             nb_plays=nb_plays, n_moves=n_moves, split=split, puzzle_id=puzzle_id)
    counts = {SPLIT_NAMES[s]: int((split == s).sum()) for s in SPLIT_NAMES}
    train = split == 0
    info = {
        "encoding_version": E.ENCODING_VERSION, "truncated_lines": bool(args.truncate),
        "n_bits": E.N_BITS, "n_cont": E.N_CONT, "n_packed": E.N_PACKED,
        "categories": CATEGORIES, "themes": THEMES,
        "n_puzzles": n, "failed_encodings": failed, "splits": counts,
        "harness_seeds": list(HARNESS_SEEDS),
        "category_counts_train": {c: int((cat[train] == i).sum()) for i, c in enumerate(CATEGORIES)},
        "rating_mean_train": float(rating[train].mean()),
        "rating_sd_train": float(rating[train].std()),
        "build_seconds": round(time.time() - t0, 1),
    }
    (args.out / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k not in ("themes",)}, indent=2))


if __name__ == "__main__":
    main()
