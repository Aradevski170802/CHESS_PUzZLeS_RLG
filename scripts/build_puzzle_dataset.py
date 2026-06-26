"""
One-shot script: reads the raw Lichess CSV and writes organised Parquet files.

Run from the project root:
    python scripts/build_puzzle_dataset.py

Outputs (written to data/processed/):
    puzzles_full.parquet          -- all ~6M puzzles, enriched + cleaned
    puzzles_by_category/          -- one Parquet per WEAKNESS_CATEGORY
    puzzles_by_tier/              -- one Parquet per DIFFICULTY_TIER
    category_summary.csv          -- puzzle counts per category
"""

import logging
import sys
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.data_cleaner import clean, report
from src.data.puzzle_loader import (
    DIFFICULTY_TIERS,
    WEAKNESS_CATEGORIES,
    load_puzzles,
    save_processed,
    split_by_category,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RAW_CSV = Path("DataSets/lichess_db_puzzle.csv")
OUT_DIR = Path("data/processed")


def main() -> None:
    t0 = time.time()

    # 1. Load full dataset (6M rows — takes ~30s)
    logger.info("Step 1/4  Loading raw CSV ...")
    df = load_puzzles(RAW_CSV)

    # 2. Clean
    logger.info("Step 2/4  Cleaning ...")
    df = clean(df)
    report(df)

    # 3. Save full enriched dataset
    logger.info("Step 3/4  Saving full dataset ...")
    save_processed(df, OUT_DIR / "puzzles_full.parquet")

    # 4a. Split by category
    logger.info("Step 4a/4  Splitting by category ...")
    cat_summary = split_by_category(df, OUT_DIR / "puzzles_by_category")

    # 4b. Split by difficulty tier
    logger.info("Step 4b/4  Splitting by difficulty tier ...")
    tier_dir = OUT_DIR / "puzzles_by_tier"
    tier_dir.mkdir(parents=True, exist_ok=True)
    for _, _, label in DIFFICULTY_TIERS:
        subset = df[df["DifficultyTier"] == label]
        safe = label.replace(" ", "_")
        subset.to_parquet(tier_dir / f"{safe}.parquet", index=False)
        logger.info("  %-15s  %d puzzles", label, len(subset))

    # 5. Write category summary CSV
    summary_df = pd.DataFrame(
        [(cat, count) for cat, count in cat_summary.items()],
        columns=["Category", "PuzzleCount"],
    ).sort_values("PuzzleCount", ascending=False)
    summary_df.to_csv(OUT_DIR / "category_summary.csv", index=False)
    logger.info("Saved category_summary.csv")

    elapsed = time.time() - t0
    logger.info("Done in %.1fs", elapsed)
    print("\n=== Category Summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
