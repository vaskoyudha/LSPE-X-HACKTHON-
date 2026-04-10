"""Import a row-level reviewed benchmark file into the canonical repo path.

Expected columns at minimum:
- source_row_idx
- reviewed_label

Recommended additional columns:
- review_confidence
- review_notes
- explanation_agrees
- explanation_clarity
- explanation_actionable
- explanation_notes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import PROCESSED_DIR

DEST_PATH = PROCESSED_DIR / "review_benchmark_500_reviewed.csv"


def main(source: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Reviewed row-level file not found: {source}")

    df = pd.read_csv(source)
    required = {"source_row_idx", "reviewed_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["source_row_idx"] = pd.to_numeric(df["source_row_idx"], errors="coerce")
    df["reviewed_label"] = pd.to_numeric(df["reviewed_label"], errors="coerce")
    df = df[
        df["source_row_idx"].notna()
        & df["reviewed_label"].isin([0, 1, 2])
    ].copy()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEST_PATH, index=False, encoding="utf-8-sig")
    print(f"Imported row-level reviewed benchmark to {DEST_PATH} ({len(df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    args = parser.parse_args()
    main(Path(args.source))
