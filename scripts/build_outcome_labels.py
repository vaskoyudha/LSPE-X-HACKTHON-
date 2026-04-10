"""Normalize confirmed-outcome labels into canonical evidence records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import PROCESSED_DIR
from src.outcomes import CONFIRMED_OUTCOMES_PATH, normalize_confirmed_outcome_rows


def main(source: Path, *, source_name: str, source_type: str) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Outcome label file not found: {source}")

    df = pd.read_csv(source)
    normalized = normalize_confirmed_outcome_rows(
        df,
        source_name=source_name,
        source_type=source_type,
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(CONFIRMED_OUTCOMES_PATH, index=False)
    print(f"Wrote confirmed outcomes to {CONFIRMED_OUTCOMES_PATH} ({len(normalized)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-type", required=True)
    args = parser.parse_args()
    main(Path(args.source), source_name=args.source_name, source_type=args.source_type)
