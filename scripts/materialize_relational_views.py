"""Materialize graph/entity-ready relational tables from downloaded OCDS JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import RAW_DIR, extract_relational_tables, save_relational_tables


def _discover_paths(years: list[int] | None = None) -> list[Path]:
    if years:
        return [RAW_DIR / f"{year}.jsonl.gz" for year in years if (RAW_DIR / f"{year}.jsonl.gz").exists()]
    return sorted(RAW_DIR.glob("*.jsonl.gz"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="*", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    paths = _discover_paths(args.years)
    if not paths:
        raise SystemExit("No raw OCDS .jsonl.gz files found")

    relations = extract_relational_tables(paths)
    outputs = save_relational_tables(relations, output_dir=args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
