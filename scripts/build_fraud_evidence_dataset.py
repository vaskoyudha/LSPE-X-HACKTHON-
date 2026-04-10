"""Materialize the separate fraud-evidence dataset artifact."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diagnostics import load_canonical_reviewed_labels, load_confirmed_outcome_labels
from src.fraud_evidence import build_fraud_evidence_dataset

OUTPUT_PATH = ROOT / "models" / "fraud_evidence_dataset.parquet"


def main() -> None:
    train_raw = pd.read_parquet(ROOT / "train_data" / "raw.parquet").assign(
        split_source="train"
    )
    test_raw = pd.read_parquet(ROOT / "test_data" / "raw.parquet").assign(
        split_source="test"
    )
    train_features = pd.read_parquet(ROOT / "train_data" / "features.parquet")
    test_features = pd.read_parquet(ROOT / "test_data" / "features.parquet")

    raw = pd.concat([train_raw, test_raw], ignore_index=True)
    features = pd.concat([train_features, test_features], ignore_index=True)
    evidence = pd.concat(
        [load_canonical_reviewed_labels(), load_confirmed_outcome_labels()],
        ignore_index=True,
    )

    dataset = build_fraud_evidence_dataset(raw, features, evidence)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH} ({len(dataset)} rows)")


if __name__ == "__main__":
    main()
