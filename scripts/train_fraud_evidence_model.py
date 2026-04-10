"""Train the separate fraud-evidence model and write its artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fraud_evidence import train_fraud_evidence_model

DATASET_PATH = ROOT / "models" / "fraud_evidence_dataset.parquet"
MODEL_PATH = ROOT / "models" / "fraud_evidence_model.ubj"
METRICS_PATH = ROOT / "models" / "fraud_evidence_metrics.json"
CALIBRATION_PATH = ROOT / "models" / "fraud_evidence_calibration.json"


def main() -> None:
    dataset = pd.read_parquet(DATASET_PATH)
    model, metrics = train_fraud_evidence_model(dataset)
    model.save_model(MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    CALIBRATION_PATH.write_text(
        json.dumps(
            {"enabled": False, "note": "calibration pending dedicated holdout"},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {MODEL_PATH}")
    print(f"Wrote {METRICS_PATH}")


if __name__ == "__main__":
    main()
