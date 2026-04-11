"""Import official/public evidence records into normalized evidence + label tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence import evidence_to_label_record, normalize_evidence_record


DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "evidence"


def _load_input(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return [payload]
        return list(payload)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    raise ValueError(f"Unsupported input format: {path.suffix}")



def import_official_evidence(input_path: Path, output_dir: Path = DEFAULT_OUT_DIR) -> dict[str, Path]:
    records = _load_input(input_path)
    evidence_rows = [normalize_evidence_record(record) for record in records]
    label_rows = [
        evidence_to_label_record(
            evidence,
            ocid=evidence.get("matched_ocid"),
            confidence_score=evidence.get("match_confidence"),
            reviewer_needed=evidence.get("matched_ocid") is None,
        )
        for evidence in evidence_rows
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "evidence_records.parquet"
    label_path = output_dir / "label_records.parquet"
    pd.DataFrame(evidence_rows).to_parquet(evidence_path, index=False, engine="pyarrow")
    pd.DataFrame(label_rows).to_parquet(label_path, index=False, engine="pyarrow")
    return {"evidence_records": evidence_path, "label_records": label_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    outputs = import_official_evidence(args.input_path, output_dir=args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
