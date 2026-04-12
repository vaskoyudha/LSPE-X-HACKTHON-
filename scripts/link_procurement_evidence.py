"""Link normalized official evidence rows back to procurement rows and refresh label records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import FLAT_PARQUET
from src.evidence import evidence_to_label_record
from src.evidence_linking import apply_match_results_to_label_records, build_evidence_match_table

DEFAULT_EVIDENCE_DIR = ROOT / "data" / "processed" / "evidence"
DEFAULT_EVIDENCE_PATH = DEFAULT_EVIDENCE_DIR / "evidence_records.parquet"
DEFAULT_LABEL_PATH = DEFAULT_EVIDENCE_DIR / "label_records.parquet"
DEFAULT_OUTPUT_DIR = DEFAULT_EVIDENCE_DIR

PROCUREMENT_COLUMNS = [
    "ocid",
    "tender_title",
    "tender_datePublished",
    "buyer_name",
    "buyer_id",
    "supplier_name",
    "supplier_id",
    "tender_value_amount",
    "award_value_amount",
]


def link_procurement_evidence(
    *,
    procurement_path: Path = FLAT_PARQUET,
    evidence_path: Path = DEFAULT_EVIDENCE_PATH,
    label_path: Path | None = DEFAULT_LABEL_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    min_confidence: float = 0.55,
) -> dict[str, Path]:
    procurement_df = pd.read_parquet(procurement_path, columns=PROCUREMENT_COLUMNS)
    evidence_df = pd.read_parquet(evidence_path)

    matches_df = build_evidence_match_table(procurement_df, evidence_df, min_confidence=min_confidence)

    output_dir.mkdir(parents=True, exist_ok=True)
    matches_path = output_dir / "evidence_matches.parquet"
    matches_df.to_parquet(matches_path, index=False, engine="pyarrow")

    outputs: dict[str, Path] = {"evidence_matches": matches_path}
    if label_path is not None and label_path.exists():
        label_df = pd.read_parquet(label_path)
    else:
        label_df = pd.DataFrame(
            [
                evidence_to_label_record(
                    record,
                    ocid=record.get("matched_ocid"),
                    confidence_score=record.get("match_confidence"),
                    reviewer_needed=record.get("matched_ocid") is None,
                )
                for record in evidence_df.to_dict(orient="records")
            ]
        )

    linked_label_df = apply_match_results_to_label_records(label_df, matches_df)
    linked_label_path = output_dir / "linked_label_records.parquet"
    linked_label_df.to_parquet(linked_label_path, index=False, engine="pyarrow")
    outputs["linked_label_records"] = linked_label_path

    summary = {
        "total_evidence_rows": int(len(evidence_df)),
        "matched_rows": int(matches_df["ocid"].notna().sum()) if not matches_df.empty else 0,
        "needs_review_rows": int(matches_df["reviewer_needed"].fillna(True).sum()) if not matches_df.empty else 0,
        "min_confidence": float(min_confidence),
    }
    summary_path = output_dir / "evidence_match_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    outputs["summary"] = summary_path
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--procurement-path", type=Path, default=FLAT_PARQUET)
    parser.add_argument("--evidence-path", type=Path, default=DEFAULT_EVIDENCE_PATH)
    parser.add_argument("--label-path", type=Path, default=DEFAULT_LABEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    args = parser.parse_args()
    outputs = link_procurement_evidence(
        procurement_path=args.procurement_path,
        evidence_path=args.evidence_path,
        label_path=args.label_path,
        output_dir=args.output_dir,
        min_confidence=args.min_confidence,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
