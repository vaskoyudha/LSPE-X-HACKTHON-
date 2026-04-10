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
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import PROCESSED_DIR
from src.outcomes import (
    REVIEWED_LABELS_PATH,
    normalize_confidence_scores,
    validate_evidence_labels,
)

DEST_PATH = PROCESSED_DIR / "review_benchmark_500_reviewed.csv"
REVIEW_BASE_PATH = PROCESSED_DIR / "review_benchmark_500.csv"
LABEL_MAP = {0: "low", 1: "medium", 2: "high"}


def _optional_series(df: pd.DataFrame, column: str, default: object = "") -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def transform_reviewed_rows(
    imported: pd.DataFrame,
    review_base: pd.DataFrame,
    *,
    source_name: str,
) -> pd.DataFrame:
    base_cols = (
        review_base[["source_row_idx", "ocid"]]
        .dropna(subset=["source_row_idx", "ocid"])
        .drop_duplicates(subset=["source_row_idx"], keep="first")
    )
    merged = imported.merge(base_cols, on="source_row_idx", how="left", validate="m:1")
    if merged["ocid"].isna().any():
        unresolved = merged.loc[merged["ocid"].isna(), "source_row_idx"].astype(int).tolist()
        raise ValueError(f"Could not resolve ocid for source_row_idx values: {unresolved}")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    normalized = pd.DataFrame(
        {
            "ocid": merged["ocid"],
            "label_family": "reviewed_risk",
            "label_value": merged["reviewed_label"].map(LABEL_MAP),
            "source_name": source_name,
            "source_type": "human_review",
            "source_record_id": merged["source_row_idx"].astype(int).astype(str),
            "decision_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "confidence_score": normalize_confidence_scores(
                _optional_series(merged, "review_confidence", 1.0)
            ),
            "review_notes": _optional_series(merged, "review_notes", "").fillna(""),
            "reviewer_id": _optional_series(merged, "reviewer_id", "").fillna(""),
            "ingested_at": now,
        }
    )
    return validate_evidence_labels(normalized)


def main(source: Path, *, source_name: str = "manual_review_import") -> None:
    if not source.exists():
        raise FileNotFoundError(f"Reviewed row-level file not found: {source}")
    if not REVIEW_BASE_PATH.exists():
        raise FileNotFoundError(
            f"Base review benchmark not found for OCID resolution: {REVIEW_BASE_PATH}"
        )

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
    df["source_row_idx"] = df["source_row_idx"].astype(int)
    df["reviewed_label"] = df["reviewed_label"].astype(int)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    review_base = pd.read_csv(REVIEW_BASE_PATH)
    canonical = transform_reviewed_rows(df, review_base, source_name=source_name)
    canonical.to_parquet(REVIEWED_LABELS_PATH, index=False)
    df.to_csv(DEST_PATH, index=False, encoding="utf-8-sig")
    print(f"Imported row-level reviewed benchmark to {DEST_PATH} ({len(df)} rows)")
    print(
        f"Wrote canonical reviewed labels to {REVIEWED_LABELS_PATH} ({len(canonical)} rows)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--source-name", default="manual_review_import")
    args = parser.parse_args()
    main(Path(args.source), source_name=args.source_name)
