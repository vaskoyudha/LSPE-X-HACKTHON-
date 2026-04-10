from __future__ import annotations

import pandas as pd
import pytest

from src.outcomes import (
    REQUIRED_EVIDENCE_COLUMNS,
    normalize_confirmed_outcome_rows,
    normalize_reviewed_evidence_rows,
    validate_evidence_labels,
)


def test_validate_evidence_labels_rejects_missing_columns():
    df = pd.DataFrame({"ocid": ["ocds-1"], "label_family": ["reviewed_risk"]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_evidence_labels(df)


def test_validate_evidence_labels_accepts_canonical_frame():
    df = pd.DataFrame(
        [
            {
                "ocid": "ocds-1",
                "label_family": "reviewed_risk",
                "label_value": "high",
                "source_name": "manual_review_round_1",
                "source_type": "human_review",
                "source_record_id": "row-1",
                "decision_date": "2026-04-10",
                "confidence_score": 0.9,
                "review_notes": "clear pricing anomaly",
                "reviewer_id": "rev-1",
                "ingested_at": "2026-04-10T12:00:00Z",
            }
        ]
    )
    normalized = validate_evidence_labels(df)
    assert list(normalized.columns) == REQUIRED_EVIDENCE_COLUMNS
    assert normalized.loc[0, "label_family"] == "reviewed_risk"
    assert normalized.loc[0, "label_value"] == "high"


def test_normalize_reviewed_evidence_rows_maps_numeric_labels_and_resolves_ocid():
    imported = pd.DataFrame(
        [
            {
                "source_row_idx": 7,
                "reviewed_label": 2,
                "review_confidence": 4,
                "review_notes": "suspicious",
                "reviewer_id": "rev-1",
            },
        ]
    )
    review_base = pd.DataFrame(
        [
            {"source_row_idx": 7, "ocid": "ocds-2024-demo"},
        ]
    )

    normalized = normalize_reviewed_evidence_rows(
        imported,
        review_base,
        source_name="manual_review_round_1",
        decision_date="2026-04-10",
        ingested_at="2026-04-10T12:00:00Z",
    )
    normalized = validate_evidence_labels(normalized)

    assert normalized.loc[0, "ocid"] == "ocds-2024-demo"
    assert normalized.loc[0, "label_family"] == "reviewed_risk"
    assert normalized.loc[0, "label_value"] == "high"
    assert normalized.loc[0, "confidence_score"] == pytest.approx(0.8)


def test_normalize_confirmed_outcome_rows_maps_fraud_and_irregularity_families():
    rows = pd.DataFrame(
        [
            {
                "ocid": "ocds-1",
                "outcome_label": "fraud",
                "source_record_id": "case-1",
                "decision_date": "2025-10-03",
            },
            {
                "ocid": "ocds-2",
                "outcome_label": "irregularity",
                "source_record_id": "case-2",
                "decision_date": "2025-10-04",
            },
        ]
    )
    normalized = normalize_confirmed_outcome_rows(
        rows,
        source_name="fixture_source",
        source_type="court",
        ingested_at="2026-04-10T12:00:00Z",
    )
    assert normalized["label_family"].tolist() == [
        "confirmed_fraud",
        "confirmed_irregularity",
    ]
    assert normalized["label_value"].tolist() == ["fraud", "irregularity"]
