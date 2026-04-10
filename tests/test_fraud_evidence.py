from __future__ import annotations

import pandas as pd
import pytest

from src.fraud_evidence import (
    build_fraud_evidence_dataset,
    train_fraud_evidence_model,
)


@pytest.mark.p1
def test_build_fraud_evidence_dataset_marks_positive_and_unlabeled_rows():
    raw = pd.DataFrame(
        {
            "ocid": ["ocds-1", "ocds-2"],
            "buyer_id": ["buyer-a", "buyer-b"],
            "supplier_id": ["sup-1", "sup-2"],
            "tender_datePublished": pd.to_datetime(
                ["2023-01-01", "2023-02-01"], utc=True
            ),
            "award_value_amount": [100.0, 200.0],
        }
    )
    base_features = pd.DataFrame({"f_tender_value_log": [1.0, 2.0]})
    evidence = pd.DataFrame(
        {
            "ocid": ["ocds-2"],
            "label_family": ["confirmed_fraud"],
            "label_value": ["fraud"],
            "evidence_strength": [1.0],
        }
    )

    dataset = build_fraud_evidence_dataset(raw, base_features, evidence)

    assert dataset["fraud_evidence_target"].tolist() == [0, 1]
    assert dataset["is_unlabeled"].tolist() == [1, 0]
    assert dataset["sample_weight"].tolist() == [0.2, 1.0]


@pytest.mark.p1
def test_build_fraud_evidence_dataset_prefers_confirmed_positive_over_reviewed_risk():
    raw = pd.DataFrame(
        {
            "ocid": ["ocds-1"],
            "buyer_id": ["buyer-a"],
            "supplier_id": ["sup-1"],
            "tender_datePublished": pd.to_datetime(["2023-01-01"], utc=True),
            "award_value_amount": [100.0],
        }
    )
    base_features = pd.DataFrame({"f_tender_value_log": [1.0]})
    evidence = pd.DataFrame(
        {
            "ocid": ["ocds-1", "ocds-1"],
            "label_family": ["reviewed_risk", "confirmed_fraud"],
            "label_value": ["high", "fraud"],
            "evidence_strength": [0.5, 1.0],
        }
    )

    dataset = build_fraud_evidence_dataset(raw, base_features, evidence)

    assert dataset["label_family"].tolist() == ["confirmed_fraud"]
    assert dataset["fraud_evidence_target"].tolist() == [1]
    assert dataset["sample_weight"].tolist() == [1.0]


@pytest.mark.p1
def test_build_fraud_evidence_dataset_prefers_positive_evidence_for_same_ocid():
    raw = pd.DataFrame(
        {
            "ocid": ["ocds-1"],
            "buyer_id": ["buyer-a"],
            "supplier_id": ["sup-1"],
            "tender_datePublished": pd.to_datetime(["2023-01-01"], utc=True),
            "award_value_amount": [100.0],
        }
    )
    base_features = pd.DataFrame({"f_tender_value_log": [1.0]})
    evidence = pd.DataFrame(
        {
            "ocid": ["ocds-1", "ocds-1"],
            "label_family": ["reviewed_risk", "confirmed_irregularity"],
            "label_value": ["high", "irregularity"],
        }
    )

    dataset = build_fraud_evidence_dataset(raw, base_features, evidence)

    assert dataset["fraud_evidence_target"].tolist() == [1]
    assert dataset["is_unlabeled"].tolist() == [0]
    assert dataset["sample_weight"].tolist() == [0.8]
    assert "f_tender_value_log" in dataset.columns
    assert "g_buyer_supplier_prev_contract_count" in dataset.columns


@pytest.mark.p1
def test_train_fraud_evidence_model_returns_binary_metrics():
    dataset = pd.DataFrame(
        {
            "f_tender_value_log": [1.0, 2.0, 3.0, 4.0],
            "g_buyer_supplier_prev_contract_count": [0, 0, 1, 2],
            "fraud_evidence_target": [0, 0, 1, 1],
            "sample_weight": [0.2, 0.2, 1.0, 1.0],
        }
    )
    model, metrics = train_fraud_evidence_model(dataset)

    assert hasattr(model, "predict_proba")
    assert metrics["label_type"] == "fraud_evidence_positive_unlabeled"
    assert "precision_at_10pct" in metrics
    assert "average_precision" in metrics
