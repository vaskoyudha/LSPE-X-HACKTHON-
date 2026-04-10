from __future__ import annotations

import pandas as pd
import pytest

from src.fraud_evidence import build_fraud_evidence_dataset


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
