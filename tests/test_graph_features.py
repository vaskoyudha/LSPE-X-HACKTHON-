from __future__ import annotations

import pandas as pd
import pytest

from src.graph_features import build_relationship_features


@pytest.mark.p1
def test_build_relationship_features_uses_only_past_rows():
    raw = pd.DataFrame(
        {
            "ocid": ["ocds-1", "ocds-2", "ocds-3"],
            "buyer_id": ["buyer-a", "buyer-a", "buyer-a"],
            "supplier_id": ["sup-1", "sup-1", "sup-2"],
            "tender_datePublished": pd.to_datetime(
                ["2023-01-01", "2023-02-01", "2023-03-01"], utc=True
            ),
            "award_value_amount": [100.0, 200.0, 300.0],
        }
    )

    feats = build_relationship_features(raw)

    assert feats["g_buyer_supplier_prev_contract_count"].tolist() == [0, 1, 0]
    assert feats["g_supplier_prev_buyer_count"].tolist() == [0, 1, 0]
    assert feats["g_buyer_prev_supplier_count"].tolist() == [0, 1, 1]
    assert feats["g_pair_prev_award_value_sum"].tolist() == [0.0, 100.0, 0.0]


@pytest.mark.p1
def test_build_relationship_features_restores_original_row_order():
    raw = pd.DataFrame(
        {
            "ocid": ["ocds-2", "ocds-1"],
            "buyer_id": ["buyer-a", "buyer-a"],
            "supplier_id": ["sup-1", "sup-1"],
            "tender_datePublished": pd.to_datetime(
                ["2023-02-01", "2023-01-01"], utc=True
            ),
            "award_value_amount": [200.0, 100.0],
        }
    )

    feats = build_relationship_features(raw)

    assert feats["g_buyer_supplier_prev_contract_count"].tolist() == [1, 0]
    assert feats["g_pair_prev_award_value_sum"].tolist() == [100.0, 0.0]
