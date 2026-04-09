import pandas as pd
import pytest

from src.diagnostics import (
    PROXY_BROAD_FEATURES,
    PROXY_CORE_FEATURES,
    resolve_proxy_feature_sets,
    summarize_data_provenance,
)


@pytest.mark.p1
def test_resolve_proxy_feature_sets_removes_expected_columns():
    features = [
        "f_single_bidder",
        "f_num_tenderers",
        "f_title_length",
        "f_supplier_hist_avg_award",
    ]
    resolved = resolve_proxy_feature_sets(features)
    assert resolved["full"] == features
    assert "f_single_bidder" not in resolved["proxy_core_removed"]
    assert "f_supplier_hist_avg_award" in resolved["proxy_core_removed"]
    assert set(PROXY_CORE_FEATURES).issuperset({"f_single_bidder", "f_num_tenderers"})
    assert set(PROXY_BROAD_FEATURES).issuperset(set(PROXY_CORE_FEATURES))


@pytest.mark.p1
def test_summarize_data_provenance_flags_synthetic_prefixes():
    train_raw = pd.DataFrame(
        {
            "ocid": ["ocds-synth-000001", "ocds-synth-000002"],
            "tender_datePublished": pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True),
            "buyer_id": ["b1", "b2"],
            "supplier_id": ["s1", "s2"],
            "tender_procurementMethod": ["open", "direct"],
        }
    )
    test_raw = pd.DataFrame(
        {
            "ocid": ["ocds-synth-000003"],
            "tender_datePublished": pd.to_datetime(["2020-02-01"], utc=True),
            "buyer_id": ["b1"],
            "supplier_id": ["s3"],
            "tender_procurementMethod": ["limited"],
        }
    )
    summary = summarize_data_provenance(train_raw, test_raw)
    assert summary["all_ocids_use_synthetic_prefix"] is True
    assert summary["data_kind"] == "synthetic_structured_benchmark"
    assert summary["row_count_total"] == 3
