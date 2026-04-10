import pandas as pd
import pytest
import numpy as np
from pathlib import Path
from pathlib import Path

from src.diagnostics import (
    PROXY_BROAD_FEATURES,
    PROXY_CORE_FEATURES,
    resolve_proxy_feature_sets,
    summarize_feature_health,
    summarize_feature_health_overview,
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


@pytest.mark.p1
def test_feature_health_marks_constant_and_all_nan_features():
    df = pd.DataFrame(
        {
            "f_live": [1.0, 2.0, 3.0],
            "f_dead_nan": [np.nan, np.nan, np.nan],
            "f_dead_constant": [0.0, 0.0, 0.0],
        }
    )

    report = summarize_feature_health(df)

    assert report["f_dead_nan"]["all_nan"] is True
    assert report["f_dead_constant"]["constant"] is True
    assert report["f_live"]["constant"] is False
    assert report["f_live"]["missing_pct"] == 0.0


@pytest.mark.p1
def test_feature_health_overview_tracks_removed_dead_slots():
    report = {
        "f_live": {"all_nan": False, "constant": False},
        "f_dead_constant": {"all_nan": False, "constant": True},
    }

    overview = summarize_feature_health_overview(
        report,
        retired_features=["f_old_dead", "f_dead_constant"],
    )

    assert overview["active_dead_feature_count"] == 1
    assert overview["active_dead_features"] == ["f_dead_constant"]
    assert overview["retired_dead_features_present"] == ["f_dead_constant"]
    assert overview["retired_dead_features_removed"] == ["f_old_dead"]


@pytest.mark.p1
def test_real_calibration_artifacts_are_not_synthetic():
    processed = Path("data/processed")
    sheet = pd.read_csv(processed / "calibration_sheet_100.csv")
    clean = pd.read_csv(processed / "clean_labels_100.csv")
    assert float(sheet["ocid"].astype(str).str.startswith("ocds-synth-").mean()) == 0.0
    assert float(clean["ocid"].astype(str).str.startswith("ocds-synth-").mean()) == 0.0


@pytest.mark.p1
def test_real_calibration_artifacts_are_not_synthetic():
    sheet = pd.read_csv(Path("data/processed/calibration_sheet_100.csv"))
    clean = pd.read_csv(Path("data/processed/clean_labels_100.csv"))
    assert float(sheet["ocid"].astype(str).str.startswith("ocds-synth-").mean()) == 0.0
    assert float(clean["ocid"].astype(str).str.startswith("ocds-synth-").mean()) == 0.0
