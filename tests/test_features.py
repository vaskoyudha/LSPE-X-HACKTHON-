"""Tests for src.features module — Tier 1 and Tier 2 feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    tier1_features,
    tier2_features,
    compute_all_features,
)


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Realistic raw procurement data for feature testing."""
    n = 20
    dates = pd.date_range("2019-01-01", periods=n, freq="2W", tz="UTC")
    rng = np.random.default_rng(42)

    return pd.DataFrame(
        {
            "ocid": [f"ocds-test-{i}" for i in range(n)],
            "tender_datePublished": dates,
            "tender_tenderPeriod_startDate": dates - pd.Timedelta(days=7),
            "tender_tenderPeriod_endDate": dates + pd.Timedelta(days=14),
            "award_date": dates + pd.Timedelta(days=30),
            "contract_dateSigned": dates + pd.Timedelta(days=45),
            "tender_value_amount": rng.uniform(1e8, 1e9, size=n),
            "award_value_amount": rng.uniform(0.7e8, 1e9, size=n),
            "contract_value_amount": rng.uniform(0.7e8, 1e9, size=n),
            "tender_numberOfTenderers": rng.choice([1, 2, 3, 5, 8, None], size=n),
            "tender_title": [
                f"Pengadaan {'A' * rng.integers(5, 60)} {i}" for i in range(n)
            ],
            "tender_description": [
                f"Deskripsi pengadaan {'B' * rng.integers(10, 100)} {i}"
                for i in range(n)
            ],
            "tender_procurementMethod": rng.choice(
                ["open", "direct", "limited", "selective"], size=n
            ),
            "buyer_id": rng.choice(["buyer-A", "buyer-B", "buyer-C"], size=n),
            "buyer_name": rng.choice(["Dinas A", "Dinas B", "Dinas C"], size=n),
            "supplier_id": rng.choice(
                ["sup-1", "sup-2", "sup-3", "sup-4"], size=n
            ),
            "supplier_name": rng.choice(
                ["PT Alpha", "PT Beta", "PT Gamma", "PT Delta"], size=n
            ),
        }
    )


@pytest.mark.p0
class TestTier1Features:
    def test_returns_15_columns(self, sample_raw_df):
        feats = tier1_features(sample_raw_df)
        assert len(feats.columns) == 15

    def test_all_numeric(self, sample_raw_df):
        feats = tier1_features(sample_raw_df)
        for col in feats.columns:
            assert pd.api.types.is_numeric_dtype(feats[col]), (
                f"Tier 1 feature '{col}' is not numeric"
            )

    def test_correct_row_count(self, sample_raw_df):
        feats = tier1_features(sample_raw_df)
        assert len(feats) == len(sample_raw_df)

    def test_log_values_non_negative(self, sample_raw_df):
        feats = tier1_features(sample_raw_df)
        for col in ["f_tender_value_log", "f_award_value_log", "f_contract_value_log"]:
            valid = feats[col].dropna()
            assert (valid >= 0).all(), f"{col} has negative values after log1p"

    def test_single_bidder_binary(self, sample_raw_df):
        feats = tier1_features(sample_raw_df)
        valid = feats["f_single_bidder"].dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})

    def test_procurement_method_encoded(self, sample_raw_df):
        feats = tier1_features(sample_raw_df)
        valid = feats["f_procurement_method_enc"].dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0, 2.0})


@pytest.mark.p0
class TestTier2Features:
    def test_returns_15_columns(self, sample_raw_df):
        feats = tier2_features(sample_raw_df)
        assert len(feats.columns) == 15

    def test_all_numeric(self, sample_raw_df):
        feats = tier2_features(sample_raw_df)
        for col in feats.columns:
            assert pd.api.types.is_numeric_dtype(feats[col]), (
                f"Tier 2 feature '{col}' is not numeric"
            )

    def test_no_look_ahead(self, sample_raw_df):
        """First row should have NaN for all history-based features
        since there's no past data to look at."""
        # Sort by date to identify the earliest row
        sorted_df = sample_raw_df.sort_values("tender_datePublished").reset_index(drop=True)
        feats = tier2_features(sorted_df)
        # History-based features should be NaN for the first row
        first_row = feats.iloc[0]
        history_cols = [
            "f_buyer_hist_avg_value",
            "f_supplier_hist_win_count",
            "f_tender_value_zscore_buyer",
        ]
        for col in history_cols:
            assert pd.isna(first_row[col]) or first_row[col] == 0, (
                f"First row has non-null history feature '{col}' = {first_row[col]}"
            )


@pytest.mark.p0
class TestCombinedFeatures:
    def test_30_total_features(self, sample_raw_df):
        feats = compute_all_features(sample_raw_df)
        assert len(feats.columns) == 30

    def test_all_numeric_onnx_safe(self, sample_raw_df):
        feats = compute_all_features(sample_raw_df)
        for col in feats.columns:
            assert pd.api.types.is_numeric_dtype(feats[col]), (
                f"Feature '{col}' is not ONNX-safe numeric"
            )

    def test_feature_names_prefixed(self, sample_raw_df):
        feats = compute_all_features(sample_raw_df)
        for col in feats.columns:
            assert col.startswith("f_"), f"Feature '{col}' not prefixed with 'f_'"
