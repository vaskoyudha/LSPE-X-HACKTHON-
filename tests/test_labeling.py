"""Tests for src.labels module — heuristic red-flag labeling."""

import numpy as np
import pandas as pd
import pytest

from src.labels import (
    CALIBRATION_SOURCE_INDEX_COL,
    flag_single_bidder,
    flag_short_title,
    flag_short_description,
    flag_q4_timing,
    flag_price_deviation,
    flag_high_value,
    flag_repeat_pair_history,
    flag_supplier_recent_surge,
    flag_buyer_value_spike,
    flag_direct_procurement,
    compute_red_flags,
    compute_risk_labels,
    RED_FLAG_FUNCTIONS,
    select_calibration_samples,
)


@pytest.fixture
def sample_procurement_df() -> pd.DataFrame:
    """Realistic procurement data for label testing."""
    return pd.DataFrame(
        {
            "ocid": [f"ocds-test-{i}" for i in range(6)],
            "tender_numberOfTenderers": [1, 5, 3, np.nan, 1, 10],
            "tender_title": [
                "Alkes",  # short (< 20 chars)
                "Pengadaan Alat Kesehatan Tahun 2020 untuk Puskesmas",
                "Test",  # short
                "Pengadaan Barang dan Jasa Konsultansi",
                "x",  # very short
                "Pengadaan Peralatan IT Dinas Kominfo Kab Bandung 2021",
            ],
            "tender_description": [
                "short desc",  # < 60 chars
                "Ini adalah deskripsi yang cukup panjang untuk pengadaan alat kesehatan di lingkungan puskesmas tahun 2020",
                "x",  # very short
                "Pengadaan jasa konsultansi untuk perencanaan pembangunan jalan nasional di wilayah Jawa Barat",
                "desc",  # short
                "Pengadaan peralatan teknologi informasi untuk mendukung digitalisasi pelayanan publik",
            ],
            "tender_datePublished": pd.to_datetime(
                [
                    "2020-11-15",  # Q4
                    "2020-03-20",  # Q1
                    "2020-10-01",  # Q4
                    "2020-06-15",  # Q2
                    "2020-12-28",  # Q4
                    "2020-07-10",  # Q3
                ],
                utc=True,
            ),
            "tender_value_amount": [100e6, 500e6, 200e6, 1e9, 50e6, 800e6],
            "award_value_amount": [100e6, 350e6, 200e6, 900e6, 50e6, 400e6],
            "tender_procurementMethod": [
                "direct",
                "open",
                "limited",
                "open",
                "direct",
                "open",
            ],
        }
    )


@pytest.mark.p0
class TestIndividualFlags:
    def test_single_bidder(self, sample_procurement_df):
        result = flag_single_bidder(sample_procurement_df)
        assert result.iloc[0] is np.True_   # 1 tenderer
        assert result.iloc[1] is np.False_  # 5 tenderers

    def test_short_title(self, sample_procurement_df):
        result = flag_short_title(sample_procurement_df)
        assert result.iloc[0] is np.True_   # "Alkes" (5 chars)
        assert result.iloc[1] is np.False_  # Long title

    def test_short_description(self, sample_procurement_df):
        result = flag_short_description(sample_procurement_df)
        assert result.iloc[0] is np.True_   # "short desc"
        assert result.iloc[1] is np.False_  # Long description

    def test_q4_timing(self, sample_procurement_df):
        result = flag_q4_timing(sample_procurement_df)
        assert result.iloc[0] is np.True_   # November
        assert result.iloc[1] is np.False_  # March

    def test_price_deviation(self, sample_procurement_df):
        result = flag_price_deviation(sample_procurement_df)
        # Row 0: 100e6/100e6 = 1.0 (ceiling hit = flagged)
        assert result.iloc[0] is np.True_
        # Row 1: 350e6/500e6 = 0.7 (flagged as <= 0.7)
        assert result.iloc[1] is np.True_

    def test_direct_procurement(self, sample_procurement_df):
        result = flag_direct_procurement(sample_procurement_df)
        assert result.iloc[0] is np.True_   # "direct"
        assert result.iloc[1] is np.False_  # "open"
        assert result.iloc[2] is np.True_   # "limited"

    def test_repeat_pair_history(self):
        df = pd.DataFrame(
            {
                "f_buyer_supplier_repeat_count": [0, 1, 2, 3, np.nan],
            }
        )

        result = flag_repeat_pair_history(df)

        assert result.tolist() == [False, False, True, True, False]

    def test_supplier_recent_surge(self):
        df = pd.DataFrame(
            {
                "f_supplier_recent_90d_award_count": [0, 2, 3, 5, np.nan],
            }
        )

        result = flag_supplier_recent_surge(df)

        assert result.tolist() == [False, False, True, True, False]

    def test_buyer_value_spike(self):
        df = pd.DataFrame(
            {
                "f_tender_value_zscore_buyer": [0.5, 1.9, 2.0, 4.2, np.nan],
            }
        )

        result = flag_buyer_value_spike(df)

        assert result.tolist() == [False, False, True, True, False]


@pytest.mark.p0
class TestCompositeLabeling:
    def test_risk_label_values(self, sample_procurement_df):
        labels = compute_risk_labels(sample_procurement_df)
        assert set(labels["risk_label"].unique()).issubset({0, 1, 2})

    def test_flag_count_matches(self, sample_procurement_df):
        labels = compute_risk_labels(sample_procurement_df)
        flags = labels[[c for c in labels.columns if c.startswith("flag_") and c != "flag_count"]]
        expected_count = flags.sum(axis=1)
        pd.testing.assert_series_equal(
            labels["flag_count"], expected_count, check_names=False
        )

    def test_high_risk_threshold(self, sample_procurement_df):
        labels = compute_risk_labels(sample_procurement_df, high_min=3)
        high_risk = labels[labels["risk_label"] == 2]
        assert (high_risk["flag_count"] >= 3).all()

    def test_low_risk_threshold(self, sample_procurement_df):
        labels = compute_risk_labels(sample_procurement_df, low_max=0)
        low_risk = labels[labels["risk_label"] == 0]
        assert (low_risk["flag_count"] == 0).all()

    def test_all_rows_labeled(self, sample_procurement_df):
        labels = compute_risk_labels(sample_procurement_df)
        assert len(labels) == len(sample_procurement_df)
        assert not labels["risk_label"].isna().any()


@pytest.mark.p1
class TestRedFlagRegistry:
    def test_eight_flags_registered(self):
        assert len(RED_FLAG_FUNCTIONS) == 8

    def test_all_flags_return_boolean(self, sample_procurement_df):
        for name, func in RED_FLAG_FUNCTIONS.items():
            result = func(sample_procurement_df)
            assert result.dtype == bool, f"Flag '{name}' returned dtype={result.dtype}"


@pytest.mark.p1
class TestCalibrationSampling:
    def test_calibration_samples_keep_source_row_index(self, sample_procurement_df):
        labels = compute_risk_labels(sample_procurement_df)
        samples = select_calibration_samples(labels, sample_procurement_df, n_samples=4, seed=42)
        assert CALIBRATION_SOURCE_INDEX_COL in samples.columns
        assert samples[CALIBRATION_SOURCE_INDEX_COL].between(0, len(sample_procurement_df) - 1).all()
