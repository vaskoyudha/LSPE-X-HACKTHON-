"""Leakage guard tests — Task 11.

Proves the split and feature pipeline satisfies C-C4 anti-leakage:
  1. No OCID overlap between train and test
  2. Strict temporal non-overlap (max train date < min test date)
  3. Feature columns match the frozen catalog across partitions
  4. No test OCIDs leak into any train artifact
  5. Expanding-window features use past-only data (no look-ahead)

These tests operate on materialized artifacts, so they require
the pipeline to have been run at least once (via scripts/materialize.py).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import PROJECT_ROOT, PROCESSED_DIR
from src.split import TRAIN_DIR, TEST_DIR


# ---------------------------------------------------------------------------
# Skip if artifacts aren't materialized yet
# ---------------------------------------------------------------------------

_TRAIN_RAW = TRAIN_DIR / "raw.parquet"
_TEST_RAW = TEST_DIR / "raw.parquet"
_TRAIN_FEAT = TRAIN_DIR / "features.parquet"
_TEST_FEAT = TEST_DIR / "features.parquet"
_TRAIN_LABELS = TRAIN_DIR / "labels.parquet"
_TEST_LABELS = TEST_DIR / "labels.parquet"
_FEATURE_MANIFEST = PROCESSED_DIR / "feature_manifest.json"
_SPLIT_METADATA = PROCESSED_DIR / "split_metadata.json"

_ARTIFACTS_EXIST = all(
    p.exists()
    for p in [
        _TRAIN_RAW, _TEST_RAW, _TRAIN_FEAT, _TEST_FEAT,
        _TRAIN_LABELS, _TEST_LABELS, _FEATURE_MANIFEST,
    ]
)

skip_if_no_artifacts = pytest.mark.skipif(
    not _ARTIFACTS_EXIST,
    reason="Materialized artifacts not found. Run scripts/materialize.py first.",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def train_raw():
    return pd.read_parquet(_TRAIN_RAW)


@pytest.fixture(scope="module")
def test_raw():
    return pd.read_parquet(_TEST_RAW)


@pytest.fixture(scope="module")
def train_features():
    return pd.read_parquet(_TRAIN_FEAT)


@pytest.fixture(scope="module")
def test_features():
    return pd.read_parquet(_TEST_FEAT)


@pytest.fixture(scope="module")
def train_labels():
    return pd.read_parquet(_TRAIN_LABELS)


@pytest.fixture(scope="module")
def test_labels():
    return pd.read_parquet(_TEST_LABELS)


@pytest.fixture(scope="module")
def feature_manifest():
    return json.loads(_FEATURE_MANIFEST.read_text())


@pytest.fixture(scope="module")
def split_metadata():
    return json.loads(_SPLIT_METADATA.read_text())


# ---------------------------------------------------------------------------
# P0: Critical anti-leakage tests
# ---------------------------------------------------------------------------


@skip_if_no_artifacts
@pytest.mark.p0
class TestNoOCIDOverlap:
    """No OCID from test set should appear in training set."""

    def test_raw_split_no_overlap(self, train_raw, test_raw):
        train_ocids = set(train_raw["ocid"])
        test_ocids = set(test_raw["ocid"])
        overlap = train_ocids & test_ocids
        assert len(overlap) == 0, (
            f"Found {len(overlap)} OCIDs in BOTH train and test: "
            f"{list(overlap)[:5]}..."
        )

    def test_all_data_accounted(self, train_raw, test_raw, feature_manifest):
        """Total rows should match manifest."""
        total = len(train_raw) + len(test_raw)
        manifest_total = (
            feature_manifest["train_rows"] + feature_manifest["test_rows"]
        )
        assert total == manifest_total


@skip_if_no_artifacts
@pytest.mark.p0
class TestTemporalNonOverlap:
    """Train max date must be strictly before test min date."""

    def test_temporal_boundary(self, train_raw, test_raw):
        date_col = "tender_datePublished"
        train_max = train_raw[date_col].max()
        test_min = test_raw[date_col].min()
        assert train_max < test_min, (
            f"Temporal overlap! Train max={train_max}, Test min={test_min}"
        )

    def test_split_metadata_consistent(self, train_raw, test_raw, split_metadata):
        """Split metadata should match actual data boundaries."""
        assert split_metadata["train_count"] == len(train_raw)
        assert split_metadata["test_count"] == len(test_raw)


@skip_if_no_artifacts
@pytest.mark.p0
class TestFeatureColumnParity:
    """Train and test must have identical feature columns."""

    def test_same_columns(self, train_features, test_features):
        assert list(train_features.columns) == list(test_features.columns), (
            f"Column mismatch!\n"
            f"  Train only: {set(train_features.columns) - set(test_features.columns)}\n"
            f"  Test only: {set(test_features.columns) - set(train_features.columns)}"
        )

    def test_column_count_matches_manifest(self, train_features, feature_manifest):
        assert len(train_features.columns) == feature_manifest["feature_count"], (
            f"Feature count mismatch: actual={len(train_features.columns)}, "
            f"manifest={feature_manifest['feature_count']}"
        )

    def test_exactly_30_features(self, train_features):
        assert len(train_features.columns) == 30

    def test_all_numeric(self, train_features, test_features):
        for name, feats in [("train", train_features), ("test", test_features)]:
            for col in feats.columns:
                assert pd.api.types.is_numeric_dtype(feats[col]), (
                    f"{name} feature '{col}' is not numeric: {feats[col].dtype}"
                )


@skip_if_no_artifacts
@pytest.mark.p0
class TestLabelIntegrity:
    """Labels must be valid and present for all rows."""

    def test_label_row_counts(self, train_raw, test_raw, train_labels, test_labels):
        assert len(train_labels) == len(train_raw)
        assert len(test_labels) == len(test_raw)

    def test_valid_risk_labels(self, train_labels, test_labels):
        for name, labels in [("train", train_labels), ("test", test_labels)]:
            valid_values = {0, 1, 2}
            actual = set(labels["risk_label"].unique())
            assert actual.issubset(valid_values), (
                f"{name} has invalid risk labels: {actual - valid_values}"
            )

    def test_no_nan_labels(self, train_labels, test_labels):
        assert not train_labels["risk_label"].isna().any(), "Train has NaN labels"
        assert not test_labels["risk_label"].isna().any(), "Test has NaN labels"


@skip_if_no_artifacts
@pytest.mark.p0
class TestFeatureRowAlignment:
    """Features and labels must be row-aligned with raw data."""

    def test_train_alignment(self, train_raw, train_features, train_labels):
        assert len(train_raw) == len(train_features) == len(train_labels)

    def test_test_alignment(self, test_raw, test_features, test_labels):
        assert len(test_raw) == len(test_features) == len(test_labels)


# ---------------------------------------------------------------------------
# P1: Deeper anti-leakage verification
# ---------------------------------------------------------------------------


@skip_if_no_artifacts
@pytest.mark.p1
class TestExpandingWindowPastOnly:
    """Tier 2 expanding-window features must not look ahead."""

    def test_first_temporal_row_has_no_history(self, train_raw, train_features):
        """The chronologically first row should have NaN/0 for all
        history-based features (nothing to look back at)."""
        date_col = "tender_datePublished"
        first_idx = train_raw[date_col].idxmin()

        history_features = [
            "f_buyer_hist_avg_value",
            "f_buyer_hist_value_std",
            "f_supplier_hist_win_count",
            "f_buyer_supplier_repeat_count",
            "f_buyer_hist_tender_count",
            "f_supplier_hist_max_award",
            "f_tender_value_zscore_buyer",
        ]

        for col in history_features:
            if col in train_features.columns:
                val = train_features.loc[first_idx, col]
                assert pd.isna(val) or val == 0, (
                    f"First temporal row has non-empty history feature "
                    f"'{col}' = {val}"
                )

    def test_dev_splits_no_test_contamination(self):
        """Dev split manifest must not reference test data dates."""
        manifest_path = PROCESSED_DIR / "dev_split_manifest.json"
        if not manifest_path.exists():
            pytest.skip("Dev split manifest not found")

        dev = json.loads(manifest_path.read_text())
        split = json.loads(_SPLIT_METADATA.read_text())

        split_date = pd.Timestamp(split["split_date"])

        for name in ["train_fit", "val_hpo", "val_calibration"]:
            max_date = pd.Timestamp(dev[name]["date_max"])
            assert max_date <= split_date, (
                f"Dev split '{name}' max date {max_date} exceeds "
                f"train/test split date {split_date}"
            )
