"""Tests for src.split module — temporal splitting and dev sub-splits."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.split import (
    external_raw_split,
    save_raw_splits,
    load_raw_split,
    internal_dev_splits,
    save_dev_split_manifest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame with temporal procurement data."""
    dates = pd.date_range("2018-01-01", periods=100, freq="W", tz="UTC")
    return pd.DataFrame(
        {
            "ocid": [f"ocds-test-{i}" for i in range(100)],
            "tender_datePublished": dates,
            "tender_title": [f"Tender {i}" for i in range(100)],
            "tender_value_amount": range(100_000, 200_000, 1000),
        }
    )


# ---------------------------------------------------------------------------
# Task 5: External raw split
# ---------------------------------------------------------------------------


@pytest.mark.p0
class TestExternalRawSplit:
    def test_temporal_no_overlap(self, sample_df: pd.DataFrame):
        """Train max date must be strictly before test min date."""
        train, test = external_raw_split(sample_df, test_ratio=0.2)
        train_max = train["tender_datePublished"].max()
        test_min = test["tender_datePublished"].min()
        assert train_max < test_min

    def test_approximate_ratio(self, sample_df: pd.DataFrame):
        train, test = external_raw_split(sample_df, test_ratio=0.2)
        total = len(train) + len(test)
        test_frac = len(test) / total
        assert 0.15 <= test_frac <= 0.25  # approximate

    def test_explicit_split_date(self, sample_df: pd.DataFrame):
        train, test = external_raw_split(
            sample_df, split_date="2019-06-01T00:00:00+00:00"
        )
        assert train["tender_datePublished"].max() <= pd.Timestamp(
            "2019-06-01", tz="UTC"
        )
        assert test["tender_datePublished"].min() > pd.Timestamp(
            "2019-06-01", tz="UTC"
        )

    def test_no_data_lost(self, sample_df: pd.DataFrame):
        """All rows with valid dates should be accounted for."""
        train, test = external_raw_split(sample_df, test_ratio=0.2)
        assert len(train) + len(test) == len(sample_df)

    def test_outputs_are_sorted_by_date(self, sample_df: pd.DataFrame):
        shuffled = sample_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        train, test = external_raw_split(shuffled, test_ratio=0.2)
        assert train["tender_datePublished"].is_monotonic_increasing
        assert test["tender_datePublished"].is_monotonic_increasing

    def test_raises_on_missing_column(self, sample_df: pd.DataFrame):
        with pytest.raises(ValueError, match="not found"):
            external_raw_split(sample_df, date_col="nonexistent_col")


@pytest.mark.p1
class TestSaveLoadRawSplits:
    def test_save_and_load_roundtrip(self, sample_df: pd.DataFrame, tmp_path: Path, monkeypatch):
        """Saved splits can be loaded back identically."""
        import src.split as split_mod

        monkeypatch.setattr(split_mod, "TRAIN_DIR", tmp_path / "train_data")
        monkeypatch.setattr(split_mod, "TEST_DIR", tmp_path / "test_data")
        monkeypatch.setattr(split_mod, "PROCESSED_DIR", tmp_path / "processed")
        monkeypatch.setattr(split_mod, "SPLIT_METADATA", tmp_path / "processed" / "split_metadata.json")

        train, test = external_raw_split(sample_df, test_ratio=0.2)
        save_raw_splits(train, test)

        loaded_train = load_raw_split("train")
        loaded_test = load_raw_split("test")

        assert len(loaded_train) == len(train)
        assert len(loaded_test) == len(test)

    def test_metadata_written(self, sample_df: pd.DataFrame, tmp_path: Path, monkeypatch):
        import src.split as split_mod

        monkeypatch.setattr(split_mod, "TRAIN_DIR", tmp_path / "train_data")
        monkeypatch.setattr(split_mod, "TEST_DIR", tmp_path / "test_data")
        monkeypatch.setattr(split_mod, "PROCESSED_DIR", tmp_path / "processed")
        monkeypatch.setattr(split_mod, "SPLIT_METADATA", tmp_path / "processed" / "split_metadata.json")

        train, test = external_raw_split(sample_df, test_ratio=0.2)
        save_raw_splits(train, test)

        meta_path = tmp_path / "processed" / "split_metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert "split_date" in meta
        assert meta["train_count"] == len(train)
        assert meta["test_count"] == len(test)


# ---------------------------------------------------------------------------
# Task 6: Internal dev sub-splits
# ---------------------------------------------------------------------------


@pytest.mark.p0
class TestInternalDevSplits:
    def test_three_partitions(self, sample_df: pd.DataFrame):
        # Use only "train" portion
        train, _ = external_raw_split(sample_df, test_ratio=0.2)
        splits = internal_dev_splits(train)
        assert "train_fit" in splits
        assert "val_hpo" in splits
        assert "val_calibration" in splits

    def test_temporal_ordering(self, sample_df: pd.DataFrame):
        train, _ = external_raw_split(sample_df, test_ratio=0.2)
        splits = internal_dev_splits(train)

        fit_max = splits["train_fit"]["tender_datePublished"].max()
        hpo_min = splits["val_hpo"]["tender_datePublished"].min()
        hpo_max = splits["val_hpo"]["tender_datePublished"].max()
        cal_min = splits["val_calibration"]["tender_datePublished"].min()

        assert fit_max <= hpo_min, "train_fit must end before val_hpo starts"
        assert hpo_max <= cal_min, "val_hpo must end before val_calibration starts"

    def test_no_data_lost(self, sample_df: pd.DataFrame):
        train, _ = external_raw_split(sample_df, test_ratio=0.2)
        splits = internal_dev_splits(train)
        total = sum(len(s) for s in splits.values())
        # May lose rows without dates, but our sample has all dates
        assert total == len(train)

    def test_test_data_never_used(self, sample_df: pd.DataFrame):
        """Verify dev splits only come from train data."""
        train, test = external_raw_split(sample_df, test_ratio=0.2)
        splits = internal_dev_splits(train)

        test_ocids = set(test["ocid"])
        for name, split_df in splits.items():
            overlap = set(split_df["ocid"]) & test_ocids
            assert len(overlap) == 0, (
                f"Dev split '{name}' contains {len(overlap)} test_data OCIDs!"
            )


@pytest.mark.p1
class TestDevSplitManifest:
    def test_manifest_written(self, sample_df: pd.DataFrame, tmp_path: Path, monkeypatch):
        import src.split as split_mod

        monkeypatch.setattr(split_mod, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(split_mod, "DEV_SPLIT_MANIFEST", tmp_path / "dev_split_manifest.json")

        train, _ = external_raw_split(sample_df, test_ratio=0.2)
        splits = internal_dev_splits(train)
        save_dev_split_manifest(splits)

        manifest_path = tmp_path / "dev_split_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert "train_fit" in manifest
        assert "val_hpo" in manifest
        assert "val_calibration" in manifest
        assert manifest["train_fit"]["count"] > 0
