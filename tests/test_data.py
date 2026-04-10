"""Tests for src.data module."""

import json
import gzip
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data import (
    _flatten_release,
    _gzip_looks_readable,
    _safe_get,
    clean_dates,
    generate_quality_report,
    flatten_jsonl_gz,
    REQUIRED_FIELDS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RECORD = {
    "ocid": "ocds-afzrfb-s-2020-12345",
    "date": "2020-06-15T10:00:00Z",
    "buyer": {"id": "buyer-001", "name": "Dinas Kesehatan Kota Bandung"},
    "tender": {
        "id": "T-001",
        "title": "Pengadaan Alat Kesehatan Tahun 2020",
        "description": "Pengadaan alat kesehatan untuk Puskesmas di wilayah Kota Bandung tahun anggaran 2020",
        "status": "complete",
        "procurementMethod": "open",
        "value": {"amount": 500000000, "currency": "IDR"},
        "tenderPeriod": {
            "startDate": "2020-06-01T00:00:00Z",
            "endDate": "2020-06-30T00:00:00Z",
        },
        "numberOfTenderers": 5,
    },
    "awards": [
        {
            "id": "A-001",
            "status": "active",
            "date": "2020-07-15T00:00:00Z",
            "value": {"amount": 450000000, "currency": "IDR"},
            "suppliers": [
                {"id": "supplier-001", "name": "PT Medika Sejahtera"}
            ],
        }
    ],
    "contracts": [
        {
            "id": "C-001",
            "awardID": "A-001",
            "value": {"amount": 450000000},
            "dateSigned": "2020-08-01T00:00:00Z",
        }
    ],
}


@pytest.fixture
def sample_jsonl_gz(tmp_path: Path) -> Path:
    """Create a temporary .jsonl.gz with sample records."""
    path = tmp_path / "test.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps(SAMPLE_RECORD) + "\n")
        # Record with no awards
        no_award = {
            "ocid": "ocds-afzrfb-s-2020-99999",
            "tender": {"id": "T-999", "title": "Test"},
            "buyer": {},
        }
        f.write(json.dumps(no_award) + "\n")
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.p0
class TestSafeGet:
    def test_nested_access(self):
        d = {"a": {"b": {"c": 42}}}
        assert _safe_get(d, "a", "b", "c") == 42

    def test_missing_key(self):
        d = {"a": {"b": 1}}
        assert _safe_get(d, "a", "x", default="N/A") == "N/A"

    def test_empty_dict(self):
        assert _safe_get({}, "a", default=None) is None


@pytest.mark.p1
def test_gzip_integrity_check_detects_truncated_file(tmp_path: Path):
    good = tmp_path / "good.jsonl.gz"
    with gzip.open(good, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(SAMPLE_RECORD) + "\n")
    assert _gzip_looks_readable(good) is True

    broken = tmp_path / "broken.jsonl.gz"
    broken.write_bytes(good.read_bytes()[:-8])
    assert _gzip_looks_readable(broken) is False


@pytest.mark.p0
class TestFlattenRelease:
    def test_single_award_record(self):
        rows = _flatten_release(SAMPLE_RECORD)
        assert len(rows) == 1
        row = rows[0]
        assert row["ocid"] == "ocds-afzrfb-s-2020-12345"
        assert row["tender_value_amount"] == 500000000
        assert row["award_value_amount"] == 450000000
        assert row["supplier_name"] == "PT Medika Sejahtera"
        assert row["contract_value_amount"] == 450000000
        assert row["buyer_name"] == "Dinas Kesehatan Kota Bandung"

    def test_no_awards(self):
        record = {
            "ocid": "ocds-test-001",
            "tender": {"id": "T-1", "title": "Test"},
            "buyer": {},
        }
        rows = _flatten_release(record)
        assert len(rows) == 1
        assert rows[0]["award_id"] is None
        assert rows[0]["supplier_id"] is None

    def test_multiple_awards(self):
        record = SAMPLE_RECORD.copy()
        record["awards"] = [
            {
                "id": "A-1",
                "status": "active",
                "date": "2020-07-15",
                "value": {"amount": 100},
                "suppliers": [{"id": "s1", "name": "Supplier 1"}],
            },
            {
                "id": "A-2",
                "status": "active",
                "date": "2020-07-16",
                "value": {"amount": 200},
                "suppliers": [{"id": "s2", "name": "Supplier 2"}],
            },
        ]
        rows = _flatten_release(record)
        assert len(rows) == 2
        assert rows[0]["supplier_name"] == "Supplier 1"
        assert rows[1]["supplier_name"] == "Supplier 2"

    def test_falls_back_to_min_value_and_title_description(self):
        record = {
            "ocid": "ocds-test-realish-001",
            "buyer": {"id": "buyer-1", "name": ""},
            "tender": {
                "id": "T-REAL-1",
                "title": "Pengadaan Alat Laboratorium",
                "description": None,
                "value": {"currency": "IDR"},
                "minValue": {"amount": 125000000, "currency": "IDR"},
                "procuringEntity": {"name": "Kementerian Contoh"},
            },
            "awards": [
                {
                    "id": "A-REAL-1",
                    "value": {"amount": 118000000, "currency": "IDR"},
                    "suppliers": [{"id": "s-real", "name": "PT Contoh"}],
                }
            ],
        }
        row = _flatten_release(record)[0]
        assert row["tender_value_amount"] == 125000000
        assert row["tender_value_currency"] == "IDR"
        assert row["tender_description"] == "Pengadaan Alat Laboratorium"
        assert row["buyer_name"] == "Kementerian Contoh"

    def test_extracts_category_and_item_counts_from_realish_record(self):
        record = {
            "ocid": "ocds-test-real-001",
            "buyer": {"id": "buyer-1", "name": "Kementerian Contoh"},
            "tender": {
                "id": "T-REAL-1",
                "title": "Pengadaan Jasa Konsultansi",
                "description": None,
                "value": {"currency": "IDR"},
                "minValue": {"amount": 125000000, "currency": "IDR"},
                "mainProcurementCategory": "services",
                "items": [{"id": "1"}, {"id": "2"}],
            },
            "awards": [
                {
                    "id": "A-REAL-1",
                    "value": {"amount": 118000000, "currency": "IDR"},
                    "items": [{"id": "1"}],
                    "suppliers": [{"id": "sup-1", "name": "PT Contoh"}],
                }
            ],
        }
        row = _flatten_release(record)[0]
        assert row["tender_value_amount"] == 125000000
        assert row["tender_mainProcurementCategory"] == "services"
        assert row["tender_items_count"] == 2
        assert row["award_items_count"] == 1


@pytest.mark.p1
class TestFlattenJsonlGz:
    def test_reads_sample(self, sample_jsonl_gz: Path):
        df = flatten_jsonl_gz(sample_jsonl_gz)
        assert len(df) == 2  # one award + one no-award
        assert "ocid" in df.columns
        assert df.iloc[0]["tender_value_amount"] == 500000000


@pytest.mark.p1
class TestCleanDates:
    def test_filters_future_dates(self):
        df = pd.DataFrame(
            {
                "tender_tenderPeriod_startDate": [
                    "2020-01-01T00:00:00Z",
                    "3020-01-01T00:00:00Z",  # typo from OCP data
                ],
            }
        )
        cleaned = clean_dates(df)
        assert cleaned["tender_tenderPeriod_startDate"].iloc[0] is not pd.NaT
        assert pd.isna(cleaned["tender_tenderPeriod_startDate"].iloc[1])

    def test_filters_ancient_dates(self):
        df = pd.DataFrame(
            {
                "award_date": [
                    "1970-01-01T00:00:00Z",  # impossible for procurement
                    "2021-05-01T00:00:00Z",
                ],
            }
        )
        cleaned = clean_dates(df)
        assert pd.isna(cleaned["award_date"].iloc[0])
        assert cleaned["award_date"].iloc[1] is not pd.NaT


@pytest.mark.p1
class TestQualityReport:
    def test_generates_markdown(self, sample_jsonl_gz: Path, tmp_path: Path):
        df = flatten_jsonl_gz(sample_jsonl_gz)
        df = clean_dates(df)
        output = tmp_path / "report.md"
        report = generate_quality_report(df, output)
        assert "# Data Quality Report" in report
        assert "Total rows" in report
        assert "Field Coverage" in report
        assert output.exists()
