import pandas as pd
import pytest

from scripts.link_procurement_evidence import link_procurement_evidence
from src.evidence_linking import apply_match_results_to_label_records, build_evidence_match_table


@pytest.fixture
def procurement_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ocid": "ocds-1",
                "tender_title": "Pengadaan Public Safety Diving Equipment",
                "tender_datePublished": "2022-12-23",
                "buyer_name": "Badan Nasional Pencarian dan Pertolongan",
                "buyer_id": "BASARNAS",
                "supplier_name": "PT. KINDAH ABADI UTAMA",
                "supplier_id": "SUP-1",
                "tender_value_amount": 17450000000,
                "award_value_amount": 17447470000,
            },
            {
                "ocid": "ocds-2",
                "tender_title": "Pengadaan Peralatan Pendeteksi Korban Reruntuhan",
                "tender_datePublished": "2023-01-09",
                "buyer_name": "Badan Nasional Pencarian dan Pertolongan",
                "buyer_id": "BASARNAS",
                "supplier_name": "PT INTERTEKNO GRAFIKASEJATI",
                "supplier_id": "SUP-2",
                "tender_value_amount": 10000000000,
                "award_value_amount": 9997104000,
            },
        ]
    )


@pytest.mark.p1
def test_build_evidence_match_table_prefers_exact_ocid(procurement_rows: pd.DataFrame):
    evidence_rows = pd.DataFrame(
        [
            {
                "source_record_id": "ev-1",
                "source_name": "kpk_procurement_case",
                "label_family": "confirmed_fraud",
                "matched_ocid": "ocds-1",
                "buyer_name": "Badan Nasional Pencarian dan Pertolongan",
                "supplier_name": "PT. KINDAH ABADI UTAMA",
                "package_name": "Pengadaan Public Safety Diving Equipment",
            }
        ]
    )

    matches = build_evidence_match_table(procurement_rows, evidence_rows)

    assert len(matches) == 1
    match = matches.iloc[0].to_dict()
    assert match["ocid"] == "ocds-1"
    assert match["match_type"] == "exact_ocid"
    assert match["match_confidence"] == pytest.approx(1.0)
    assert bool(match["reviewer_needed"]) is False
    assert "ocid_exact" in match["matched_on"]


@pytest.mark.p1
def test_build_evidence_match_table_scores_strong_name_title_value_matches(procurement_rows: pd.DataFrame):
    evidence_rows = pd.DataFrame(
        [
            {
                "source_record_id": "ev-2",
                "source_name": "kpk_procurement_case",
                "label_family": "confirmed_fraud",
                "buyer_name": "Badan Nasional Pencarian dan Pertolongan",
                "supplier_name": "PT Intertekno Grafika Sejati",
                "package_name": "Pengadaan Peralatan Pendeteksi Korban Reruntuhan",
                "package_year": "2023",
                "package_value_amount": 9997000000,
            }
        ]
    )

    matches = build_evidence_match_table(procurement_rows, evidence_rows)

    assert len(matches) == 1
    match = matches.iloc[0].to_dict()
    assert match["ocid"] == "ocds-2"
    assert match["match_type"] == "supplier_buyer_title"
    assert match["match_confidence"] >= 0.9
    assert bool(match["reviewer_needed"]) is False
    assert "supplier_exact" in match["matched_on"]
    assert "buyer_exact" in match["matched_on"]
    assert "title_jaccard" in match["matched_on"]


@pytest.mark.p1
def test_build_evidence_match_table_uses_exact_ids_when_names_are_missing(procurement_rows: pd.DataFrame):
    evidence_rows = pd.DataFrame(
        [
            {
                "source_record_id": "ev-3",
                "source_name": "lkpp_inaproc_blacklist",
                "label_family": "sanctioned_supplier",
                "buyer_id": "BASARNAS",
                "supplier_id": "SUP-2",
            }
        ]
    )

    matches = build_evidence_match_table(procurement_rows, evidence_rows)

    assert len(matches) == 1
    match = matches.iloc[0].to_dict()
    assert match["ocid"] == "ocds-2"
    assert match["match_type"] == "supplier_buyer_id"
    assert match["match_confidence"] >= 0.55
    assert bool(match["reviewer_needed"]) is False
    assert "supplier_id_exact" in match["matched_on"]
    assert "buyer_id_exact" in match["matched_on"]


@pytest.mark.p1
def test_build_evidence_match_table_marks_ambiguous_best_matches_for_review():
    procurement_rows = pd.DataFrame(
        [
            {
                "ocid": "ocds-a",
                "tender_title": "Pengadaan Laptop Pendidikan",
                "tender_datePublished": "2024-01-01",
                "buyer_name": "Dinas Pendidikan Kota X",
                "buyer_id": "BUY-1",
                "supplier_name": "PT Maju Jaya",
                "supplier_id": "SUP-1",
                "tender_value_amount": 100000000,
                "award_value_amount": 99000000,
            },
            {
                "ocid": "ocds-b",
                "tender_title": "Pengadaan Laptop Pendidikan",
                "tender_datePublished": "2024-01-01",
                "buyer_name": "Dinas Pendidikan Kota X",
                "buyer_id": "BUY-1",
                "supplier_name": "PT Maju Jaya",
                "supplier_id": "SUP-1",
                "tender_value_amount": 100000000,
                "award_value_amount": 99000000,
            },
        ]
    )
    evidence_rows = pd.DataFrame(
        [
            {
                "source_record_id": "ev-ambiguous",
                "source_name": "kpk_procurement_case",
                "label_family": "confirmed_fraud",
                "buyer_name": "Dinas Pendidikan Kota X",
                "supplier_name": "PT Maju Jaya",
                "package_name": "Pengadaan Laptop Pendidikan",
                "package_year": "2024",
                "package_value_amount": 99000000,
            }
        ]
    )

    matches = build_evidence_match_table(procurement_rows, evidence_rows)

    assert len(matches) == 1
    match = matches.iloc[0].to_dict()
    assert pd.isna(match["ocid"])
    assert match["match_type"] == "ambiguous_best_match"
    assert match["match_confidence"] >= 0.9
    assert bool(match["reviewer_needed"]) is True
    assert "ambiguous_best_match" in match["matched_on"]


@pytest.mark.p1
def test_apply_match_results_to_label_records_updates_ocid_and_review_flags(procurement_rows: pd.DataFrame):
    evidence_rows = pd.DataFrame(
        [
            {
                "source_record_id": "ev-2",
                "source_name": "kpk_procurement_case",
                "label_family": "confirmed_fraud",
                "buyer_name": "Badan Nasional Pencarian dan Pertolongan",
                "supplier_name": "PT Intertekno Grafika Sejati",
                "package_name": "Pengadaan Peralatan Pendeteksi Korban Reruntuhan",
                "package_year": "2023",
                "package_value_amount": 9997000000,
            }
        ]
    )
    label_rows = pd.DataFrame(
        [
            {
                "ocid": None,
                "label_family": "confirmed_fraud",
                "source_name": "kpk_procurement_case",
                "source_record_id": "ev-2",
                "confidence_score": None,
                "reviewer_needed": True,
            }
        ]
    )

    matches = build_evidence_match_table(procurement_rows, evidence_rows)
    linked_labels = apply_match_results_to_label_records(label_rows, matches)

    assert linked_labels.iloc[0]["ocid"] == "ocds-2"
    assert linked_labels.iloc[0]["confidence_score"] >= 0.9
    assert bool(linked_labels.iloc[0]["reviewer_needed"]) is False


@pytest.mark.p1
def test_link_procurement_evidence_emits_linked_labels_even_without_input_label_file(
    procurement_rows: pd.DataFrame, tmp_path
):
    procurement_path = tmp_path / "procurement.parquet"
    evidence_path = tmp_path / "evidence_records.parquet"
    missing_label_path = tmp_path / "missing_label_records.parquet"
    output_dir = tmp_path / "output"

    procurement_rows.to_parquet(procurement_path, index=False)
    pd.DataFrame(
        [
            {
                "source_record_id": "ev-2",
                "source_name": "kpk_procurement_case",
                "label_family": "confirmed_fraud",
                "label_value": "kpk_final_outcome",
                "source_type": "case_press_release",
                "case_stage": "final_outcome",
                "buyer_name": "Badan Nasional Pencarian dan Pertolongan",
                "supplier_name": "PT Intertekno Grafika Sejati",
                "package_name": "Pengadaan Peralatan Pendeteksi Korban Reruntuhan",
                "package_year": "2023",
                "package_value_amount": 9997000000,
                "source_url": "https://www.kpk.go.id/example",
                "decision_date": "2023-12-20",
            }
        ]
    ).to_parquet(evidence_path, index=False)

    outputs = link_procurement_evidence(
        procurement_path=procurement_path,
        evidence_path=evidence_path,
        label_path=missing_label_path,
        output_dir=output_dir,
    )

    linked_label_path = outputs["linked_label_records"]
    linked_labels = pd.read_parquet(linked_label_path)

    assert linked_label_path.exists()
    assert linked_labels.iloc[0]["ocid"] == "ocds-2"
    assert linked_labels.iloc[0]["source_record_id"] == "ev-2"
    assert linked_labels.iloc[0]["label_family"] == "confirmed_fraud"
    assert linked_labels.iloc[0]["confidence_score"] >= 0.9
    assert bool(linked_labels.iloc[0]["reviewer_needed"]) is False
