from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import scripts.generate_judge_casebook as casebook


def test_generate_judge_casebook_includes_four_level_scale_and_official_case(tmp_path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True)
    linked_labels_path = evidence_dir / "linked_label_records.parquet"
    review_benchmark_path = tmp_path / "review_benchmark_500.csv"
    output_path = tmp_path / "judge_casebook.md"
    flat_path = tmp_path / "ocds_flat.parquet"
    evidence_summary_path = evidence_dir / "evidence_match_summary.json"

    json.dump(
        {
            "total_evidence_rows": 2,
            "matched_rows": 1,
            "needs_review_rows": 1,
            "min_confidence": 0.55,
        },
        evidence_summary_path.open("w", encoding="utf-8"),
    )

    pd.DataFrame(
        [
            {
                "ocid": "ocds-1",
                "label_family": "confirmed_fraud",
                "source_name": "kpk_procurement_case",
                "source_record_id": "kpk-case-1",
                "case_stage": "final_outcome",
                "decision_date": "2023-12-20",
                "confidence_score": 1.0,
                "reviewer_needed": False,
                "package_name": "Pengadaan Public Safety Diving Equipment",
                "provenance_note": "Official KPK case",
            },
            {
                "ocid": "ocds-1",
                "label_family": "confirmed_fraud",
                "source_name": "kpk_ppid_report",
                "source_record_id": "kpk-ppid-1",
                "case_stage": "final_outcome",
                "decision_date": "2023-12-21",
                "confidence_score": 1.0,
                "reviewer_needed": False,
                "package_name": "Pengadaan Public Safety Diving Equipment",
                "provenance_note": "Official PPID case report",
            },
            {
                "ocid": None,
                "label_family": "sanctioned_supplier",
                "source_name": "lkpp_inaproc_blacklist",
                "source_record_id": "lkpp-1",
                "case_stage": "administrative_sanction",
                "decision_date": "2024-03-01",
                "confidence_score": 0.42,
                "reviewer_needed": True,
                "package_name": "Pembangunan Gedung",
                "provenance_note": "Needs review",
            },
        ]
    ).to_parquet(linked_labels_path, index=False)

    pd.DataFrame(
        [
            {
                "ocid": "ocds-1",
                "tender_title": "Pengadaan Public Safety Diving Equipment",
                "buyer_name": "Basarnas",
                "supplier_name": "PT KINDAH ABADI UTAMA",
                "predicted_label_name": "High Risk",
                "predicted_probability": 0.91,
                "business_rating_label": "Risiko Kritis",
                "business_rating_source": "official_evidence",
                "review_priority_score": 0.88,
                "sampling_reason": "official_evidence_linked",
                "top_factors": "f_title_length=2.1",
                "narrative_id": "Peringkat investigatif paket ini adalah **Risiko Kritis**.",
                "evidence_label_families": "confirmed_fraud",
            },
            {
                "ocid": "ocds-2",
                "tender_title": "Pembangunan Drainase",
                "buyer_name": "Pemkab Contoh",
                "supplier_name": "CV Contoh",
                "predicted_label_name": "High Risk",
                "predicted_probability": 0.53,
                "business_rating_label": "Risiko Tinggi",
                "business_rating_source": "model_only",
                "review_priority_score": 0.74,
                "sampling_reason": "priority_mix",
                "top_factors": "f_buyer_supplier_repeat_count=3.2",
                "narrative_id": "Peringkat investigatif paket ini adalah **Risiko Tinggi**.",
                "evidence_label_families": "",
            },
        ]
    ).to_csv(review_benchmark_path, index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {
                "ocid": "ocds-1",
                "tender_title": "Pengadaan Public Safety Diving Equipment",
                "tender_datePublished": "2023-09-01",
                "buyer_name": "Basarnas",
                "supplier_name": "PT KINDAH ABADI UTAMA",
                "tender_value_amount": 17499969180.0,
                "award_value_amount": 17447468400.0,
            }
        ]
    ).to_parquet(flat_path, index=False)

    original_summary = casebook.EVIDENCE_SUMMARY_PATH
    original_linked = casebook.LINKED_LABELS_PATH
    original_review = casebook.REVIEW_BENCHMARK_PATH
    original_output = casebook.OUTPUT_PATH
    original_flat = casebook.FLAT_PARQUET
    try:
        casebook.EVIDENCE_SUMMARY_PATH = evidence_summary_path
        casebook.LINKED_LABELS_PATH = linked_labels_path
        casebook.REVIEW_BENCHMARK_PATH = review_benchmark_path
        casebook.OUTPUT_PATH = output_path
        casebook.FLAT_PARQUET = flat_path

        generated = casebook.generate_judge_casebook(output_path=output_path)
        content = generated.read_text(encoding="utf-8")
    finally:
        casebook.EVIDENCE_SUMMARY_PATH = original_summary
        casebook.LINKED_LABELS_PATH = original_linked
        casebook.REVIEW_BENCHMARK_PATH = original_review
        casebook.OUTPUT_PATH = original_output
        casebook.FLAT_PARQUET = original_flat

    assert generated == output_path
    assert "Aman" in content
    assert "Perlu Pantauan" in content
    assert "Risiko Tinggi" in content
    assert "Risiko Kritis" in content
    assert "Official Evidence-Linked Cases" in content
    assert "confirmed_fraud" in content
    assert "Basarnas" in content
    assert "Supporting evidence rows: 2" in content
    assert "Official source count: 2" in content
    assert "reviewer_needed=True" in content
    assert "Judge Demo Archetypes" in content
    assert "Archetype 1" in content
    assert "official evidence-linked critical case" in content.lower()
    assert "model-only high-risk triage case" in content.lower()
