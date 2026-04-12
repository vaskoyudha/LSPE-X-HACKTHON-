import pytest

from src.evidence import (
    VALID_LABEL_FAMILIES,
    evidence_to_label_record,
    normalize_evidence_record,
)


@pytest.mark.p1
class TestEvidenceNormalization:
    def test_normalize_evidence_record_coerces_core_fields(self):
        record = {
            "source_record_id": 123,
            "source_name": "putusan-ma",
            "source_type": "court_decision",
            "label_family": "confirmed_fraud",
            "label_value": "confirmed_true",
            "case_stage": "final_outcome",
            "decision_date": "2024-01-05",
            "source_url": "https://example.test/case/123",
            "match_confidence": "0.85",
        }

        normalized = normalize_evidence_record(record)

        assert normalized["source_record_id"] == "123"
        assert normalized["label_family"] == "confirmed_fraud"
        assert normalized["source_type"] == "court_decision"
        assert normalized["match_confidence"] == pytest.approx(0.85)
        assert normalized["decision_date"] == "2024-01-05"

    def test_invalid_label_family_raises(self):
        with pytest.raises(ValueError):
            normalize_evidence_record(
                {
                    "source_record_id": "bad-1",
                    "source_name": "bad-source",
                    "source_type": "court_decision",
                    "label_family": "totally_fake_family",
                }
            )

    def test_evidence_to_label_record_preserves_provenance(self):
        evidence = normalize_evidence_record(
            {
                "source_record_id": "ev-1",
                "source_name": "bpk-audit",
                "source_type": "audit_report",
                "label_family": "confirmed_irregularity",
                "label_value": "audit_finding",
                "case_stage": "audit_finding",
                "source_url": "https://example.test/audit/1",
                "decision_date": "2023-06-01",
                "provenance_note": "Linked by buyer, title, and date window",
            }
        )

        label = evidence_to_label_record(
            evidence,
            ocid="ocds-123",
            confidence_score=0.91,
            reviewer_needed=True,
        )

        assert label["ocid"] == "ocds-123"
        assert label["label_family"] == "confirmed_irregularity"
        assert label["source_record_id"] == "ev-1"
        assert label["confidence_score"] == pytest.approx(0.91)
        assert label["reviewer_needed"] is True
        assert "Linked by buyer" in label["provenance_note"]

    def test_normalize_evidence_record_parses_localized_numeric_strings(self):
        normalized = normalize_evidence_record(
            {
                "source_record_id": "ev-localized-numeric",
                "source_name": "lkpp-inaproc",
                "source_type": "sanction_list",
                "label_family": "sanctioned_supplier",
                "case_stage": "administrative_sanction",
                "package_value_amount": "9.999.012.000",
            }
        )

        assert normalized["package_value_amount"] == pytest.approx(9999012000.0)


@pytest.mark.p1
def test_valid_label_families_include_fraud_and_review_lanes():
    assert {
        "confirmed_fraud",
        "confirmed_irregularity",
        "sanctioned_supplier",
        "reviewed_risk",
        "candidate_review_queue",
        "unlabeled",
    }.issubset(VALID_LABEL_FAMILIES)
