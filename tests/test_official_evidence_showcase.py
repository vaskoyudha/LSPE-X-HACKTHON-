from __future__ import annotations

import pandas as pd

import scripts.generate_official_evidence_showcase as showcase


def test_build_showcase_rows_merges_official_groups_with_split_rows(monkeypatch) -> None:
    official_groups = {
        "ocds-1": [
            {
                "label_family": "confirmed_fraud",
                "source_name": "kpk_procurement_case",
                "case_stage": "final_outcome",
                "confidence_score": 1.0,
            }
        ]
    }
    split_artifacts = {
        "train": (
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
            ),
            pd.DataFrame(
                [
                    {
                        "f_tender_value_log": 1.0,
                        "f_buyer_supplier_repeat_count": 2.0,
                    }
                ]
            ),
        ),
        "test": (
            pd.DataFrame(columns=showcase.RAW_COLUMNS),
            pd.DataFrame(columns=["f_tender_value_log", "f_buyer_supplier_repeat_count"]),
        ),
    }

    monkeypatch.setattr(
        showcase,
        "_score_case",
        lambda model, explainer, calibration, thresholds, feature_row, evidence_records: {
            "predicted_label_name": "High Risk",
            "predicted_probability": 0.91,
            "prob_low": 0.04,
            "prob_medium": 0.05,
            "prob_high": 0.91,
            "business_rating_label": "Risiko Kritis",
            "business_rating_source": "official_evidence",
            "business_rating_reason": "Evidence escalation",
            "top_factors": "f_tender_value_log=2.3",
            "narrative_id": "Peringkat investigatif paket ini adalah **Risiko Kritis**.",
        },
    )

    showcase_df = showcase._build_showcase_rows(
        official_groups,
        split_artifacts,
        model=object(),
        explainer=object(),
        calibration=None,
        thresholds={"high_risk": 0.5, "low_risk": 0.5},
    )

    assert len(showcase_df) == 1
    row = showcase_df.iloc[0]
    assert row["ocid"] == "ocds-1"
    assert row["source_partition"] == "train"
    assert row["buyer_name"] == "Basarnas"
    assert row["predicted_label_name"] == "High Risk"
    assert row["business_rating_label"] == "Risiko Kritis"
    assert row["evidence_label_families"] == "confirmed_fraud"
    assert row["evidence_record_count"] == 1
    assert row["evidence_source_count"] == 1


def test_build_showcase_markdown_reports_evidence_override_cases() -> None:
    showcase_df = pd.DataFrame(
        [
            {
                "ocid": "ocds-1",
                "source_partition": "train",
                "tender_title": "Case A",
                "buyer_name": "Buyer A",
                "supplier_name": "Supplier A",
                "evidence_label_families": "confirmed_fraud",
                "evidence_sources": "kpk_procurement_case",
                "evidence_case_stages": "final_outcome",
                "evidence_record_count": 2,
                "evidence_source_count": 2,
                "predicted_label_name": "Medium Risk",
                "predicted_probability": 0.61,
                "prob_low": 0.1,
                "prob_medium": 0.61,
                "prob_high": 0.29,
                "business_rating_label": "Risiko Kritis",
                "business_rating_source": "official_evidence",
                "business_rating_reason": "Evidence escalation",
                "top_factors": "f_title_length=1.2",
                "narrative_id": "Narrative A",
            },
            {
                "ocid": "ocds-2",
                "source_partition": "train",
                "tender_title": "Case B",
                "buyer_name": "Buyer B",
                "supplier_name": "Supplier B",
                "evidence_label_families": "confirmed_fraud",
                "evidence_sources": "kpk_procurement_case",
                "evidence_case_stages": "final_outcome",
                "evidence_record_count": 1,
                "evidence_source_count": 1,
                "predicted_label_name": "High Risk",
                "predicted_probability": 0.82,
                "prob_low": 0.08,
                "prob_medium": 0.1,
                "prob_high": 0.82,
                "business_rating_label": "Risiko Kritis",
                "business_rating_source": "official_evidence",
                "business_rating_reason": "Evidence escalation",
                "top_factors": "f_repeat=3.4",
                "narrative_id": "Narrative B",
            },
        ]
    )

    markdown = showcase._build_showcase_markdown(showcase_df)

    assert "Official evidence-linked cases found in current split artifacts: 2" in markdown
    assert "Supporting official evidence rows behind those cases: 3" in markdown
    assert "Cases with multi-source official corroboration: 1" in markdown
    assert "Model predicted High Risk: 1" in markdown
    assert "Model predicted Medium Risk: 1" in markdown
    assert "Cases where evidence lane corrected a non-High-Risk model output: 1" in markdown
    assert "Case 1: ocds-1" in markdown
    assert "Supporting evidence rows: 2" in markdown
    assert "Official source count: 2" in markdown
    assert "Final business rating: Risiko Kritis [official_evidence]" in markdown
