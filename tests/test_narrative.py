import pytest

from src.narrative import (
    derive_business_rating,
    render_counterfactuals,
    render_explanation_narrative,
    render_factor_sentence,
)


@pytest.mark.p1
def test_render_factor_sentence_mentions_direction_and_feature():
    sentence = render_factor_sentence(
        {
            "feature": "f_single_bidder",
            "value": 1.0,
            "shap_value": 0.42,
            "direction": "increases_risk",
        }
    )
    assert "peserta tunggal" in sentence
    assert "meningkatkan" in sentence


@pytest.mark.p1
def test_render_explanation_narrative_contains_summary_and_factors():
    narrative = render_explanation_narrative(
        {
            "predicted_class": 2,
            "predicted_label": "High Risk",
            "probability": 0.87,
            "factors": [
                {
                    "feature": "f_num_tenderers",
                    "value": 1.0,
                    "shap_value": 0.55,
                    "direction": "increases_risk",
                }
            ],
        }
    )
    assert "High Risk" in narrative
    assert "87.00%" in narrative
    assert "jumlah peserta tender" in narrative.lower()


@pytest.mark.p1
def test_render_counterfactuals_formats_actionable_output():
    rendered = render_counterfactuals(
        [
            {
                "feature": "f_num_tenderers",
                "suggestion": "tingkatkan jumlah peserta yang valid",
                "impact": 0.33,
            }
        ]
    )
    assert rendered[0].startswith("- Jumlah peserta tender")
    assert "0.3300" in rendered[0]


@pytest.mark.p1
def test_derive_business_rating_maps_model_only_predictions_to_four_level_scale():
    rating = derive_business_rating(
        {
            "predicted_class": 1,
            "probability": 0.62,
        }
    )

    assert rating["rating_label"] == "Perlu Pantauan"
    assert rating["rating_source"] == "model_only"
    assert "triase model" in rating["rating_reason"].lower()


@pytest.mark.p1
def test_derive_business_rating_escalates_to_critical_when_official_evidence_exists():
    rating = derive_business_rating(
        {
            "predicted_class": 2,
            "probability": 0.74,
        },
        evidence_records=[
            {
                "label_family": "confirmed_fraud",
                "case_stage": "final_outcome",
                "reviewer_needed": False,
                "source_name": "kpk_procurement_case",
            }
        ],
    )

    assert rating["rating_label"] == "Risiko Kritis"
    assert rating["rating_source"] == "official_evidence"
    assert "confirmed_fraud" in rating["rating_reason"]


@pytest.mark.p1
def test_render_explanation_narrative_includes_business_rating_and_triage_caveat():
    narrative = render_explanation_narrative(
        {
            "predicted_class": 2,
            "predicted_label": "High Risk",
            "probability": 0.87,
            "factors": [
                {
                    "feature": "f_num_tenderers",
                    "value": 1.0,
                    "shap_value": 0.55,
                    "direction": "increases_risk",
                }
            ],
        }
    )
    assert "Risiko Tinggi" in narrative
    assert "bukan bukti fraud final" in narrative
    assert "triase risiko" in narrative
