import pytest

from src.narrative import (
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
