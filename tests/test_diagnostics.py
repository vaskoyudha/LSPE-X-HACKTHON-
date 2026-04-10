import numpy as np
from pathlib import Path
import pandas as pd
import pytest

from src.diagnostics import (
    PROXY_BROAD_FEATURES,
    PROXY_CORE_FEATURES,
    build_explanation_validation_from_summary,
    build_reviewed_subset_metrics_from_summary,
    compute_operational_review_metrics,
    load_canonical_reviewed_labels,
    load_confirmed_outcome_labels,
    load_metrics_artifact,
    load_reviewed_labels,
    load_manual_review_summary,
    load_row_level_reviewed_benchmark,
    resolve_proxy_feature_sets,
    select_reviewed_rows,
    summarize_confirmed_outcome_alignment,
    summarize_explanation_validation,
    summarize_evidence_label_coverage,
    summarize_evaluation_lanes,
    summarize_feature_health,
    summarize_feature_health_overview,
    summarize_data_provenance,
)


@pytest.mark.p1
def test_resolve_proxy_feature_sets_removes_expected_columns():
    features = [
        "f_title_length",
        "f_buyer_supplier_repeat_count",
        "f_supplier_hist_avg_award",
    ]
    resolved = resolve_proxy_feature_sets(features)
    assert resolved["full"] == features
    assert "f_title_length" not in resolved["proxy_core_removed"]
    assert "f_buyer_supplier_repeat_count" not in resolved["proxy_core_removed"]
    assert "f_supplier_hist_avg_award" in resolved["proxy_core_removed"]
    assert set(PROXY_CORE_FEATURES).issuperset(
        {"f_title_length", "f_buyer_supplier_repeat_count"}
    )
    assert set(PROXY_BROAD_FEATURES).issuperset(set(PROXY_CORE_FEATURES))


@pytest.mark.p1
def test_summarize_data_provenance_flags_synthetic_prefixes():
    train_raw = pd.DataFrame(
        {
            "ocid": ["ocds-synth-000001", "ocds-synth-000002"],
            "tender_datePublished": pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True),
            "buyer_id": ["b1", "b2"],
            "supplier_id": ["s1", "s2"],
            "tender_procurementMethod": ["open", "direct"],
        }
    )
    test_raw = pd.DataFrame(
        {
            "ocid": ["ocds-synth-000003"],
            "tender_datePublished": pd.to_datetime(["2020-02-01"], utc=True),
            "buyer_id": ["b1"],
            "supplier_id": ["s3"],
            "tender_procurementMethod": ["limited"],
        }
    )
    summary = summarize_data_provenance(train_raw, test_raw)
    assert summary["all_ocids_use_synthetic_prefix"] is True
    assert summary["data_kind"] == "synthetic_structured_benchmark"
    assert summary["row_count_total"] == 3


@pytest.mark.p1
def test_feature_health_marks_constant_and_all_nan_features():
    df = pd.DataFrame(
        {
            "f_live": [1.0, 2.0, 3.0],
            "f_dead_nan": [np.nan, np.nan, np.nan],
            "f_dead_constant": [0.0, 0.0, 0.0],
        }
    )

    report = summarize_feature_health(df)

    assert report["f_dead_nan"]["all_nan"] is True
    assert report["f_dead_constant"]["constant"] is True
    assert report["f_live"]["constant"] is False
    assert report["f_live"]["missing_pct"] == 0.0


@pytest.mark.p1
def test_feature_health_overview_tracks_removed_dead_slots():
    report = {
        "f_live": {"all_nan": False, "constant": False},
        "f_dead_constant": {"all_nan": False, "constant": True},
    }

    overview = summarize_feature_health_overview(
        report,
        retired_features=["f_old_dead", "f_dead_constant"],
    )

    assert overview["active_dead_feature_count"] == 1
    assert overview["active_dead_features"] == ["f_dead_constant"]
    assert overview["retired_dead_features_present"] == ["f_dead_constant"]
    assert overview["retired_dead_features_removed"] == ["f_old_dead"]


@pytest.mark.p1
def test_real_calibration_artifacts_are_not_synthetic():
    processed = Path("data/processed")
    sheet = pd.read_csv(processed / "calibration_sheet_100.csv")
    clean = pd.read_csv(processed / "clean_labels_100.csv")
    assert float(sheet["ocid"].astype(str).str.startswith("ocds-synth-").mean()) == 0.0
    assert float(clean["ocid"].astype(str).str.startswith("ocds-synth-").mean()) == 0.0


@pytest.mark.p1
def test_compute_operational_review_metrics_reports_precision_and_recall():
    probs = np.array(
        [
            [0.1, 0.2, 0.7],
            [0.2, 0.3, 0.5],
            [0.8, 0.1, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    y_true = pd.Series([2, 1, 0, 2])
    metrics = compute_operational_review_metrics(probs, y_true, budgets=[1, 2])
    assert metrics["total_positive"] == 2
    assert metrics["budgets"]["1"]["precision_at_k"] == 1.0
    assert metrics["budgets"]["2"]["recall_at_k"] == 1.0


@pytest.mark.p1
def test_select_reviewed_rows_prefers_source_row_idx_alignment():
    reviewed = pd.DataFrame(
        {
            "source_row_idx": [2, 0],
            "reviewed_label": [1, 2],
        }
    )
    raw = pd.DataFrame({"ocid": ["a", "b", "c"]})
    features = pd.DataFrame({"f_x": [10, 20, 30]})
    raw_subset, feature_subset, labels = select_reviewed_rows(reviewed, raw, features)
    assert raw_subset["ocid"].tolist() == ["c", "a"]
    assert feature_subset["f_x"].tolist() == [30, 10]
    assert labels.tolist() == [1, 2]


@pytest.mark.p1
def test_load_reviewed_labels_filters_invalid_rows(tmp_path):
    path = tmp_path / "review.csv"
    pd.DataFrame(
        {
            "source_row_idx": [0, 1, 2],
            "reviewed_label": ["2", "bad", ""],
            "explanation_agrees": ["yes", "", ""],
        }
    ).to_csv(path, index=False)
    reviewed = load_reviewed_labels(path)
    assert reviewed["reviewed_label"].tolist() == [2]
    assert reviewed["source_row_idx"].tolist() == [0]


@pytest.mark.p1
def test_summarize_explanation_validation_uses_filled_review_fields():
    review_df = pd.DataFrame(
        {
            "reviewed_label": [2, 1],
            "explanation_agrees": ["yes", "no"],
            "explanation_actionable": ["yes", "yes"],
            "explanation_clarity": [4, 5],
        }
    )
    summary = summarize_explanation_validation(review_df)
    assert summary["status"] == "available"
    assert summary["agreement_rate"] == 0.5
    assert summary["actionable_rate"] == 1.0
    assert summary["clarity_mean"] == 4.5


@pytest.mark.p1
def test_build_reviewed_subset_metrics_from_summary_uses_confusion_counts():
    summary = pd.DataFrame(
        [
            {"section": "2. Agreement", "dimension": "overall", "metric": "agree_count", "value": 4, "pct": "80%"},
            {"section": "3. Confusion Matrix", "dimension": "predicted_0_reviewed_0", "metric": "count", "value": 1},
            {"section": "3. Confusion Matrix", "dimension": "predicted_0_reviewed_1", "metric": "count", "value": 0},
            {"section": "3. Confusion Matrix", "dimension": "predicted_0_reviewed_2", "metric": "count", "value": 0},
            {"section": "3. Confusion Matrix", "dimension": "predicted_1_reviewed_0", "metric": "count", "value": 0},
            {"section": "3. Confusion Matrix", "dimension": "predicted_1_reviewed_1", "metric": "count", "value": 1},
            {"section": "3. Confusion Matrix", "dimension": "predicted_1_reviewed_2", "metric": "count", "value": 1},
            {"section": "3. Confusion Matrix", "dimension": "predicted_2_reviewed_0", "metric": "count", "value": 0},
            {"section": "3. Confusion Matrix", "dimension": "predicted_2_reviewed_1", "metric": "count", "value": 0},
            {"section": "3. Confusion Matrix", "dimension": "predicted_2_reviewed_2", "metric": "count", "value": 2},
        ]
    )
    metrics = build_reviewed_subset_metrics_from_summary(summary)
    assert metrics["status"] == "available"
    assert metrics["reviewed_rows"] == 5
    assert metrics["accuracy"] == 0.8
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 2]]


@pytest.mark.p1
def test_build_explanation_validation_from_summary_reads_summary_scores():
    summary = pd.DataFrame(
        [
            {"section": "6. Explanation Quality", "dimension": "explanation_agrees", "metric": "yes", "value": 8},
            {"section": "6. Explanation Quality", "dimension": "explanation_agrees", "metric": "partial", "value": 2},
            {"section": "6. Explanation Quality", "dimension": "explanation_agrees", "metric": "no", "value": 0},
            {"section": "6. Explanation Quality", "dimension": "explanation_clarity", "metric": "mean", "value": 3.5},
            {"section": "6. Explanation Quality", "dimension": "explanation_clarity", "metric": "median", "value": 4.0},
            {"section": "6. Explanation Quality", "dimension": "explanation_actionable", "metric": "mean", "value": 4.2},
            {"section": "6. Explanation Quality", "dimension": "explanation_actionable", "metric": "median", "value": 4.0},
            {"section": "5. By Sampling Group", "dimension": "high_uncertainty", "metric": "avg_explanation_clarity", "value": 2.0},
            {"section": "7. Top Factors", "dimension": "f_is_q4", "metric": "frequency", "value": 10, "pct": "10%", "notes": "demo"},
        ]
    )
    metrics = build_explanation_validation_from_summary(summary)
    assert metrics["status"] == "available"
    assert metrics["agreement_yes_rate"] == 0.8
    assert metrics["clarity_mean"] == 3.5
    assert metrics["actionable_mean"] == 4.2
    assert metrics["by_sampling_group"]["high_uncertainty"]["avg_explanation_clarity"] == 2
    assert metrics["top_factors"][0]["feature"] == "f_is_q4"


@pytest.mark.p1
def test_load_manual_review_summary_reads_csv(tmp_path):
    path = tmp_path / "manual_review_summary.csv"
    pd.DataFrame(
        [{"section": "x", "dimension": "y", "metric": "z", "value": 1}]
    ).to_csv(path, index=False)
    loaded = load_manual_review_summary(path)
    assert len(loaded) == 1


@pytest.mark.p1
def test_load_row_level_reviewed_benchmark_filters_invalid_rows(tmp_path):
    path = tmp_path / "reviewed.csv"
    pd.DataFrame(
        [
            {"source_row_idx": 0, "reviewed_label": 2},
            {"source_row_idx": "bad", "reviewed_label": 1},
            {"source_row_idx": 2, "reviewed_label": 9},
        ]
    ).to_csv(path, index=False)
    loaded = load_row_level_reviewed_benchmark(path)
    assert loaded["source_row_idx"].tolist() == [0]
    assert loaded["reviewed_label"].tolist() == [2]


@pytest.mark.p1
def test_summarize_evidence_label_coverage_counts_rows_per_family():
    reviewed = pd.DataFrame(
        [{"ocid": "ocds-1", "label_family": "reviewed_risk", "label_value": "high"}]
    )
    outcomes = pd.DataFrame(
        [{"ocid": "ocds-2", "label_family": "confirmed_fraud", "label_value": "fraud"}]
    )

    summary = summarize_evidence_label_coverage(reviewed, outcomes)
    assert summary["reviewed_rows"] == 1
    assert summary["confirmed_outcome_rows"] == 1
    assert summary["families"] == {
        "reviewed_risk": 1,
        "confirmed_fraud": 1,
    }


@pytest.mark.p1
def test_load_canonical_reviewed_labels_validates_parquet(tmp_path):
    path = tmp_path / "reviewed.parquet"
    pd.DataFrame(
        [
            {
                "ocid": "ocds-1",
                "label_family": "reviewed_risk",
                "label_value": "high",
                "source_name": "manual_review_round_1",
                "source_type": "human_review",
                "source_record_id": "row-1",
                "decision_date": "2026-04-10",
                "confidence_score": 0.9,
                "review_notes": "",
                "reviewer_id": "",
                "ingested_at": "2026-04-10T12:00:00Z",
            }
        ]
    ).to_parquet(path, index=False)
    loaded = load_canonical_reviewed_labels(path)
    assert loaded["label_family"].tolist() == ["reviewed_risk"]


@pytest.mark.p1
def test_load_confirmed_outcome_labels_validates_parquet(tmp_path):
    path = tmp_path / "outcomes.parquet"
    pd.DataFrame(
        [
            {
                "ocid": "ocds-2",
                "label_family": "confirmed_fraud",
                "label_value": "fraud",
                "source_name": "court_fixture",
                "source_type": "court",
                "source_record_id": "case-1",
                "decision_date": "2025-10-03",
                "confidence_score": 1.0,
                "review_notes": "",
                "reviewer_id": "",
                "ingested_at": "2026-04-10T12:00:00Z",
            }
        ]
    ).to_parquet(path, index=False)
    loaded = load_confirmed_outcome_labels(path)
    assert loaded["label_family"].tolist() == ["confirmed_fraud"]


@pytest.mark.p1
def test_summarize_confirmed_outcome_alignment_reports_descriptive_stats():
    outcomes = pd.DataFrame(
        [
            {
                "ocid": "ocds-1",
                "label_family": "confirmed_fraud",
                "label_value": "fraud",
            },
            {
                "ocid": "ocds-missing",
                "label_family": "confirmed_irregularity",
                "label_value": "irregularity",
            },
        ]
    )
    raw = pd.DataFrame({"ocid": ["ocds-1", "ocds-2"]})
    probs = np.array([[0.05, 0.1, 0.85], [0.7, 0.2, 0.1]])
    preds = np.array([2, 0])

    summary = summarize_confirmed_outcome_alignment(outcomes, raw, probs, preds)

    assert summary["status"] == "descriptive_only"
    assert summary["matched_rows"] == 1
    assert summary["unmatched_rows"] == 1
    assert summary["predicted_label_distribution"] == {"High Risk": 1}
    assert summary["label_family_distribution"] == {"confirmed_fraud": 1}
    assert summary["high_risk_probability_mean"] == 0.85


@pytest.mark.p1
def test_summarize_evaluation_lanes_separates_families():
    heuristic_metrics = {
        "final_test_thresholded": {
            "partition": "test_thresholded",
            "label_type": "heuristic_risk_labels",
            "n_samples": 10,
            "accuracy": 0.9,
            "macro_f1": 0.8,
            "weighted_f1": 0.85,
        }
    }
    reviewed_metrics = {
        "status": "available",
        "source": "row_level_reviewed_benchmark",
        "reviewed_rows": 2,
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "weighted_f1": 1.0,
    }
    explanation_validation = {
        "status": "available",
        "agreement_rate": 1.0,
        "clarity_mean": 4.5,
    }
    evidence_coverage = {"reviewed_rows": 2, "confirmed_outcome_rows": 1}
    confirmed_alignment = {
        "status": "descriptive_only",
        "matched_rows": 1,
        "matched_unique_ocids": 1,
        "predicted_label_distribution": {"High Risk": 1},
        "label_family_distribution": {"confirmed_fraud": 1},
        "high_risk_probability_mean": 0.9,
        "message": "descriptive only",
    }

    summary = summarize_evaluation_lanes(
        heuristic_metrics,
        reviewed_metrics,
        explanation_validation,
        evidence_coverage,
        confirmed_alignment,
    )

    assert summary["heuristic_risk_lane"]["label_family"] == "heuristic_risk"
    assert summary["reviewed_risk_lane"]["label_family"] == "reviewed_risk"
    assert summary["confirmed_outcome_lane"]["label_family"] == "confirmed_outcome"
    assert summary["lane_separation_checks"]["heuristic_lane_uses_heuristic_metrics"] is True
    assert summary["lane_separation_checks"]["confirmed_outcomes_descriptive_only"] is True
